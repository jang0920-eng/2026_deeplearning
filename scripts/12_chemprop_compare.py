"""
12. Chemprop D-MPNN 을 2단계 활성 예측에 추가한다 — 지문/트랜스포머/그래프 3계열 비교.

05 단계(Morgan, ChemBERTa)와 **같은 scaffold split, 같은 평가 함수**(`eval_utils`)를
쓴다. 그래야 "같은 조건에서 비교했다"고 쓸 수 있다.

- hyperparameter 는 val 의 AP 로 고르고 test 는 **마지막에 한 번만** 평가한다
- threshold 도 val 에서 고른다 (`pick_threshold`)
- 시드는 42 로 고정한다

실행:
    conda activate chemprop-env
    python scripts/12_chemprop_compare.py --outdir experiment/12_chemprop
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate, pick_threshold, results_table  # noqa: E402

SEED = 42
SPLIT_CSV = Path("experiment/04_split/04_scaffold_split.csv")
SMILES_COL, LABEL_COL, SPLIT_COL = "std_smiles", "label", "split"

# CPU 로 2,176개를 학습하므로 격자는 작게 둔다. 근거는 summary 에 적는다.
GRID = [
    dict(depth=3, hidden=300, dropout=0.0),
    dict(depth=3, hidden=300, dropout=0.2),
    dict(depth=5, hidden=300, dropout=0.0),
    dict(depth=5, hidden=300, dropout=0.2),
]
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64


def load_splits():
    df = pd.read_csv(SPLIT_CSV)
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df[SPLIT_COL] == s]
        out[s] = (sub[SMILES_COL].tolist(), sub[LABEL_COL].to_numpy(int))

    # 누출 확인 — 스캐폴드가 split 사이에 겹치면 즉시 중단한다
    sc = {s: set(df[df[SPLIT_COL] == s]["scaffold"]) for s in ("train", "val", "test")}
    overlap = {
        "train-val": len(sc["train"] & sc["val"]),
        "train-test": len(sc["train"] & sc["test"]),
        "val-test": len(sc["val"] & sc["test"]),
    }
    if any(overlap.values()):
        raise SystemExit(f"스캐폴드 누출: {overlap}")
    return out, overlap


def build(smis, ys, featurizer, shuffle):
    from chemprop import data

    pts = [data.MoleculeDatapoint.from_smi(s, np.array([float(y)]))
           for s, y in zip(smis, ys)]
    dset = data.MoleculeDataset(pts, featurizer)
    return data.build_dataloader(dset, batch_size=BATCH_SIZE,
                                 num_workers=0, shuffle=shuffle)


def train_one(cfg, loaders, seed=SEED):
    """한 설정으로 학습하고 val·test 예측 점수를 돌려준다."""
    from lightning import pytorch as pl
    from chemprop import models, nn

    pl.seed_everything(seed, workers=True)

    mp = nn.BondMessagePassing(d_h=cfg["hidden"], depth=cfg["depth"],
                               dropout=cfg["dropout"])
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(input_dim=cfg["hidden"],
                                     hidden_dim=cfg["hidden"],
                                     dropout=cfg["dropout"])
    model = models.MPNN(mp, agg, ffn, batch_norm=True)

    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, accelerator="cpu", devices=1,
        max_epochs=MAX_EPOCHS, deterministic=True,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=PATIENCE, mode="min")],
    )
    trainer.fit(model, loaders["train"], loaders["val"])

    def predict(loader):
        out = trainer.predict(model, loader)
        return np.concatenate([t.numpy().reshape(-1) for t in out])

    return predict(loaders["val"]), predict(loaders["test"]), trainer.current_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiment/12_chemprop")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import chemprop
        from chemprop import featurizers
    except ImportError as e:
        raise SystemExit(f"chemprop 환경이 아니다 ({e}). conda activate chemprop-env")

    torch.manual_seed(SEED)
    torch.set_num_threads(8)

    splits, overlap = load_splits()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    loaders = {
        "train": build(*splits["train"], featurizer, shuffle=True),
        "val": build(*splits["val"], featurizer, shuffle=False),
        "test": build(*splits["test"], featurizer, shuffle=False),
    }

    y_val, y_test = splits["val"][1], splits["test"][1]

    print(f"chemprop {chemprop.__version__} / torch {torch.__version__}")
    print(f"train {len(splits['train'][1]):,} / val {len(y_val)} / test {len(y_test)}"
          f"  스캐폴드 겹침 {overlap}")

    from sklearn.metrics import average_precision_score

    runs, t0 = [], time.time()
    for i, cfg in enumerate(GRID, 1):
        t = time.time()
        sv, st, epochs = train_one(cfg, loaders)
        val_ap = average_precision_score(y_val, sv)
        runs.append(dict(cfg=cfg, val_ap=val_ap, val_scores=sv,
                         test_scores=st, epochs=epochs,
                         seconds=round(time.time() - t, 1)))
        print(f"  [{i}/{len(GRID)}] {cfg}  val AP {val_ap:.4f}  "
              f"{epochs} epoch  {runs[-1]['seconds']}초")

    best = max(runs, key=lambda r: r["val_ap"])
    print(f"\nval AP 최고 설정: {best['cfg']}  (AP {best['val_ap']:.4f})")

    # threshold 도 val 에서 고른다. test 는 이 아래에서 한 번만 쓴다.
    thr = pick_threshold(y_val, best["val_scores"])
    row = evaluate(y_test, best["test_scores"], thr, label="Chemprop D-MPNN")
    row["val_ap"] = round(best["val_ap"], 4)
    row["params"] = json.dumps(best["cfg"])
    row["epochs"] = best["epochs"]

    pd.DataFrame([row]).to_csv(out / "12_chemprop_result.csv", index=False)

    test_ids = pd.read_csv(SPLIT_CSV)
    test_ids = test_ids[test_ids[SPLIT_COL] == "test"]["molecule_chembl_id"].to_numpy()
    pd.DataFrame({"molecule_chembl_id": test_ids, "label": y_test,
                  "score": best["test_scores"]}).to_csv(
        out / "12_chemprop_test_scores.csv", index=False)

    # 05 단계 결과와 합쳐 3계열 비교표를 만든다
    prev = pd.read_csv("experiment/05_model_comparison/05_model_comparison.csv")
    merged = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    merged = merged.sort_values("pr_auc", ascending=False)
    merged.to_csv(out / "12_three_family_comparison.csv", index=False)

    elapsed = round(time.time() - t0)
    md = [
        "### 분자 표현 3계열 비교 (같은 scaffold split, test 273개 / 활성 159개)",
        "",
        results_table(merged.to_dict("records"), "지문 · 트랜스포머 · 그래프"),
        "",
        "| 계열 | 모델 | 표현 방식 |",
        "|---|---|---|",
        "| 지문 | Morgan (ECFP4, 2048 bit) | 부분구조를 비트로 센다 — 규칙 고정 |",
        "| 트랜스포머 | ChemBERTa (frozen) | SMILES 토큰 문맥을 사전학습으로 배운다 |",
        "| 그래프 | Chemprop D-MPNN | 원자·결합 그래프에서 메시지 전달로 배운다 |",
        "",
        f"- Chemprop {chemprop.__version__}, torch {torch.__version__}, "
        f"시드 {SEED} 고정, CPU 학습",
        f"- hyperparameter 격자 {len(GRID)}개를 val AP 로 비교해 "
        f"`{best['cfg']}` 를 골랐다",
        f"- early stopping: val_loss 기준 patience {PATIENCE}, 최대 {MAX_EPOCHS} epoch "
        f"(선택된 설정은 {best['epochs']} epoch 에서 멈췄다)",
        f"- threshold 는 val 에서 F1 이 최대인 {thr:.4f} 로 정하고 test 는 "
        "한 번만 평가했다",
        f"- 격자를 {len(GRID)}개로 제한한 이유: 학습 분자 2,176개를 로컬 CPU 로 "
        f"돌리므로 탐색 비용을 계산 자원 제약에 맞췄다 (총 {elapsed}초)",
        f"- 스캐폴드 겹침 재확인: {overlap} — 전부 0",
        "",
        "⚠ D-MPNN 이 Morgan+RF 보다 낮게 나오더라도 그것이 결론이다. "
        "데이터가 2,721개뿐이고 활성 비율이 높아 표현을 학습하는 모델에 "
        "유리한 조건이 아니다. 이 점을 보고서에 그대로 적는다.",
    ]
    (out / "12_chemprop_summary.md").write_text("\n".join(md), encoding="utf-8")

    json.dump({"best_cfg": best["cfg"], "val_ap": round(best["val_ap"], 4),
               "threshold": thr, "seed": SEED,
               "chemprop": chemprop.__version__, "torch": torch.__version__,
               "grid": GRID, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "batch_size": BATCH_SIZE, "scaffold_overlap": overlap,
               "total_seconds": elapsed},
              open(out / "12_chemprop_stats.json", "w"), indent=1)

    print("\n" + merged[["model", "pr_auc", "ap", "roc_auc", "ef5"]]
          .to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
