"""
활성 예측 모델 비교 — 같은 scaffold split 에서 세 가지 표현을 직접 비교한다.

표현 (feature)
  1. Morgan fingerprint (ECFP4, 2048 bit)   — 수업 baseline
  2. ChemBERTa-10M-MTR  frozen embedding    — masked mean pooling, 384차원
  3. ChemBERTa-77M-MLM  frozen embedding    — (모델 폴더가 있을 때만)

분류기: Logistic Regression, Random Forest

규칙 (W09 수업 방식, 누출 방지)
  - scaler·분류기는 **train 에서만 fit**
  - hyperparameter 와 threshold 는 **val** 에서 고른다
  - **test 는 마지막에 한 번만** 평가한다

사전학습 모델은 고정(frozen)한다. 파인튜닝하지 않으므로 CPU 로 충분하다.

입력:  experiment/04_split/04_scaffold_split.csv
       models/<모델이름>/            (config.json, pytorch_model.bin, tokenizer.json ...)
출력:  experiment/05_model_comparison/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from eval_utils import ef_ceiling, evaluate, pick_threshold, results_table

SEED = 42


# ---------------------------------------------------------------- 표현 만들기

def morgan_features(smiles_list, radius=2, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = np.array(gen.GetFingerprint(mol), dtype=np.float32)
    return out


def chemberta_features(smiles_list, model_dir, batch_size=64, max_len=256):
    """encoder 출력을 masked mean pooling 해서 분자 하나당 벡터 하나."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()  # dropout 끄기

    vecs = []
    for start in range(0, len(smiles_list), batch_size):
        batch = list(smiles_list[start:start + batch_size])
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        with torch.no_grad():  # 역전파 안 함 — frozen
            hidden = model(**enc).last_hidden_state      # (B, L, 384)

        # [CLS]·[SEP]·padding 을 제외한 내용 token 만 평균
        mask = enc["attention_mask"].clone()
        special = torch.zeros_like(mask, dtype=torch.bool)
        for sid in tok.all_special_ids:
            special |= enc["input_ids"] == sid
        mask[special] = 0
        m = mask.unsqueeze(-1).float()

        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vecs.append(pooled.numpy())

    return np.vstack(vecs).astype(np.float32)


# ---------------------------------------------------------------- 학습·선택

GRIDS = {
    "LogReg": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "RF": [
        dict(n_estimators=500, max_depth=d, min_samples_leaf=l)
        for d in (None, 20) for l in (1, 3)
    ],
}


def make_clf(kind, params):
    if kind == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=SEED,
                                  class_weight="balanced", **params)
    return RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                  class_weight="balanced", **params)


def fit_select_evaluate(X, y, splits, feat_name, clf_kind):
    tr, va, te = splits

    # scaler 는 train 에서만 fit (test 로 fit 하면 누출)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])

    # hyperparameter 는 val 에서 고른다 (val PR-AUC 기준)
    best, best_score = None, -1.0
    for params in GRIDS[clf_kind]:
        clf = make_clf(clf_kind, params).fit(Xtr, y[tr])
        s = average_precision_score(y[va], clf.predict_proba(Xva)[:, 1])
        if s > best_score:
            best, best_score = params, s

    clf = make_clf(clf_kind, best).fit(Xtr, y[tr])
    val_scores = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(y[va], val_scores)       # threshold 도 val 에서

    test_scores = clf.predict_proba(Xte)[:, 1]    # test 는 여기서 한 번만
    row = evaluate(y[te], test_scores, thr, label=f"{feat_name} + {clf_kind}")
    row["best_params"] = json.dumps(best)
    row["val_ap"] = round(best_score, 4)
    return row, test_scores


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="experiment/04_split/04_scaffold_split.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--outdir", default="experiment/05_model_comparison")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split)
    y = df["label"].to_numpy()
    idx = {name: np.where(df["split"] == name)[0] for name in ("train", "val", "test")}
    splits = (idx["train"], idx["val"], idx["test"])
    print(f"train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])}")

    smiles = df["std_smiles"].tolist()
    features = {}

    print("\nMorgan fingerprint (ECFP4, 2048 bit) 생성 ...")
    features["Morgan"] = morgan_features(smiles)
    print("  shape", features["Morgan"].shape)

    for name in ("ChemBERTa-10M-MTR", "ChemBERTa-77M-MLM"):
        mdir = Path(args.models_dir) / name
        if not mdir.exists():
            print(f"\n{name}: 모델 폴더가 없어 건너뜀 ({mdir})")
            continue
        print(f"\n{name} embedding 생성 (frozen, masked mean pooling) ...")
        features[name] = chemberta_features(smiles, str(mdir))
        print("  shape", features[name].shape)

    rows, score_cols = [], {}
    for feat_name, X in features.items():
        for clf_kind in ("LogReg", "RF"):
            row, test_scores = fit_select_evaluate(X, y, splits, feat_name, clf_kind)
            rows.append(row)
            score_cols[row["model"]] = test_scores
            print(f"  {row['model']:34s} PR-AUC {row['pr_auc']:.3f}  "
                  f"ROC-AUC {row['roc_auc']:.3f}  EF5% {row['ef5']:.2f}")

    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    res.to_csv(outdir / "05_model_comparison.csv", index=False)

    # test 예측값 저장 — 나중에 보고서 그림(PR 곡선, EF 비교) 그릴 때 쓴다
    sc = pd.DataFrame(score_cols)
    sc.insert(0, "label", y[idx["test"]])
    sc.insert(0, "molecule_chembl_id", df.iloc[idx["test"]]["molecule_chembl_id"].values)
    sc.to_csv(outdir / "05_test_scores.csv", index=False)

    base = y[idx["test"]].mean()
    md = [
        results_table(rows, "활성 예측 모델 비교 (scaffold split test, 마지막 1회 평가)"),
        "",
        f"- test {len(idx['test'])}개 중 활성 {int(y[idx['test']].sum())}개 (기준선 {base:.1%})",
        "- EF 5% = 상위 5%의 활성 비율 ÷ 전체 활성 비율. **1이면 무작위와 같음**",
        f"- ⚠ **이 test 에서 EF 의 이론적 최대값은 {ef_ceiling(y[idx['test']], 0.05):.3f}** "
        f"(= 1 / 활성 비율 {base:.3f}). 상위를 전부 활성으로 맞혀도 이 값을 넘을 수 없다.",
        "  활성 비율이 높아 EF 가 천장에 붙으므로, 모델 구분은 PR-AUC·ROC-AUC 로 한다.",
        "  실제 스크리닝에서는 활성 비율이 훨씬 낮아 EF 의 천장이 높고 지표로서 의미가 있다.",
        "- PR-AUC 는 사다리꼴 적분, AP 는 계단 합이라 값이 다를 수 있어 둘 다 적는다",
        "- hyperparameter 와 threshold 는 val 에서 선택했고 test 는 한 번만 평가했다",
        "- scaler 는 train 에서만 fit 했다 (누출 방지)",
        "- ChemBERTa 는 **frozen**, encoder 출력을 masked mean pooling 한 384차원 벡터",
        "",
        "| 모델 | val AP | 선택된 hyperparameter |",
        "|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['model']} | {r['val_ap']} | `{r['best_params']}` |")

    (outdir / "05_model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + res[["model", "pr_auc", "ap", "roc_auc", "ef5", "ef1"]].to_string(index=False))
    print(f"\n저장: {outdir}")


if __name__ == "__main__":
    main()
