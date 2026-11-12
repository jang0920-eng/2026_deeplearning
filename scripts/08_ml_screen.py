"""
2단계 ML 활성 예측 — 학습한 Morgan+RF 를 필터 통과 라이브러리 전체에 적용한다.

입력:  experiment/07_filter/07_filtered_unique.csv(.gz)   1,423,957개
       experiment/04_split/04_scaffold_split.csv          라벨 2,721개
출력:  experiment/08_ml_screen/

모델은 05 단계에서 PR-AUC 1위였던 조합을 그대로 쓴다.
  표현     Morgan fingerprint (ECFP4, 2048 bit)
  분류기   RandomForest(n_estimators=500, max_depth=None, min_samples_leaf=1,
                        class_weight="balanced", random_state=42)
  전처리   StandardScaler  — 05 단계 파이프라인과 동일하게 유지
hyperparameter 는 05 단계에서 **val** 로 고른 것이고 여기서 다시 고르지 않는다.

## 학습에 쓰는 데이터 (--fit-on, 기본 trainval)
스크리닝 모델은 train+val(2,448개)로 학습하고 **test 273개는 끝까지 제외한다.**
test 를 빼 두면 라이브러리 142만 개를 배경으로 한 **정직한 농축 측정**이 가능해진다.
05 단계의 test 활성 비율은 58.2%여서 EF 천장이 1.717 밖에 안 됐는데,
라이브러리를 배경으로 하면 활성 비율이 훨씬 낮아 EF 가 지표로서 의미를 가진다.
즉 test 분자를 142만 개 속에 섞어 둔 채로 순위를 보는 것이 사실상 spike-in 실험이 된다.

## 알려진 CDK2 분자 처리
라이브러리는 ChEMBL 전체이므로 학습에 쓴 분자가 그 안에 들어 있다.
이들은 **신규 후보가 아니므로 후보 목록에서 제외**하되, 순위는 기록한다.
학습 분자가 상위에 몰리는지는 모델이 제대로 적용됐는지 확인하는 양성 대조가 된다.

## 실행
    python scripts/08_ml_screen.py --library <07_filtered_unique.csv.gz>
도중에 끊기면 같은 명령을 다시 실행하면 된다 (점수 파일을 이어서 쓴다).
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
RADIUS, N_BITS = 2, 2048
RF_PARAMS = dict(n_estimators=500, max_depth=None, min_samples_leaf=1)

# ---------------------------------------------------------------- worker 쪽

_GEN = None
_UNCHARGER = None


def _init_worker():
    """worker 마다 한 번만 생성한다. Uncharger 를 분자마다 새로 만들면 5배 느리다."""
    global _GEN, _UNCHARGER
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    _UNCHARGER = rdMolStandardize.Uncharger()


def standardize_mol(mol):
    """04 단계 `standardize()` 와 같은 순서: Cleanup → FragmentParent → 중성화."""
    from rdkit.Chem.MolStandardize import rdMolStandardize

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return mol


def featurize(record):
    """(id, smiles) → (id, 패킹된 fingerprint). 실패하면 None."""
    from rdkit import Chem

    cid, smi = record
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = standardize_mol(mol)
    if mol is None:
        return None
    fp = _GEN.GetFingerprintAsNumPy(mol)          # uint8 0/1, 2048
    return cid, np.packbits(fp).tobytes()         # 2048 bit → 256 byte


def standardize_smiles_only(record):
    """상위 후보의 SMILES 만 다시 표준화할 때 쓴다 (2차 통과)."""
    from rdkit import Chem

    cid, smi = record
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return cid, None
    mol = standardize_mol(mol)
    return cid, (Chem.MolToSmiles(mol) if mol is not None else None)


# ---------------------------------------------------------------- 학습

def morgan_matrix(smiles_list):
    """라벨 데이터용 — 단일 프로세스로 충분한 양(2,721개)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    X = np.zeros((len(smiles_list), N_BITS), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            X[i] = gen.GetFingerprintAsNumPy(mol)
    return X


def train_model(split_csv, fit_on, cache):
    """스크리닝용 Morgan+RF 를 학습한다. 캐시가 있으면 재사용."""
    import joblib

    df = pd.read_csv(split_csv)
    if cache.exists():
        obj = joblib.load(cache)
        if obj.get("fit_on") == fit_on:
            print(f"학습 모델 재사용: {cache}")
            return obj["scaler"], obj["rf"], df, obj["fit_idx"]

    want = {"train": ["train"], "trainval": ["train", "val"],
            "all": ["train", "val", "test"]}[fit_on]
    fit_idx = np.where(df["split"].isin(want))[0]

    X = morgan_matrix(df["std_smiles"].tolist())
    y = df["label"].to_numpy()

    t = time.time()
    scaler = StandardScaler().fit(X[fit_idx])
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                class_weight="balanced", **RF_PARAMS)
    rf.fit(scaler.transform(X[fit_idx]), y[fit_idx])
    print(f"학습 완료 ({fit_on}, {len(fit_idx)}개, 활성 {int(y[fit_idx].sum())}개) "
          f"{time.time() - t:.1f}초")

    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(rf=rf, scaler=scaler, fit_on=fit_on, fit_idx=fit_idx,
                     params=RF_PARAMS, seed=SEED, radius=RADIUS, n_bits=N_BITS),
                cache, compress=3)
    return scaler, rf, df, fit_idx


# ---------------------------------------------------------------- 라이브러리 통과

def open_library(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def iter_library(path, skip=0):
    """(chembl_id, smiles) 를 흘려 준다. 142만 줄을 메모리에 올리지 않는다."""
    with open_library(path) as f:
        f.readline()                                  # 헤더
        for i, line in enumerate(f):
            if i < skip:
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def score_library(library, scaler, rf, out_csv, batch, workers):
    """라이브러리 전체에 모델을 적용해 (id, score) 를 이어서 쓴다.

    배치마다 flush 하므로 중간에 끊겨도 지금까지 계산한 점수는 남는다.
    다시 실행하면 이미 쓴 줄 수만큼 건너뛴다.
    """
    done = 0
    if out_csv.exists():
        with open(out_csv) as f:
            done = max(0, sum(1 for _ in f) - 1)
        print(f"이어서 실행: 이미 {done:,}개 점수 있음")

    mode = "a" if done else "w"
    t0 = time.time()
    n_fail = 0
    with open(out_csv, mode, buffering=1 << 20) as fo, \
            Pool(workers, initializer=_init_worker) as pool:
        if not done:
            fo.write("chembl_id,score\n")

        for bi, batch_rows in enumerate(chunks(iter_library(library, done), batch)):
            res = [r for r in pool.imap(featurize, batch_rows, chunksize=200)
                   if r is not None]
            n_fail += len(batch_rows) - len(res)
            if not res:
                continue

            ids = [r[0] for r in res]
            packed = np.frombuffer(b"".join(r[1] for r in res), dtype=np.uint8)
            X = np.unpackbits(packed.reshape(len(res), -1), axis=1)[:, :N_BITS]
            p = rf.predict_proba(scaler.transform(X.astype(np.float32)))[:, 1]

            fo.write("".join(f"{c},{s:.4f}\n" for c, s in zip(ids, p)))
            fo.flush()

            done += len(batch_rows)
            el = time.time() - t0
            print(f"  배치 {bi + 1:4d}  누적 {done:>9,}개  "
                  f"{done / max(el, 1e-9):.0f}개/초  경과 {el / 60:.1f}분", flush=True)

    print(f"점수 계산 완료: {done:,}개 (구조 생성 실패 {n_fail}개) "
          f"{(time.time() - t0) / 60:.1f}분")
    return n_fail


# ---------------------------------------------------------------- 결과 정리

def build_report(library, split_df, fit_idx, scores_csv, outdir, top_n, workers):
    sc = pd.read_csv(scores_csv)
    sc = sc.drop_duplicates("chembl_id", keep="first")
    n_lib = len(sc)
    print(f"점수 {n_lib:,}개")

    # 순위 (1 = 가장 높은 점수). 동점은 평균 순위.
    sc["rank"] = sc["score"].rank(ascending=False, method="average").astype(np.int64)
    sc = sc.sort_values("rank")

    known = set(split_df["molecule_chembl_id"])
    sc["is_known"] = sc["chembl_id"].isin(known)

    # --- 알려진 분자가 라이브러리 안에서 어디에 있는지
    kr = split_df[["molecule_chembl_id", "label", "split", "pActivity"]].merge(
        sc[["chembl_id", "score", "rank"]],
        left_on="molecule_chembl_id", right_on="chembl_id", how="left")
    kr["in_library"] = kr["rank"].notna()
    kr["percentile"] = kr["rank"] / n_lib * 100
    kr = kr.drop(columns=["chembl_id"]).sort_values("rank")
    kr.to_csv(outdir / "08_known_ranks.csv", index=False)

    # --- test 분자를 142만 개 배경에 섞은 셈이므로 농축을 정직하게 잴 수 있다
    enrich = {}
    for split_name in ("test", "train", "val"):
        sub = kr[(kr["split"] == split_name) & kr["in_library"]]
        act = sub[sub["label"] == 1]
        row = {"n_in_library": int(len(sub)), "n_active_in_library": int(len(act))}
        for frac in (0.001, 0.01, 0.05):
            cut = frac * n_lib
            hit = int((act["rank"] <= cut).sum())
            row[f"active_top{frac:g}"] = hit
            row[f"ef_top{frac:g}"] = round(hit / len(act) / frac, 2) if len(act) else float("nan")
        row["median_percentile_active"] = round(float(act["percentile"].median()), 3) if len(act) else float("nan")
        inact = sub[sub["label"] == 0]
        row["median_percentile_inactive"] = round(float(inact["percentile"].median()), 3) if len(inact) else float("nan")
        enrich[split_name] = row

    # --- 필터 통과율 (알려진 활성이 1단계 필터에서 얼마나 살아남았는지)
    filt = {
        "known_total": int(len(kr)),
        "known_in_library": int(kr["in_library"].sum()),
        "known_active_total": int((kr["label"] == 1).sum()),
        "known_active_in_library": int(((kr["label"] == 1) & kr["in_library"]).sum()),
    }

    # --- 점수 분포 (보고서 그림용)
    bins = np.linspace(0, 1, 101)
    cnt, _ = np.histogram(sc["score"].to_numpy(), bins=bins)
    pd.DataFrame({"bin_left": bins[:-1].round(2), "bin_right": bins[1:].round(2),
                  "count": cnt}).to_csv(outdir / "08_score_hist.csv", index=False)

    # --- 상위 후보 (알려진 분자 제외) + SMILES 를 2차 통과로 채운다
    cand = sc[~sc["is_known"]].head(top_n).copy()
    need = set(cand["chembl_id"])
    raw = {}
    for cid, smi in iter_library(library):
        if cid in need:
            raw[cid] = smi
            if len(raw) == len(need):
                break
    with Pool(workers, initializer=_init_worker) as pool:
        std = dict(pool.map(standardize_smiles_only, list(raw.items()), chunksize=100))
    cand["smiles"] = cand["chembl_id"].map(std)
    cand = cand.dropna(subset=["smiles"])
    cand = cand[["rank", "chembl_id", "smiles", "score"]]
    cand.to_csv(outdir / "08_top_candidates.csv", index=False)

    stats = {
        "library_scored": n_lib,
        "top_n_saved": int(len(cand)),
        "known_in_top_n": int(sc.head(top_n)["is_known"].sum()),
        "score_min": float(sc["score"].min()),
        "score_median": float(sc["score"].median()),
        "score_max": float(sc["score"].max()),
        "score_at_top_n": float(cand["score"].iloc[-1]) if len(cand) else None,
        "n_score_ge_0.5": int((sc["score"] >= 0.5).sum()),
        "n_score_ge_0.7": int((sc["score"] >= 0.7).sum()),
        "n_score_ge_0.9": int((sc["score"] >= 0.9).sum()),
        "filter_retention": filt,
        "enrichment": enrich,
    }
    (outdir / "08_screen_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    write_summary(outdir, stats, len(fit_idx))
    return stats


def write_summary(outdir, s, n_fit):
    e = s["enrichment"]["test"]
    f = s["filter_retention"]
    md = f"""### 2단계 ML 활성 예측 (Morgan + RF)

- 입력: 필터 통과 라이브러리 **{s['library_scored']:,}개**
- 모델: 05 단계 PR-AUC 1위 조합(Morgan ECFP4 2048bit + RF 500tree)을 그대로 적용
- 학습: train+val **{n_fit:,}개**. test 273개는 제외했다
- 점수 분포: 최소 {s['score_min']:.3f} / 중앙값 {s['score_median']:.3f} / 최대 {s['score_max']:.3f}

| 예측 확률 | 분자 수 | 비율 |
|---|---|---|
| ≥ 0.5 | {s['n_score_ge_0.5']:,} | {s['n_score_ge_0.5'] / s['library_scored'] * 100:.2f}% |
| ≥ 0.7 | {s['n_score_ge_0.7']:,} | {s['n_score_ge_0.7'] / s['library_scored'] * 100:.2f}% |
| ≥ 0.9 | {s['n_score_ge_0.9']:,} | {s['n_score_ge_0.9'] / s['library_scored'] * 100:.3f}% |

상위 {s['top_n_saved']:,}개를 다음 단계로 넘긴다 (경계 점수 {s['score_at_top_n']:.3f}).
상위 {s['top_n_saved']:,}위 안에 들어 있던 **알려진 CDK2 분자 {s['known_in_top_n']:,}개는 후보에서 제외**했다.
신규 후보가 아니기 때문이며, 순위는 `08_known_ranks.csv` 에 남겼다.

#### 라이브러리를 배경으로 한 농축 — EF 천장 문제의 해결

05 단계 test 는 활성 비율이 58.2%여서 EF 의 이론적 최대값이 1.717 이었다.
스크리닝 모델을 train+val 로만 학습했으므로, **test 273개는 라이브러리
{s['library_scored']:,}개 안에 섞여 있는 미지의 분자**가 된다.
배경이 142만 개로 커져 활성 비율이 낮아지므로 EF 천장이 사라지고
(상위 1% 기준 이론적 최대 100) 농축을 정직하게 측정할 수 있다.

| 기준 | 라이브러리 안 test 활성 | 그 중 상위에 든 수 | EF |
|---|---|---|---|
| 상위 0.1% | {e['n_active_in_library']} | {e['active_top0.001']} | **{e['ef_top0.001']}** |
| 상위 1% | {e['n_active_in_library']} | {e['active_top0.01']} | **{e['ef_top0.01']}** |
| 상위 5% | {e['n_active_in_library']} | {e['active_top0.05']} | **{e['ef_top0.05']}** |

test 활성 분자의 순위 중앙값은 상위 {e['median_percentile_active']}%,
test 비활성 분자는 상위 {e['median_percentile_inactive']}% 였다.
scaffold split 이므로 이 분자들의 골격은 학습에 전혀 쓰이지 않았다.

#### 1단계 필터가 알려진 활성 분자를 얼마나 남겼는가

| 항목 | 수 | 비율 |
|---|---|---|
| 라벨 데이터 전체 | {f['known_total']:,} | — |
| 그 중 필터 통과 | {f['known_in_library']:,} | {f['known_in_library'] / f['known_total'] * 100:.1f}% |
| 알려진 활성 | {f['known_active_total']:,} | — |
| 그 중 필터 통과 | {f['known_active_in_library']:,} | {f['known_active_in_library'] / f['known_active_total'] * 100:.1f}% |

lead-like 조건(MW ≤ 450, 회전결합 ≤ 7)은 알려진 CDK2 활성 분자의 일부도 떨어뜨린다.
도킹 시간을 줄이기 위해 감수한 손실이며, 그 크기를 위 표로 실측해 둔다.

#### 한계
- 예측 확률은 **실험으로 검증된 결합 친화도가 아니다.** 다음 단계로 넘길 분자의
  우선순위를 정하는 용도이며, 활성을 주장하는 근거가 아니다.
- 학습 데이터(CDK2 억제제 2,721개)는 화학적으로 몇 개 골격에 몰려 있어,
  그 골격과 많이 다른 분자에 대한 예측은 외삽이다.
"""
    (outdir / "08_screen_summary.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True,
                    help="07_filtered_unique.csv 또는 .csv.gz")
    ap.add_argument("--split", default="experiment/04_split/04_scaffold_split.csv")
    ap.add_argument("--outdir", default="experiment/08_ml_screen")
    ap.add_argument("--fit-on", default="trainval",
                    choices=["train", "trainval", "all"])
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage", default="all", choices=["all", "score", "report"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scores_csv = outdir / "08_library_scores.csv"

    scaler, rf, split_df, fit_idx = train_model(
        Path(args.split), args.fit_on, outdir / "08_rf_screen.joblib")

    if args.stage in ("all", "score"):
        score_library(args.library, scaler, rf, scores_csv, args.batch, args.workers)
    if args.stage in ("all", "report"):
        stats = build_report(args.library, split_df, fit_idx, scores_csv,
                             outdir, args.top, args.workers)
        print(json.dumps(stats["enrichment"]["test"], indent=1, ensure_ascii=False))
        print(f"저장: {outdir}")


if __name__ == "__main__":
    main()
