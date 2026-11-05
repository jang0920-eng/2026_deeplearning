"""
CDK2 활성 데이터 scaffold split (train/val/test = 80/10/10)

W09 수업 방식 그대로:
  1. 표준화·중복 제거를 split **전에** 수행
  2. Murcko scaffold 기준으로 분자를 묶고, 큰 묶음부터 train → val → test 에 배분
  3. 같은 scaffold 가 두 split 에 걸치지 않게 한다

무작위 분할을 쓰지 않는 이유: CDK2 억제제는 같은 골격의 유사체가 몰려 있어
무작위로 나누면 test 에 train 과 거의 같은 분자가 들어가 성능이 부풀려진다.
그 모델로 50만 개를 거르면 실패한다.

입력:  experiment/01_chembl_cdk2_activity.csv
출력:  experiment/04_split/04_scaffold_split.csv      (split 컬럼이 붙은 전체 데이터)
       experiment/04_split/04_split_summary.md        (skill 문서에 붙일 요약)
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

TRAIN_FRAC, VAL_FRAC = 0.80, 0.10
ACTIVITY_CUTOFF = 6.0  # pActivity >= 6.0 이면 활성 (01 단계와 동일 기준)


def standardize(smiles):
    """염 제거 → 중성화 → 정규 SMILES. 실패하면 None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)  # 가장 큰 조각만 (염 제거)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def murcko_scaffold(smiles):
    """Murcko scaffold 의 정규 SMILES. 고리가 없으면 빈 문자열이 나온다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def scaffold_split(scaffolds, n_total):
    """scaffold → index 목록. 큰 묶음부터 train, 그다음 val, 나머지 test."""
    groups = sorted(scaffolds.values(), key=lambda idx: (len(idx), idx[0]), reverse=True)

    n_train, n_val = TRAIN_FRAC * n_total, VAL_FRAC * n_total
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(val) + len(group) <= n_val:
            val += group
        else:
            test += group
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiment/01_chembl_cdk2_activity.csv")
    ap.add_argument("--outdir", default="experiment/04_split")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    n_raw = len(df)
    print(f"입력 {n_raw}개")

    # --- 1. 표준화 (split 전에) ---
    df["std_smiles"] = df["canonical_smiles"].map(standardize)
    n_bad = df["std_smiles"].isna().sum()
    df = df.dropna(subset=["std_smiles"]).copy()
    print(f"표준화 실패 제거: {n_bad}개 → {len(df)}개")

    # --- 2. 중복 제거 (split 전에) ---
    # 같은 구조가 여러 ChEMBL ID 로 들어온 경우 pActivity 평균으로 합친다.
    n_before = len(df)
    agg = (
        df.groupby("std_smiles")
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pActivity=("pActivity", "mean"),
            n_measurements=("n_measurements", "sum"),
            n_duplicates=("std_smiles", "size"),
        )
        .reset_index()
    )
    print(f"중복 구조 통합: {n_before} → {len(agg)}개 ({n_before - len(agg)}개 감소)")

    agg["label"] = (agg["pActivity"] >= ACTIVITY_CUTOFF).astype(int)

    # --- 3. Murcko scaffold ---
    agg["scaffold"] = agg["std_smiles"].map(murcko_scaffold)
    n_noscaf = (agg["scaffold"].isna() | (agg["scaffold"] == "")).sum()
    # 고리가 없어 scaffold 가 빈 분자는 각각 독립 묶음으로 둔다
    # (한 덩어리로 묶으면 서로 무관한 분자들이 같은 split 에 몰린다)
    agg["scaffold"] = [
        s if s else f"__acyclic_{i}__" for i, s in enumerate(agg["scaffold"].fillna(""))
    ]
    print(f"고리 없는 분자: {n_noscaf}개 (각각 독립 묶음 처리)")

    scaffolds = defaultdict(list)
    for i, s in enumerate(agg["scaffold"]):
        scaffolds[s].append(i)
    print(f"고유 scaffold: {len(scaffolds)}종")

    # --- 4. split ---
    train, val, test = scaffold_split(scaffolds, len(agg))
    agg["split"] = None
    for name, idx in [("train", train), ("val", val), ("test", test)]:
        agg.loc[idx, "split"] = name

    # --- 5. 검증: scaffold 가 split 에 걸치지 않는지 ---
    overlap = (
        agg.groupby("scaffold")["split"].nunique().pipe(lambda s: (s > 1).sum())
    )
    assert overlap == 0, f"scaffold 가 {overlap}종에서 split 에 걸쳐 있습니다"

    cols = [
        "molecule_chembl_id", "std_smiles", "pActivity", "label",
        "scaffold", "split", "n_measurements", "n_duplicates",
    ]
    agg[cols].to_csv(outdir / "04_scaffold_split.csv", index=False)

    # --- 6. 요약 ---
    rows = []
    for name in ["train", "val", "test"]:
        g = agg[agg["split"] == name]
        rows.append(
            dict(
                split=name,
                n=len(g),
                pct=round(100 * len(g) / len(agg), 1),
                actives=int(g["label"].sum()),
                active_pct=round(100 * g["label"].mean(), 1),
                scaffolds=g["scaffold"].nunique(),
                pAct_median=round(g["pActivity"].median(), 2),
            )
        )
    summary = pd.DataFrame(rows)
    print()
    print(summary.to_string(index=False))

    top = (
        agg.groupby("scaffold")
        .agg(n=("scaffold", "size"), split=("split", "first"))
        .sort_values("n", ascending=False)
        .head(5)
    )

    md = [
        "### scaffold split (train/val/test = 80/10/10)\n",
        f"- 원본 {n_raw}개 → 표준화 실패 {n_bad}개 제거 → "
        f"중복 구조 {n_before - len(agg)}개 통합 → **{len(agg)}개**",
        "- 표준화: 염 제거(FragmentParent) → 중성화 → 정규 SMILES, **split 전에 수행**",
        f"- Murcko scaffold {len(scaffolds)}종 기준으로 분할, "
        f"같은 골격이 두 split 에 걸치지 않음 (검증 완료)",
        f"- 활성 기준 pActivity ≥ {ACTIVITY_CUTOFF}\n",
        "| split | 분자 수 | 비율 | 활성 | 활성 비율 | scaffold 수 | pActivity 중앙값 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['split']} | {r['n']} | {r['pct']}% | {r['actives']} | "
            f"{r['active_pct']}% | {r['scaffolds']} | {r['pAct_median']} |"
        )
    md += [
        "",
        f"- 가장 큰 scaffold 묶음 {int(top['n'].iloc[0])}개 (train 에 배정)",
        "- 무작위 분할을 쓰지 않은 이유: 같은 골격의 유사체가 몰려 있어 "
        "무작위 분할 시 test 성능이 부풀려지고, 그 모델로 대규모 라이브러리를 거르면 실패한다.",
    ]
    (outdir / "04_split_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n저장: {outdir / '04_scaffold_split.csv'}")
    print(f"저장: {outdir / '04_split_summary.md'}")


if __name__ == "__main__":
    main()
