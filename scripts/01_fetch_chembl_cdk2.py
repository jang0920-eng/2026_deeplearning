"""
CDK2 (CHEMBL301) 활성 데이터 수집

출력: experiment/01_chembl_cdk2_activity.csv
      experiment/01_fetch_summary.txt

실행:
    pip install chembl_webresource_client pandas
    python 01_fetch_chembl_cdk2.py

Colab에서 실행할 때는 첫 줄에 !pip install chembl_webresource_client 를 넣으세요.
"""

import os
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET = "CHEMBL301"          # Cyclin-dependent kinase 2 (human)
ACTIVITY_TYPES = ["IC50", "Ki", "Kd"]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment")

FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "target_chembl_id",
    "document_year",
]


def fetch(activity_type):
    """한 가지 활성 지표를 모두 가져온다."""
    print(f"  {activity_type} 수집 중...", flush=True)
    records = new_client.activity.filter(
        target_chembl_id=TARGET,
        standard_type=activity_type,
    ).only(FIELDS)
    df = pd.DataFrame(list(records))
    print(f"    원본 {len(df)}건")
    return df


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 타겟 정보 확인 — 엉뚱한 타겟을 긁지 않았는지 검증
    target = new_client.target.get(TARGET)
    print(f"타겟: {target['pref_name']} ({target['organism']})")
    print(f"타겟 유형: {target['target_type']}\n")

    frames = [fetch(t) for t in ACTIVITY_TYPES]
    raw = pd.concat([f for f in frames if len(f)], ignore_index=True)
    n_raw = len(raw)
    print(f"\n원본 합계: {n_raw}건")

    # --- 정제 ---
    log = [f"타겟: {target['pref_name']} ({TARGET})", f"원본: {n_raw}건"]

    def drop(df, cond, reason):
        before = len(df)
        df = df[cond].copy()
        removed = before - len(df)
        msg = f"{reason}: -{removed}건 -> {len(df)}건"
        print(f"  {msg}")
        log.append(msg)
        return df

    df = raw
    print("\n정제:")
    df = drop(df, df["canonical_smiles"].notna(), "SMILES 없음 제거")
    df = drop(df, df["standard_value"].notna(), "활성값 없음 제거")
    df = drop(df, df["standard_units"] == "nM", "단위가 nM이 아닌 항목 제거")
    # '>' '<' 같은 부등호 데이터는 정확한 값이 아니므로 제외
    df = drop(df, df["standard_relation"] == "=", "부등호(>, <) 데이터 제거")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = drop(df, df["standard_value"].notna(), "수치 변환 실패 제거")
    df = drop(df, df["standard_value"] > 0, "0 이하 값 제거")

    # pIC50 계산 (nM -> M -> -log10)
    df["pActivity"] = -np.log10(df["standard_value"] * 1e-9)

    # 같은 분자가 여러 번 측정된 경우 중앙값으로 통합
    before = len(df)
    agg = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)
        .agg(
            pActivity=("pActivity", "median"),
            n_measurements=("pActivity", "size"),
            activity_types=("standard_type", lambda s: ",".join(sorted(set(s)))),
        )
    )
    msg = f"분자 단위 통합(중앙값): {before}건 -> {len(agg)}개 분자"
    print(f"  {msg}")
    log.append(msg)

    # 활성/비활성 라벨 — 분류 모델용 (pActivity 6.0 = 1 uM 기준)
    agg["label"] = (agg["pActivity"] >= 6.0).astype(int)
    n_act = int(agg["label"].sum())

    out_csv = os.path.join(OUTDIR, "01_chembl_cdk2_activity.csv")
    agg.to_csv(out_csv, index=False)

    summary = [
        "",
        f"최종 분자 수: {len(agg)}",
        f"  활성 (pActivity >= 6.0, IC50 <= 1 uM): {n_act}",
        f"  비활성: {len(agg) - n_act}",
        f"pActivity 범위: {agg['pActivity'].min():.2f} ~ {agg['pActivity'].max():.2f}",
        f"pActivity 중앙값: {agg['pActivity'].median():.2f}",
        f"저장: {out_csv}",
    ]
    print("\n".join(summary))
    log.extend(summary)

    with open(os.path.join(OUTDIR, "01_fetch_summary.txt"), "w") as f:
        f.write("\n".join(log) + "\n")

    # 모델 학습 가능성 판단
    print()
    if len(agg) >= 2000 and n_act >= 500:
        print("판정: ML 활성 예측 모델 학습에 충분합니다.")
    elif len(agg) >= 500:
        print("판정: 학습은 가능하지만 데이터가 적습니다. BindingDB 추가를 고려하세요.")
    else:
        print("판정: 데이터 부족. 다른 소스를 추가해야 합니다.")


if __name__ == "__main__":
    main()
