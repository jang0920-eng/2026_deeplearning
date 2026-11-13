"""
13. ADMET 예측 — 도킹 상위 100개에서 최종 우선순위를 추린다 (깔때기 5단계).

도구: ADMET-AI (chemprop D-MPNN, Therapeutics Data Commons 학습)
판정 기준은 `skill/04_admet.md` 에 **도킹 결과를 보기 전에** 확정해 둔 것을 그대로 쓴다.
이 스크립트는 기준을 재정의하지 않는다.

  1. 하드 필터 — hERG / AMES / DILI 예측 확률 > 0.5 면 탈락
  2. 통과 분자를 4단계 도킹 순위(rank_mean)로 그대로 정렬
  3. 상위 N개를 최종 후보로 제시. 통과 분자가 N 미만이면 전부 제시하고 그 수를 보고한다

ADMET 지표에 임의의 가중치를 주어 도킹 순위와 섞지 않는다.
가중치를 정할 근거가 없고, 결과를 보고 가중치를 고르면 맞춘 셈이 되기 때문이다.

실행:
    conda activate chemprop-env
    python scripts/13_admet.py --input experiment/11_docking/11_top_for_admet.csv \
                               --outdir experiment/13_admet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# skill/04_admet.md 3-1 에 적어 둔 탈락 기준. 0.5 는 이진 분류 기본 결정 경계다.
HARD_FILTERS = {
    "hERG": ("hERG 저해 (심장 독성)", 0.5),
    "AMES": ("AMES 돌연변이성 (유전독성)", 0.5),
    "DILI": ("약물 유발 간손상", 0.5),
}

# 탈락시키지 않고 표에 적어 두는 지표 (skill/04_admet.md 3-2)
REPORT_ONLY = [
    "QED", "Solubility_AqSolDB", "Caco2_Wang", "HIA_Hou", "Bioavailability_Ma",
    "BBB_Martins", "PPBR_AZ", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
    "CYP1A2_Veith", "CYP2C9_Veith", "CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
    "ClinTox", "Carcinogens_Lagunin", "LD50_Zhu",
]

N_FINAL = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiment/11_docking/11_top_for_admet.csv")
    ap.add_argument("--outdir", default="experiment/13_admet")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--n-final", type=int, default=N_FINAL)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(
            f"입력이 없다: {src}\n"
            "4단계 도킹을 끝내고 노트북 마지막 셀로 11_top_for_admet.csv 를 만들어야 한다."
        )

    df = pd.read_csv(src)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"{args.smiles_col} 컬럼이 없다. 있는 컬럼: {list(df.columns)}")

    try:
        import admet_ai
        from admet_ai import ADMETModel
    except ImportError as e:
        raise SystemExit(f"admet-ai 가 없다 ({e}). conda activate chemprop-env")

    smis = df[args.smiles_col].astype(str).tolist()
    print(f"입력 {len(smis)}개 → ADMET 예측 (admet-ai {admet_ai.__version__})")

    model = ADMETModel()
    preds = model.predict(smiles=smis)
    if isinstance(preds, dict):
        preds = pd.DataFrame([preds], index=smis)
    preds.index.name = args.smiles_col
    preds = preds.reset_index()

    merged = df.merge(preds, on=args.smiles_col, how="left")
    dropped_invalid = int(merged[list(HARD_FILTERS)[0]].isna().sum()) \
        if list(HARD_FILTERS)[0] in merged.columns else 0

    # DrugBank 백분위 컬럼은 접미사가 버전에 따라 다르므로 실행 시점에 찾는다
    pct_cols = [c for c in preds.columns if "percentile" in c.lower()]
    pct_suffix = ""
    if pct_cols:
        base = pct_cols[0]
        for k in HARD_FILTERS:
            if base.startswith(k + "_"):
                pct_suffix = base[len(k) + 1:]
                break
        if not pct_suffix:
            pct_suffix = base.split("_", 1)[1] if "_" in base else base
    print(f"예측 컬럼 {len(preds.columns) - 1}개 "
          f"(DrugBank 백분위 {len(pct_cols)}개, 접미사 '{pct_suffix}')")

    merged.to_csv(out / "13_admet_raw.csv", index=False)

    # --- 1. 하드 필터 ---
    missing = [k for k in HARD_FILTERS if k not in merged.columns]
    if missing:
        raise SystemExit(f"탈락 기준 컬럼이 출력에 없다: {missing}")

    flags, counts = {}, {}
    for key, (label, thr) in HARD_FILTERS.items():
        bad = merged[key] > thr
        flags[key] = bad
        counts[key] = int(bad.sum())
        print(f"  {label}: p > {thr} 인 분자 {counts[key]}개")

    any_bad = pd.concat(flags.values(), axis=1).any(axis=1)
    merged["admet_fail"] = any_bad
    merged["admet_fail_reason"] = [
        ", ".join(k for k in HARD_FILTERS if flags[k].iloc[i]) or ""
        for i in range(len(merged))
    ]

    passed = merged[~any_bad].copy()
    print(f"\n하드 필터 통과 {len(passed)} / {len(merged)}개 "
          f"(중복 위반 제외 탈락 {int(any_bad.sum())}개)")
    passed.to_csv(out / "13_admet_filtered.csv", index=False)

    # --- 2. 도킹 순위 그대로 정렬 ---
    order_col = "rank_mean" if "rank_mean" in passed.columns else None
    if order_col is None:
        raise SystemExit("도킹 순위 컬럼 rank_mean 이 없다. 11_top_for_admet.csv 확인")
    passed = passed.sort_values(order_col)

    # --- 3. 최종 후보 ---
    n_final = min(args.n_final, len(passed))
    final = passed.head(n_final).copy()
    final.insert(0, "final_rank", range(1, len(final) + 1))
    final.to_csv(out / "13_final_candidates.csv", index=False)

    short = ["final_rank", "dock_id", "chembl_id", "vina_top", "cnn_affinity_top"]
    short += [c for c in ("QED", "Solubility_AqSolDB", "hERG", "AMES", "DILI")
              if c in final.columns]
    short = [c for c in short if c in final.columns]

    md = [
        "### 5단계 ADMET — 최종 우선순위",
        "",
        f"- 도구: ADMET-AI {admet_ai.__version__} (chemprop D-MPNN, "
        "Therapeutics Data Commons 학습), 로컬 CPU",
        f"- 입력: 4단계 도킹 상위 {len(df)}개 (Vina·CNNaffinity 순위 평균)",
        f"- 예측 항목 {len(preds.columns) - 1}개",
        "",
        "| 단계 | 개수 |",
        "|---|---|",
        f"| 입력 | {len(df)} |",
    ]
    for key, (label, thr) in HARD_FILTERS.items():
        md.append(f"| {label} p > {thr} 탈락 | {counts[key]} |")
    md += [
        f"| 하드 필터 통과 (중복 위반 1회 집계) | {len(passed)} |",
        f"| **최종 후보** | **{len(final)}** |",
        "",
        "판정 기준은 도킹 결과를 보기 전에 `skill/04_admet.md` 에 확정해 둔 것이다. "
        "결과를 보고 문턱값을 조정하지 않았다.",
        "",
        "통과 분자는 ADMET 지표로 다시 줄 세우지 않고 **4단계 도킹 순위를 그대로** 썼다. "
        "ADMET 지표에 가중치를 줄 근거가 없고, 결과를 보고 가중치를 고르면 "
        "원하는 분자가 올라오도록 맞춘 셈이 되기 때문이다.",
        "",
        f"#### 최종 후보 {len(final)}개",
        "",
        final[short].round(3).to_markdown(index=False),
        "",
        "⚠ 위 값은 전부 **계산 예측값**이고 실험으로 측정한 약물동태·독성 자료가 아니다. "
        "실험 검증은 수행하지 않았다.",
        "",
        "⚠ BBB(혈액-뇌 장벽) 투과도는 높을 필요가 없다. 적응증이 HR+/HER2− 유방암이므로 "
        "뇌가 표적 장기가 아니고, 낮은 편이 중추신경계 부작용 위험이 작다.",
    ]
    (out / "13_admet_summary.md").write_text("\n".join(md), encoding="utf-8")

    json.dump({
        "admet_ai": admet_ai.__version__,
        "n_input": len(df),
        "n_invalid_smiles": dropped_invalid,
        "n_pred_columns": len(preds.columns) - 1,
        "drugbank_percentile_suffix": pct_suffix,
        "hard_filters": {k: {"label": v[0], "threshold": v[1], "n_failed": counts[k]}
                         for k, v in HARD_FILTERS.items()},
        "n_passed": len(passed),
        "n_failed": int(any_bad.sum()),
        "n_final": len(final),
        "order_by": order_col,
    }, open(out / "13_admet_stats.json", "w"), indent=1)

    print(f"\n최종 후보 {len(final)}개")
    print(final[short].round(3).to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
