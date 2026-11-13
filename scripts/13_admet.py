"""
13. ADMET 예측 — 도킹 상위 100개에서 최종 우선순위를 추린다 (깔때기 5단계).

도구: ADMET-AI (chemprop D-MPNN, Therapeutics Data Commons 학습)
판정 기준은 `skill/04_admet.md` 에 적어 둔 것을 그대로 쓴다. 여기서 재정의하지 않는다.

  1. 하드 필터 — hERG / AMES / DILI 예측값이 **승인 약물 분포의 상위 10% 안**
     (백분위 > 90) 이면 탈락. 즉 "이미 시판 중인 약의 90% 보다 나쁘다"는 뜻이다.
  2. 통과 분자를 4단계 도킹 순위(rank_mean)로 그대로 정렬
  3. 상위 N개를 최종 후보로 제시. 통과 분자가 N 미만이면 전부 제시하고 그 수를 보고한다

ADMET 지표에 임의의 가중치를 주어 도킹 순위와 섞지 않는다.
가중치를 정할 근거가 없고, 결과를 보고 가중치를 고르면 맞춘 셈이 되기 때문이다.

⚠ 절대 확률 0.5 를 기준으로 쓰지 않는 이유는 실측으로 확인했다.
   ADMET-AI 가 함께 제공하는 승인 약물 2,845개에서 p > 0.5 인 비율이
   DILI 44.3%, hERG 37.2%, AMES 20.9% 다. 0.5 로 자르면 시판 중인 약의
   상당수가 탈락하므로 안전성 기준으로 쓸 수 없다. 자세한 내용은 skill/04_admet.md.

실행:
    conda activate chemprop-env
    python scripts/13_admet.py --input experiment/11_docking/11_top_for_admet.csv \
                               --outdir experiment/13_admet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# 한국어 Windows 콘솔은 기본 코드페이지가 cp949 다. admet-ai 내부의 lightning
# 진행바(rich)가 '•'(U+2022) 를 쓰려다 UnicodeEncodeError 로 죽는다.
# 예측이 끝난 뒤 진행바 teardown 에서 터지므로 원인을 찾기 어렵다.
# 출력 스트림을 utf-8 로 바꾸고 인코딩 불가 문자는 대체한다. Linux 에서는 무해하다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# skill/04_admet.md 3-1 에 적어 둔 탈락 기준.
# 값은 승인 약물 분포에서의 백분위이고, 100 에 가까울수록 나쁘다.
HARD_FILTERS = {
    "hERG": "hERG 저해 (심장 독성)",
    "AMES": "AMES 돌연변이성 (유전독성)",
    "DILI": "약물 유발 간손상",
}
PCT_CUTOFF = 90.0
# 기준이 한 값에 의존하지 않는다는 것을 보이기 위해 함께 보고한다
SENSITIVITY = (80.0, 90.0, 95.0)

# 탈락시키지 않고 표에 적어 두는 지표 (skill/04_admet.md 3-2)
REPORT_ONLY = [
    "QED", "Solubility_AqSolDB", "Caco2_Wang", "HIA_Hou", "Bioavailability_Ma",
    "BBB_Martins", "PPBR_AZ", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
    "CYP1A2_Veith", "CYP2C9_Veith", "CYP2C19_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
    "ClinTox", "Carcinogens_Lagunin", "LD50_Zhu",
]

N_FINAL = 30


def find_percentile_suffix(columns):
    """DrugBank 백분위 컬럼의 접미사를 찾는다.

    접미사는 admet-ai 버전에 따라 달라질 수 있어 하드코딩하지 않는다.
    `molecular_weight` 처럼 밑줄이 들어간 항목 이름이 있으므로
    단순히 첫 밑줄로 자르면 틀린다. 긴 이름부터 맞춰 본다.
    """
    pct = [c for c in columns if "percentile" in c.lower()]
    base = [c for c in columns if c not in pct]
    for name in sorted(base, key=len, reverse=True):
        for c in pct:
            if c.startswith(name + "_"):
                return c[len(name) + 1:], pct
    return "", pct


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

    suffix, pct_cols = find_percentile_suffix(
        [c for c in preds.columns if c != args.smiles_col])
    if not suffix:
        raise SystemExit(
            "DrugBank 백분위 컬럼을 찾지 못했다. ADMETModel(drugbank_path=...) 확인 필요")
    print(f"예측 컬럼 {len(preds.columns) - 1}개 "
          f"(DrugBank 백분위 {len(pct_cols)}개, 접미사 '{suffix}')")

    merged = df.merge(preds, on=args.smiles_col, how="left")
    n_invalid = len(df) - int(merged[list(HARD_FILTERS)[0]].notna().sum())
    if n_invalid:
        print(f"⚠ SMILES 를 읽지 못한 분자 {n_invalid}개")
    merged.to_csv(out / "13_admet_raw.csv", index=False)

    # --- 1. 하드 필터 (승인 약물 분포 백분위 기준) ---
    pct_of = {k: f"{k}_{suffix}" for k in HARD_FILTERS}
    missing = [c for c in pct_of.values() if c not in merged.columns]
    if missing:
        raise SystemExit(f"탈락 기준 컬럼이 출력에 없다: {missing}")

    flags, counts = {}, {}
    for key, label in HARD_FILTERS.items():
        bad = merged[pct_of[key]] > PCT_CUTOFF
        flags[key] = bad
        counts[key] = int(bad.sum())
        print(f"  {label}: 승인 약물 백분위 > {PCT_CUTOFF:.0f} 인 분자 {counts[key]}개")

    any_bad = pd.concat(flags.values(), axis=1).any(axis=1)
    merged["admet_fail"] = any_bad
    merged["admet_fail_reason"] = [
        ", ".join(k for k in HARD_FILTERS if bool(flags[k].iloc[i]))
        for i in range(len(merged))
    ]

    # 기준이 한 값에 의존하지 않음을 보인다
    sens = {}
    for cut in SENSITIVITY:
        bad = pd.concat([merged[pct_of[k]] > cut for k in HARD_FILTERS],
                        axis=1).any(axis=1)
        sens[cut] = int((~bad).sum())
    print("  기준 민감도 (통과 개수): "
          + ", ".join(f"백분위>{int(c)} → {n}" for c, n in sens.items()))

    passed = merged[~any_bad].copy()
    print(f"\n하드 필터 통과 {len(passed)} / {len(merged)}개 "
          f"(중복 위반 1회 집계, 탈락 {int(any_bad.sum())}개)")
    passed.to_csv(out / "13_admet_filtered.csv", index=False)

    # --- 2. 도킹 순위 그대로 정렬 ---
    if "rank_mean" not in passed.columns:
        raise SystemExit("도킹 순위 컬럼 rank_mean 이 없다. 11_top_for_admet.csv 확인")
    passed = passed.sort_values("rank_mean")

    # --- 3. 최종 후보 ---
    final = passed.head(min(args.n_final, len(passed))).copy()
    final.insert(0, "final_rank", range(1, len(final) + 1))
    final.to_csv(out / "13_final_candidates.csv", index=False)

    short = ["final_rank", "dock_id", "chembl_id", "vina_top", "cnn_affinity_top"]
    short += [pct_of[k] for k in HARD_FILTERS]
    short += [c for c in ("QED", "Solubility_AqSolDB") if c in final.columns]
    short = [c for c in short if c in final.columns]
    rename = {pct_of[k]: f"{k}_백분위" for k in HARD_FILTERS}

    md = [
        "### 5단계 ADMET — 최종 우선순위",
        "",
        f"- 도구: ADMET-AI {admet_ai.__version__} (chemprop D-MPNN, "
        "Therapeutics Data Commons 학습), 로컬 CPU",
        f"- 입력: 4단계 도킹 상위 {len(df)}개 (Vina·CNNaffinity 순위 평균)",
        f"- 예측 항목 {len(preds.columns) - 1 - len(pct_cols)}개 "
        f"(+ 승인 약물 분포 백분위 {len(pct_cols)}개)",
        "",
        "#### 깔때기",
        "",
        "| 단계 | 개수 |",
        "|---|---|",
        f"| 입력 | {len(df)} |",
    ]
    for key, label in HARD_FILTERS.items():
        md.append(f"| {label} 백분위 > {PCT_CUTOFF:.0f} 탈락 | {counts[key]} |")
    md += [
        f"| 하드 필터 통과 (중복 위반 1회 집계) | {len(passed)} |",
        f"| **최종 후보** | **{len(final)}** |",
        "",
        "#### 기준 민감도",
        "",
        "문턱값 하나에 결과가 좌우되지 않는다는 것을 보이기 위해 함께 적는다.",
        "",
        "| 백분위 기준 | 통과 개수 |",
        "|---|---|",
    ]
    for cut, n in sens.items():
        mark = " ← 채택" if cut == PCT_CUTOFF else ""
        md.append(f"| > {int(cut)} 탈락 | {n}{mark} |")
    md += [
        "",
        "#### 판정 기준",
        "",
        f"hERG · AMES · DILI 예측값이 **승인 약물 2,845개 분포의 상위 "
        f"{100 - PCT_CUTOFF:.0f}% 안**(백분위 > {PCT_CUTOFF:.0f})이면 탈락시켰다. "
        "\"이미 시판 중인 약의 90% 보다 나쁘다\"는 뜻이다.",
        "",
        "절대 확률 0.5 를 쓰지 않았다. 승인 약물에서 p > 0.5 인 비율이 "
        "DILI 44.3%, hERG 37.2%, AMES 20.9% 이므로, 0.5 로 자르면 "
        "**시판 중인 약의 상당수가 탈락한다.** 안전성 기준으로 쓸 수 없는 값이다.",
        "",
        "통과 분자는 ADMET 지표로 다시 줄 세우지 않고 **4단계 도킹 순위를 그대로** 썼다. "
        "ADMET 지표에 가중치를 줄 근거가 없고, 결과를 보고 가중치를 고르면 "
        "원하는 분자가 올라오도록 맞춘 셈이 되기 때문이다.",
        "",
        f"#### 최종 후보 {len(final)}개",
        "",
        final[short].rename(columns=rename).round(1).to_markdown(index=False),
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
        "n_invalid_smiles": n_invalid,
        "n_pred_columns": len(preds.columns) - 1,
        "n_percentile_columns": len(pct_cols),
        "percentile_suffix": suffix,
        "criterion": f"drugbank_approved_percentile > {PCT_CUTOFF}",
        "hard_filters": {k: {"label": v, "column": pct_of[k],
                             "n_failed": counts[k]}
                         for k, v in HARD_FILTERS.items()},
        "sensitivity_n_passed": {str(int(c)): n for c, n in sens.items()},
        "n_passed": len(passed),
        "n_failed": int(any_bad.sum()),
        "n_final": len(final),
        "order_by": "rank_mean",
    }, open(out / "13_admet_stats.json", "w"), indent=1)

    print(f"\n최종 후보 {len(final)}개")
    print(final[short].rename(columns=rename).round(1).to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
