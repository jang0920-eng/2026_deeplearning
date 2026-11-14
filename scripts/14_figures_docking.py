"""
보고서용 그림 — 도킹·ADMET·최종 후보 (4·5단계).

  fig8_docking_scores.png   (a) Vina 점수 분포 / (b) Vina vs CNNaffinity /
                            (c) 2단계 ML 점수 vs 4단계 도킹 점수
  fig9_admet_funnel.png     (a) 항목별 탈락 수 / (b) 백분위 기준 민감도
  fig10_candidates.png      최종 후보 2D 구조 격자 (보고서 필수 포함 3번)

색·서체는 `06_figures.py` 의 것을 그대로 불러 쓴다. 새 색을 만들지 않는다.

**입력이 없으면 그 그림만 건너뛴다.** 도킹이 끝나기 전에도 돌릴 수 있고,
부분 결과(예: 375/996)로 돌리면 그린 분자 수를 제목에 그대로 적는다.
부분 결과로 그린 그림을 보고서에 넣지 않는다 — 확인용이다.

(c) 패널을 넣는 이유: 2단계 ML 점수와 4단계 도킹 점수가 강하게 상관하면
4단계가 새로운 정보를 주지 못한다는 뜻이다. 깔때기를 쌓은 근거가 되므로
상관이 **낮다는 것** 또는 높다는 것을 실측해서 보인다. 결과를 보기 전에
이 패널을 그리기로 정했다.

실행:
    python scripts/14_figures_docking.py
    python scripts/14_figures_docking.py --docking experiment/11_docking
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_DOCK_TOTAL = 996          # 도킹 입력 수 (10단계 리간드 준비 결과)


def load_style():
    """06_figures.py 의 스타일·색을 재사용한다 (모듈 이름이 숫자로 시작해 import 불가)."""
    spec = importlib.util.spec_from_file_location("fig06", HERE / "06_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_style()
    return mod


def partial_note(n):
    """부분 결과면 제목에 그대로 적는다. 숨기지 않는다."""
    if n < N_DOCK_TOTAL:
        return f"  ⚠ 부분 결과 {n}/{N_DOCK_TOTAL}"
    return ""


# ---- 그림 8: 도킹 점수 --------------------------------------------------------

def fig_docking_scores(S, dock, n_top, outpath):
    ok = dock.dropna(subset=["vina_top", "cnn_affinity_top"]).copy()
    n = len(ok)
    note = partial_note(n)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7))

    # (a) Vina 점수 분포 — 상위 n_top 경계를 함께 그린다
    ax = axes[0]
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.hist(ok["vina_top"], bins=40, color=S.BLUE, zorder=3)
    cut = ok["vina_top"].nsmallest(min(n_top, n)).max()
    ax.axvline(cut, color=S.ORANGE, lw=1.6, ls="--", zorder=4)
    ax.annotate(f"상위 {min(n_top, n)}개 경계\n{cut:.2f} kcal/mol",
                xy=(cut, 1), xycoords=("data", "axes fraction"),
                xytext=(-7, -8), textcoords="offset points",
                ha="right", va="top", fontsize=8.2, color=S.ORANGE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=S.SURFACE,
                          edgecolor="none"))
    n_pos = int((ok["vina_top"] > 0).sum())
    if n_pos:
        ax.text(0.98, 0.72, f"양수 점수 {n_pos}개\n(결합 불성립)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
                color=S.INK2)
    ax.set_xlabel("Vina 점수 (kcal/mol) — 낮을수록 결합에 유리")
    ax.set_ylabel("분자 수")
    ax.set_title(f"(a) 도킹 점수 분포 (n={n:,}){note}", loc="left", pad=10)

    # (b) Vina vs CNNaffinity — 두 점수가 같은 순위를 주는가
    ax = axes[1]
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.plot(ok["vina_top"], ok["cnn_affinity_top"], "o", ms=3.4, lw=0,
            color=S.MUTED, alpha=0.55, zorder=3, label=f"전체 (n={n:,})")

    # 두 점수의 순위 평균으로 고른 상위 n_top 을 강조한다 (11 노트북과 같은 규칙)
    # 11 노트북과 같이 rank_mean 으로 정렬해 앞에서 n_top 개를 가져온다.
    # `rank_mean <= n_top` 으로 자르면 두 점수가 어긋날 때 개수가 모자란다.
    r = (ok["vina_top"].rank(ascending=True)
         + ok["cnn_affinity_top"].rank(ascending=False)) / 2
    sel = ok.loc[r.nsmallest(min(n_top, n)).index]
    ax.plot(sel["vina_top"], sel["cnn_affinity_top"], "s", ms=4.6, lw=0,
            color=S.BLUE, zorder=4, markeredgecolor=S.SURFACE,
            markeredgewidth=0.8, label=f"순위 평균 상위 {len(sel)}개")

    rho = ok[["vina_top", "cnn_affinity_top"]].corr(method="spearman").iloc[0, 1]
    ax.annotate(f"Spearman ρ = {rho:.3f}",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
                fontsize=8.6, color=S.INK2,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=S.SURFACE,
                          edgecolor=S.MUTED, linewidth=0.7))
    ax.set_xlabel("Vina 점수 (kcal/mol)")
    ax.set_ylabel("CNNaffinity — 높을수록 유리")
    ax.set_title("(b) 물리 점수와 CNN 점수의 일치도", loc="left", pad=10)
    ax.legend(loc="lower left", fontsize=8)

    # (c) 2단계 ML 점수 vs 4단계 도킹 점수 — 단계가 독립적인 정보를 주는가
    ax = axes[2]
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    if "ml_score" in ok.columns and ok["ml_score"].notna().any():
        ax.plot(ok["ml_score"], ok["vina_top"], "o", ms=3.4, lw=0,
                color=S.AQUA, alpha=0.6, zorder=3)
        rho2 = ok[["ml_score", "vina_top"]].corr(method="spearman").iloc[0, 1]
        ax.annotate(f"Spearman ρ = {rho2:.3f}",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
                    fontsize=8.6, color=S.INK2,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=S.SURFACE,
                              edgecolor=S.MUTED, linewidth=0.7))
        ax.set_xlabel("2단계 RF 예측 확률")
        ax.set_ylabel("4단계 Vina 점수 (kcal/mol)")
    else:
        ax.text(0.5, 0.5, "ml_score 컬럼 없음", transform=ax.transAxes,
                ha="center", va="center", color=S.MUTED)
    ax.set_title("(c) 2단계 ML 점수와 4단계 도킹 점수", loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 9: ADMET 깔때기 -----------------------------------------------------

def fig_admet_funnel(S, stats, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    # (a) 항목별 탈락 수 — 중복 위반이 있으므로 합이 전체 탈락 수와 다르다
    ax = axes[0]
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    hf = stats["hard_filters"]
    names = [hf[k]["label"] for k in hf]
    vals = [hf[k]["n_failed"] for k in hf]
    yy = np.arange(len(names))[::-1]
    bars = ax.barh(yy, vals, height=0.55, color=S.ORANGE, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                f"{v}", va="center", fontsize=8.5, color=S.INK2)
    ax.set_yticks(yy, names, fontsize=8.5)
    ax.set_xlim(0, max(max(vals), 1) * 1.22)
    ax.set_xlabel("탈락 분자 수 (승인 약물 백분위 > 90)")
    ax.set_title(f"(a) 항목별 탈락 — 입력 {stats['n_input']}개 중 "
                 f"{stats['n_failed']}개 탈락", loc="left", pad=10)
    # 축 안에 두면 막대와 겹친다. x 축 라벨 아래 여백으로 내린다.
    ax.annotate("한 분자가 여러 항목을 위반할 수 있어 합이 전체 탈락 수와 다르다",
                xy=(0, -0.36), xycoords="axes fraction", ha="left", va="top",
                fontsize=7.8, color=S.MUTED)
    ax.tick_params(axis="y", length=0)
    for s in ("left",):
        ax.spines[s].set_visible(False)

    # (b) 기준 민감도 — 결과가 문턱값 하나에 좌우되지 않음을 보인다
    ax = axes[1]
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    sens = stats["sensitivity_n_passed"]
    cuts = sorted(sens, key=lambda c: float(c))
    # "drugbank_approved_percentile > 90.0" 에서 90 을 뽑는다.
    # rstrip(".0") 은 문자 단위라 "90.0" 을 "9" 로 만든다 — 수로 바꿔서 비교한다.
    adopted = float(stats["criterion"].split(">")[-1])
    colors = [S.BLUE if float(c) == adopted else S.MUTED for c in cuts]
    bars = ax.bar([f"> {c}" for c in cuts], [sens[c] for c in cuts],
                  width=0.52, color=colors, zorder=3)
    for b, c in zip(bars, cuts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{sens[c]}" + ("\n← 채택" if float(c) == adopted else ""),
                ha="center", va="bottom", fontsize=8.4,
                color=S.INK if float(c) == adopted else S.INK2)
    ax.set_ylim(0, max(sens.values()) * 1.3)
    ax.set_xlabel("탈락 기준 — 승인 약물 분포 백분위")
    ax.set_ylabel("통과 분자 수")
    # 제목으로 결론을 단정하지 않는다. 기준을 바꾼 결과는 막대가 보여 준다.
    ax.set_title("(b) 탈락 기준에 따른 통과 개수", loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 10: 최종 후보 구조 --------------------------------------------------

def fig_candidates(S, final, outpath, n_show=8):
    """최종 후보의 2D 구조. 보고서 필수 포함 3번(후보 물질 구조)에 쓴다.

    RDKit 이 그린 격자 이미지를 matplotlib 축에 얹고, 분자마다 도킹 점수를
    캡션으로 단다. 점수는 **계산 예측값**이므로 단위를 반드시 함께 적는다.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    RDLogger.DisableLog("rdApp.*")

    df = final.head(n_show)
    mols, legends = [], []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        mols.append(m)
        # RDKit 격자 범례는 줄 간격을 좁게 잡아 두 줄로 쓰면 글자가 겹친다.
        # 한 줄로 쓰고 그만큼 칸 너비를 넓힌다.
        # 가운뎃점(·)은 RDKit 범례 글꼴에 없어 빈칸으로 빠진다. 공백으로 띄운다.
        legends.append(f"{int(r['final_rank'])}. {r['chembl_id']}     "
                       f"Vina {r['vina_top']:.2f}     "
                       f"CNNaff {r['cnn_affinity_top']:.2f}")
    if not mols:
        print("건너뜀: 그릴 구조가 없다")
        return

    ncol = 4
    img = Draw.MolsToGridImage(mols, molsPerRow=ncol, subImgSize=(340, 265),
                               legends=legends, returnPNG=False)
    # 축 비율을 이미지 비율에 맞춰야 위아래 빈 공간이 생기지 않는다
    fig, ax = plt.subplots(figsize=(img.width / 150, img.height / 150))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"계산 예측 기준 상위 {len(mols)}개 후보의 2D 구조",
                 loc="left", pad=8)
    fig.text(0.0, -0.01,
             "Vina 점수 단위는 kcal/mol. 두 점수 모두 도킹 계산값이고 "
             "실험으로 측정한 결합 친화도가 아니다.",
             ha="left", va="top", fontsize=7.8, color=S.INK2)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print("저장:", outpath)


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docking", default="experiment/11_docking")
    ap.add_argument("--admet", default="experiment/13_admet")
    ap.add_argument("--outdir", default="experiment/06_figures")
    ap.add_argument("--n-top", type=int, default=100,
                    help="도킹에서 5단계로 넘기는 개수")
    args = ap.parse_args()

    S = load_style()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    drew = 0

    dpath = Path(args.docking) / "11_docking_results.csv"
    if dpath.exists():
        dock = pd.read_csv(dpath).drop_duplicates("dock_id", keep="last")
        fig_docking_scores(S, dock, args.n_top, out / "fig8_docking_scores.png")
        drew += 1
    else:
        print(f"건너뜀 (입력 없음): {dpath}")

    spath = Path(args.admet) / "13_admet_stats.json"
    if spath.exists():
        fig_admet_funnel(S, json.loads(spath.read_text(encoding="utf-8")),
                         out / "fig9_admet_funnel.png")
        drew += 1
    else:
        print(f"건너뜀 (입력 없음): {spath}")

    fpath = Path(args.admet) / "13_final_candidates.csv"
    if fpath.exists():
        fig_candidates(S, pd.read_csv(fpath), out / "fig10_candidates.png")
        drew += 1
    else:
        print(f"건너뜀 (입력 없음): {fpath}")

    print(f"\n그림 {drew}개 생성")


if __name__ == "__main__":
    main()
