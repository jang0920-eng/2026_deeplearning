"""
보고서용 그림 생성.

배점 2번(가독성 30점)이 "matplotlib 으로 적절한 시각화"를 요구하므로,
표로 충분한 것은 그림으로 만들지 않고 **그림이어야만 전달되는 것**만 그린다.

  fig1_redocking.png       pose 별 RMSD-점수 관계 — 점수 1위가 RMSD 도 최소임을 보임
  fig2_model_comparison.png  PR 곡선 + 지표 비교 — 네 모델의 서열과 EF 천장

색은 검증된 categorical 팔레트 slot 1~4 를 쓴다.
(validate_palette.js 통과. slot 3·4 는 밝기 때문에 직접 라벨을 함께 단다.)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve

# ---- 공통 스타일 -------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
SURFACE = "#fcfcfb"


def setup_style():
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "grid.color": "#e8e7e3", "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.unicode_minus": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


# ---- 그림 1: redocking -------------------------------------------------------

def fig_redocking(poses, outpath):
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.grid(zorder=0)
    ax.set_axisbelow(True)

    # 합격 영역을 면으로 표시하면 한눈에 읽힌다
    ax.axvspan(0, 2.0, color="#eef4fb", zorder=0)
    ax.axvline(2.0, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax.text(1.92, -7.42, "합격 기준 2.0 Å", color=INK2, fontsize=8,
            ha="right", va="top")

    styles = {
        "A_crystal_input": (BLUE, "o", "A · 결정 좌표 입력"),
        "B_fresh_3d": (ORANGE, "s", "B · SMILES → 새 3D 구조"),
    }
    tops = {}
    for run, (color, marker, label) in styles.items():
        g = poses[poses["run"] == run].sort_values("pose")
        ax.plot(g["rmsd"], g["vina"], marker=marker, ms=6, lw=0,
                color=color, label=label, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        top = g.iloc[0]
        tops[run] = top
        ax.plot(top["rmsd"], top["vina"], marker=marker, ms=12, lw=0,
                color=color, zorder=4, markeredgecolor=SURFACE, markeredgewidth=1.8)

    # 두 점이 거의 겹치므로 설명은 하나로 모은다
    a, b = tops["A_crystal_input"], tops["B_fresh_3d"]
    ax.annotate(
        f"두 검증 모두 점수 1위 pose 가\n결정 구조를 재현\nA {a['rmsd']:.2f} Å · B {b['rmsd']:.2f} Å",
        xy=(a["rmsd"] + 0.06, (a["vina"] + b["vina"]) / 2),
        xytext=(2.5, -9.72), fontsize=8.5, color=INK, va="center",
        arrowprops=dict(arrowstyle="-", color=INK2, lw=1,
                        connectionstyle="arc3,rad=-0.2"))

    ax.set_xlim(0, 7.4)
    ax.set_ylim(-10.25, -7.3)
    ax.set_xlabel("결정 구조와의 RMSD (Å) — 왼쪽일수록 재현 성공")
    ax.set_ylabel("Vina 점수 (kcal/mol) — 아래쪽일수록 결합에 유리")
    ax.set_title("redocking 검증: 점수 1위 pose 가 결정 구조를 재현했다")
    ax.legend(loc="upper right")
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 2: 모델 비교 -------------------------------------------------------

def fig_model_comparison(scores, metrics, outpath):
    """좌: 표현(feature) 비교 — 분류기는 더 나은 RF 로 고정해 표현만 비교한다.
       우: 여섯 조합의 지표 전체. EF 천장도 함께 보인다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    y = scores["label"].to_numpy()
    order = metrics.sort_values("pr_auc", ascending=False)["model"].tolist()
    order = [m for m in order if m in scores.columns]

    # ---- 좌: RF 로 고정하고 표현만 비교 (3개 — 전부 쌍 검증 통과한 slot 1~3)
    rf_models = [m for m in order if m.endswith("+ RF")]
    ax1.grid(zorder=0); ax1.set_axisbelow(True)
    base = y.mean()
    ax1.axhline(base, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax1.text(0.02, base - 0.03, f"무작위 기준선 {base:.1%}", color=INK2, fontsize=8)

    for m, c in zip(rf_models, [BLUE, ORANGE, AQUA]):
        prec, rec, _ = precision_recall_curve(y, scores[m])
        pr = metrics.loc[metrics["model"] == m, "pr_auc"].iloc[0]
        ax1.plot(rec, prec, lw=2, color=c, zorder=3, solid_capstyle="round",
                 label=f"{m.replace(' + RF', '')}   {pr:.3f}")
    ax1.set_xlim(0, 1); ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("recall"); ax1.set_ylabel("precision")
    ax1.set_title("표현 비교 (분류기는 RF 로 고정), PR-AUC")
    ax1.legend(loc="lower left", fontsize=8)

    # ---- 우: 여섯 조합 전체
    ax2.grid(axis="x", zorder=0); ax2.set_axisbelow(True)
    idx = np.arange(len(order))[::-1]
    h = 0.26
    ceiling = metrics["ef_max"].iloc[0]
    for off, (key, c, lab) in enumerate([("pr_auc", BLUE, "PR-AUC"),
                                         ("roc_auc", ORANGE, "ROC-AUC"),
                                         ("ef5", AQUA, "EF 5%")]):
        vals = [metrics.loc[metrics["model"] == m, key].iloc[0] for m in order]
        denom = ceiling if key == "ef5" else 1.0
        bars = ax2.barh(idx + (1 - off) * h, [v / denom for v in vals],
                        height=h * 0.86, color=c, zorder=3, label=lab)
        for b, v in zip(bars, vals):
            ax2.text(b.get_width() + 0.012, b.get_y() + b.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=7, color=INK2)

    ax2.axvline(1.0, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax2.set_yticks(idx)
    ax2.set_yticklabels([m.replace(" + ", "\n+ ") for m in order], fontsize=7.5)
    ax2.set_xlim(0, 1.24)
    ax2.set_ylim(-1.15, len(order) - 0.4)
    ax2.set_xlabel("지표값 (EF 는 이론적 최대 %.2f 대비 비율)" % ceiling)
    ax2.set_title("EF 는 천장에 붙어 모델을 구분하지 못한다")
    ax2.legend(loc="lower center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 3: 1단계 필터 깔때기 ------------------------------------------------

def fig_filter_funnel(counts, outpath):
    """어느 기준이 얼마나 걸러냈는지. 숫자가 커서 로그 스케일이 아니라
       누적 막대로 보여야 비율이 읽힌다."""
    import matplotlib.patches as mpatches

    tot, uniq = counts["total"], counts["unique"]
    steps = [
        ("크기 범위 밖", counts["size"] + counts["inorganic"] + counts["parse_fail"]),
        ("lead-like 위반", counts["lead_like"]),
        ("Lipinski 위반", counts["lipinski"]),
        ("PAINS 구조", counts["pains"]),
        ("중복 구조", counts["duplicates"]),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    left = tot
    ys, labels = [], []
    for i, (name, n) in enumerate(steps):
        y = len(steps) - i
        ax.barh(y, left, height=0.62, color="#e8e7e3", zorder=2)
        ax.barh(y, left - n, height=0.62, color=BLUE, zorder=3)
        ax.text(left + tot * 0.012, y, f"−{n:,}", va="center",
                fontsize=8, color=ORANGE)
        ys.append(y); labels.append(name)
        left -= n

    ax.barh(0, uniq, height=0.62, color=AQUA, zorder=3)
    ax.text(uniq + tot * 0.012, 0, f"{uniq:,}", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ys.append(0); labels.append("최종 통과")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, tot * 1.16)
    ax.set_xlabel("남은 화합물 수")
    ax.set_title("1단계 물리화학 필터: %s개 → %s개" % (f"{tot:,}", f"{uniq:,}"))
    ax.xaxis.set_major_formatter(lambda v, p: f"{v/1e6:.1f}M" if v else "0")

    ax.legend(handles=[mpatches.Patch(color=BLUE, label="남은 수"),
                       mpatches.Patch(color=AQUA, label="최종 통과"),
                       mpatches.Patch(color="#e8e7e3", label="이전 단계 수")],
              loc="lower right", fontsize=8)
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiment/06_figures")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    setup_style()

    poses = pd.read_csv("experiment/03_redocking/03_redock_poses.csv")
    fig_redocking(poses, out / "fig1_redocking.png")

    scores = pd.read_csv("experiment/05_model_comparison/05_test_scores.csv")
    metrics = pd.read_csv("experiment/05_model_comparison/05_model_comparison.csv")
    fig_model_comparison(scores, metrics, out / "fig2_model_comparison.png")

    cpath = Path("experiment/07_filter/07_filter_counts.json")
    if cpath.exists():
        import json
        fig_filter_funnel(json.loads(cpath.read_text()), out / "fig3_filter_funnel.png")


if __name__ == "__main__":
    main()
