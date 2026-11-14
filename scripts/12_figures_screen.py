"""
보고서용 그림 — 스크리닝 단계 (2·3단계).

  fig4_ml_screen.png        좌: 142만 개 예측 점수 분포 / 우: 미지 분자 회수 곡선
  fig5_diversity.png        좌: 선별 방식별 최근접 유사도 / 우: 선별 분자의 물성 공간
  fig6_pipeline_funnel.png  깔때기 전체 — 네 자리 수가 넘는 범위라 로그 축 가로 막대로 그린다

(fig3_filter_funnel.png 은 1단계 필터 내부 분해로, `06_figures.py` 가 그린다)

색·서체는 `06_figures.py` 의 것을 그대로 불러 쓴다.
새 색을 만들지 않으므로 팔레트 검증을 다시 하지 않는다.
한 그림에 계열이 둘일 때는 색 외에 **표식 모양**으로도 구분해 색만으로 읽히지 않게 했다.

표로 충분한 것은 그림으로 만들지 않는다 (분량이 짧을수록 가산점).
그래서 "골격 수 2개 비교" 같은 것은 막대 두 개로 그리지 않고 글상자로 적는다.

실행:
    python scripts/12_figures_screen.py
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


def plain_log_ticks(ax, axis="x", unit=""):
    """로그 축 눈금을 10^n 수식 대신 보통 숫자로 쓴다.

    matplotlib 의 로그 눈금은 `\\mathdefault{10^{-4}}` 라는 수식으로 그려지는데,
    이 수식은 한글 글꼴을 그대로 따라가고 그 글꼴에는 U+2212 마이너스가 없어
    지수가 네모로 나온다. 읽기에도 "0.01%" 쪽이 "10^-2" 보다 낫다.
    """
    def fmt(v, _pos):
        if v <= 0:
            return ""
        if v >= 1000:
            return f"{v / 1000:,.0f}K{unit}" if v < 1e6 else f"{v / 1e6:,.0f}M{unit}"
        if v >= 1:
            return f"{v:,.0f}{unit}"
        return f"{v:g}{unit}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(plt.FuncFormatter(fmt))
    target.set_minor_formatter(plt.NullFormatter())


def load_style():
    """06_figures.py 의 스타일·색을 재사용한다 (모듈 이름이 숫자로 시작해 import 불가)."""
    spec = importlib.util.spec_from_file_location("fig06", HERE / "06_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_style()
    return mod


# ---- 그림 3: 깔때기 ----------------------------------------------------------

def fig_funnel(S, stats08, stats09, stats10, outpath):
    """단계별 분자 수. 2,897,819 → 996 은 네 자리 수 범위이므로 로그 축으로 그린다.

    ⚠ 3단계에서 1,000개를 뽑았지만 도킹 입력은 996개다. 3D 구조 생성에서 4개가
    실패했다(MMFF 힘 상수 없음 3, embed 실패 1). 그림에서 1,000 → 4단계로 바로
    이으면 숫자가 README·10단계 요약과 어긋나므로 **별도 칸으로 드러낸다.**
    """
    done = [
        ("ChEMBL 37 전체", 2897819),
        ("1단계 물리화학 필터", 1423957),
        ("2단계 ML 활성 예측", stats08["top_n_saved"]),
        ("3단계 다양성 선별", stats09["selected"]),
    ]
    if stats10:
        done.append(("리간드 3D 구조 생성", stats10["prepared"]))
    planned = [("4단계 GNINA 도킹", 100), ("5단계 ADMET", 20)]

    labels = [d[0] for d in done + planned]
    vals = [d[1] for d in done + planned]
    y = np.arange(len(vals))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xscale("log")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    for i, (yy, v) in enumerate(zip(y, vals)):
        is_planned = i >= len(done)
        ax.barh(yy, v, height=0.52, zorder=3,
                color=S.SURFACE if is_planned else S.BLUE,
                edgecolor=S.MUTED if is_planned else S.SURFACE,
                linewidth=1.0, hatch="///" if is_planned else None)
        note = ""
        if is_planned:
            note = "  (예정)"
        elif stats10 and labels[i] == "리간드 3D 구조 생성":
            # 1,000 과 996 은 로그 축에서 길이가 같아 보인다. 숫자로만 읽히므로 적는다.
            note = f"  (1,000개 중 {stats10['input'] - stats10['prepared']}개 생성 실패)"
        ax.text(v * 1.35, yy, f"{v:,}{note}",
                va="center", ha="left", fontsize=8.5,
                color=S.MUTED if is_planned else S.INK)

    ax.set_yticks(y, labels)
    ax.set_xlim(5, 2.9e7)
    plain_log_ticks(ax, "x")
    ax.set_xlabel("분자 수 (로그 축)")
    ax.set_title("단계별 통과 분자 수", loc="left", pad=10)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 4: ML 스크리닝 -----------------------------------------------------

def fig_ml_screen(S, hist, known, stats08, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) 142만 개 점수 분포 — 꼬리가 얇아 y 를 로그로 둔다
    ax = axes[0]
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    centers = (hist["bin_left"] + hist["bin_right"]) / 2
    ax.bar(centers, hist["count"], width=0.0092, color=S.BLUE, zorder=3)
    ax.set_yscale("log")
    plain_log_ticks(ax, "y")
    ax.set_xlim(0, 1)
    ax.set_xlabel("RF 예측 확률")
    ax.set_ylabel("분자 수 (로그 축)")
    ax.set_title("(a) 라이브러리 142만 개 예측 점수 분포", loc="left", pad=10)

    cut = stats08["score_at_top_n"]
    ax.axvline(cut, color=S.ORANGE, lw=1.6, ls="--", zorder=4)
    # 막대 위에 겹치지 않도록 축 상단에 붙이고 흰 상자를 깐다
    ax.annotate(f"상위 {stats08['top_n_saved']:,}개 경계 {cut:.3f}",
                xy=(cut, 1), xycoords=("data", "axes fraction"),
                xytext=(7, -8), textcoords="offset points",
                ha="left", va="top", fontsize=8.2, color=S.ORANGE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=S.SURFACE,
                          edgecolor="none"))

    # (b) 미지 분자 회수 곡선 — 라이브러리를 배경으로 한 농축
    ax = axes[1]
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    n_lib = stats08["library_scored"]
    grid = np.logspace(np.log10(1 / n_lib), 0, 300)

    series = [
        (1, S.BLUE, "o", "test 활성 분자"),
        (0, S.ORANGE, "s", "test 비활성 분자"),
    ]
    for label, color, marker, name in series:
        r = known[(known["split"] == "test") & (known["label"] == label)
                  & known["in_library"]]["rank"].to_numpy()
        if len(r) == 0:
            continue
        frac = np.array([(r <= g * n_lib).mean() for g in grid]) * 100
        ax.plot(grid * 100, frac, lw=2.0, color=color, zorder=3,
                label=f"{name} (n={len(r)})")
        # 표식을 몇 개만 얹어 색 외의 구분 수단을 둔다
        pick = np.linspace(0, len(grid) - 1, 7).astype(int)
        ax.plot(grid[pick] * 100, frac[pick], marker, ms=5, lw=0, color=color,
                markeredgecolor=S.SURFACE, markeredgewidth=1.0, zorder=4)

    ax.plot(grid * 100, grid * 100, lw=1.2, ls=":", color=S.MUTED, zorder=2,
            label="무작위 (기준선)")
    ax.set_xscale("log")
    plain_log_ticks(ax, "x", unit="%")
    ax.set_xlim(100 / n_lib, 100)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("점수 상위 몇 % 까지 봤는가 (로그 축)")
    ax.set_ylabel("회수한 분자 비율 (%)")
    ax.set_title("(b) 학습에 쓰지 않은 분자의 회수 곡선", loc="left", pad=10)
    ax.legend(loc="upper left")

    ef = None
    for k in ("ef_top0.01",):
        ef = stats08["enrichment"]["test"].get(k)
    if ef:
        ax.annotate(f"상위 1% 에서\ntest 활성 {stats08['enrichment']['test']['active_top0.01']}개 회수"
                    f"\nEF {ef}",
                    xy=(1, stats08["enrichment"]["test"]["active_top0.01"]
                        / max(1, stats08["enrichment"]["test"]["n_active_in_library"]) * 100),
                    xytext=(-14, 34), textcoords="offset points", fontsize=8.4,
                    ha="right", va="bottom", color=S.INK2, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=S.SURFACE,
                              edgecolor=S.MUTED, linewidth=0.7),
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.9))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- 그림 5: 다양성 선별 -----------------------------------------------------

def fig_diversity(S, nn, sel, known_props, stats09, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) 집합 안에서 가장 비슷한 분자와의 유사도 분포
    ax = axes[0]
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    bins = np.linspace(0, 1, 41)
    styles = [("diversity", S.BLUE, "다양성 선별"),
              ("naive_top", S.ORANGE, "점수 상위만")]
    for method, color, name in styles:
        v = nn[nn["method"] == method]["nn_similarity"].to_numpy()
        ax.hist(v, bins=bins, histtype="step", lw=2.0, color=color, zorder=3,
                label=f"{name} (평균 {v.mean():.3f})")
    ax.axvline(stats09["cutoff"], color=S.MUTED, lw=1.1, ls="--", zorder=2)
    ax.annotate(f"cutoff {stats09['cutoff']}",
                xy=(stats09["cutoff"], 0.72), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.2, color=S.INK2, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=S.SURFACE,
                          edgecolor="none"))
    ax.set_xlabel("집합 내 최근접 Tanimoto 유사도")
    ax.set_ylabel("분자 수")
    ax.set_title("(a) 선별 방식에 따른 구조 중복", loc="left", pad=10)
    ax.legend(loc="upper left")

    ax.text(0.98, 0.97,
            f"서로 다른 Murcko 골격\n다양성 선별 {stats09['n_murcko_scaffold']:,}종"
            f"\n점수 상위 {stats09['naive_top_n_scaffold']:,}종",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            color=S.INK2,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=S.SURFACE,
                      edgecolor=S.MUTED, linewidth=0.7))

    # (b) 물성 공간 — 선별 분자가 알려진 억제제 영역을 벗어나지 않는지 본다
    ax = axes[1]
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.plot(known_props["mw"], known_props["logp"], "s", ms=4.5, lw=0,
            color=S.ORANGE, alpha=0.5, zorder=3,
            label=f"알려진 CDK2 활성 (n={len(known_props):,})")
    ax.plot(sel["mw"], sel["logp"], "o", ms=3.2, lw=0, color=S.BLUE,
            alpha=0.6, zorder=4, label=f"도킹 대상 (n={len(sel):,})")
    ax.axvline(450, color=S.MUTED, lw=1.0, ls="--", zorder=2)
    ax.annotate("1단계 필터 상한 MW 450",
                xy=(450, 1), xycoords=("data", "axes fraction"),
                xytext=(-6, -6), textcoords="offset points",
                ha="right", va="top", fontsize=8.2, color=S.INK2, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=S.SURFACE,
                          edgecolor="none"))
    ax.set_xlabel("분자량")
    ax.set_ylabel("logP")
    ax.set_title("(b) 도킹 대상의 물성 분포", loc="left", pad=10)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print("저장:", outpath)


# ---- main --------------------------------------------------------------------

def known_actives_properties(split_csv):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    df = pd.read_csv(split_csv)
    df = df[df["label"] == 1]
    rows = []
    for smi in df["std_smiles"]:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            rows.append((Descriptors.MolWt(m), Descriptors.MolLogP(m)))
    return pd.DataFrame(rows, columns=["mw", "logp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="experiment/08_ml_screen")
    ap.add_argument("--diversity", default="experiment/09_diversity")
    ap.add_argument("--split", default="experiment/04_split/04_scaffold_split.csv")
    ap.add_argument("--outdir", default="experiment/06_figures")
    args = ap.parse_args()

    S = load_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    screen, div = Path(args.screen), Path(args.diversity)
    stats08 = json.loads((screen / "08_screen_stats.json").read_text(encoding="utf-8"))
    stats09 = json.loads((div / "09_selection_stats.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(screen / "08_score_hist.csv")
    known = pd.read_csv(screen / "08_known_ranks.csv")
    nn = pd.read_csv(div / "09_nn_similarity.csv")
    sel = pd.read_csv(div / "09_selected_for_docking.csv")

    fig_ml_screen(S, hist, known, stats08, outdir / "fig4_ml_screen.png")
    fig_diversity(S, nn, sel, known_actives_properties(args.split), stats09,
                  outdir / "fig5_diversity.png")
    p10 = Path("experiment/10_ligands/10_ligand_prep_stats.json")
    stats10 = json.loads(p10.read_text(encoding="utf-8")) if p10.exists() else None
    fig_funnel(S, stats08, stats09, stats10, outdir / "fig6_pipeline_funnel.png")


if __name__ == "__main__":
    main()
