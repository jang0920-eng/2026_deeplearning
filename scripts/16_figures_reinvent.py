"""보고서용 그림 — 6단계 REINVENT 생성 (보고서 §9 탐색 범위).

  fig11_reinvent_space.png
      (a) 화학공간 t-SNE — 생성 분자 / 3단계 선별 1,000개 / CDK2 활성 분자
      (b) 2단계 ML 점수 분포 — 생성 분자 vs 3단계 선별 1,000개
      (c) 최근접 Tanimoto 분포 — 생성 분자가 기존 집합에서 얼마나 떨어져 있는가

색·서체는 `06_figures.py` 의 것을 그대로 불러 쓴다. 새 색을 만들지 않는다.

**생성 깔때기는 그림으로 만들지 않는다.** `skill/06_reinvent.md` §5 와 README 의
표로 이미 전달된다. `CLAUDE.md` 의 "표로 전달되는 것을 그림으로 중복해서 만들지
않는다" 를 따른다.

제목은 중립으로 쓴다. 결론은 그림의 값이 말하게 둔다
(11/14 에 fig8 제목에서 결과를 단정했다가 고친 것과 같은 이유다).

## 두 단계로 나눈 이유

지문·t-SNE 계산은 rdkit 2026.03.3 이 있는 `my-rdkit-env` 에서,
그림은 matplotlib 이 있는 `aizynth-env` 에서 돌린다
(`my-rdkit-env` 에서 matplotlib 은 죽고, `aizynth-env` 의 rdkit 은 2023.09.6 으로
분석에 쓴 것과 버전이 다르다). 좌표를 CSV 로 넘긴다.

실행:
    # 1) 계산 (my-rdkit-env)
    python scripts/16_figures_reinvent.py --stage prep
    # 2) 그림 (aizynth-env)
    python scripts/16_figures_reinvent.py --stage plot
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 3단계 선별 1,000개의 최저 ML 점수 (실측). 이 중 996개가 도킹 입력이다
# (3D 구조 생성에서 4개 실패). 점수 분포는 1,000개 전부로 계산한다.
DOCK_MIN_SCORE = 0.76
TSNE_SEED = 42
N_GEN_SAMPLE = 3000         # t-SNE 가 감당할 크기로 줄인다. 뽑는 수를 그림에 적는다

SET_GEN, SET_DOCK, SET_ACT = "생성 분자", "3단계 선별 1,000개", "CDK2 활성 분자"


# ================================================================ prep


def _fp_matrix(smiles, desc):
    """Morgan fingerprint (radius 2, 2048 bit) 행렬. 2·3단계와 같은 설정."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            rows.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
            keep.append(i)
        except Exception:
            continue
    n_fail = len(smiles) - len(rows)
    print("   [%s] %d개 → 지문 %d개 (실패 %d)" % (desc, len(smiles), len(rows), n_fail))
    return np.asarray(rows, dtype=np.uint8), keep


def stage_prep(args):
    from sklearn.manifold import TSNE

    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    # ---- 입력
    scores = pd.read_csv(ROOT / "experiment/15_reinvent/ml/08_library_scores.csv")
    filt = pd.read_csv(ROOT / "experiment/15_reinvent/filter/07_filtered.csv")
    gen = scores.merge(filt[["chembl_id", "smiles"]], on="chembl_id", how="inner")
    print("생성 분자 (필터+ML 통과): %d개" % len(gen))

    dock = pd.read_csv(ROOT / "experiment/09_diversity/09_selected_for_docking.csv")
    act = pd.read_csv(ROOT / "experiment/01_chembl_cdk2_activity.csv")
    act = act[act["label"] == 1]
    print("3단계 선별: %d개 (이 중 996개가 도킹 입력) / CDK2 활성 분자: %d개"
          % (len(dock), len(act)))

    # ---- 점수 분포는 전수를 그대로 쓴다 (표본 추출하지 않는다)
    pd.concat([
        pd.DataFrame({"set": SET_GEN, "score": gen["score"]}),
        pd.DataFrame({"set": SET_DOCK, "score": dock["score"]}),
    ]).to_csv(out / "16_scores.csv", index=False, encoding="utf-8")

    # ---- t-SNE 는 크기를 줄인다
    rng = np.random.default_rng(TSNE_SEED)
    if len(gen) > N_GEN_SAMPLE:
        idx = rng.choice(len(gen), N_GEN_SAMPLE, replace=False)
        gen_s = gen.iloc[np.sort(idx)]
    else:
        gen_s = gen
    print("t-SNE 입력: 생성 %d (전체 %d 에서 seed %d 로 추출) + 도킹 %d + 활성 %d"
          % (len(gen_s), len(gen), TSNE_SEED, len(dock), len(act)))

    frames = []
    for label, smi in ((SET_GEN, gen_s["smiles"]), (SET_DOCK, dock["smiles"]),
                       (SET_ACT, act["canonical_smiles"])):
        X, _ = _fp_matrix(list(smi), label)
        frames.append((label, X))

    X = np.vstack([x for _, x in frames])
    sets = np.concatenate([[lab] * len(x) for lab, x in frames])
    print("t-SNE 계산 중 (jaccard, perplexity 30, seed %d, n=%d) ..." % (TSNE_SEED, len(X)))
    emb = TSNE(n_components=2, metric="jaccard", init="random",
               perplexity=30, random_state=TSNE_SEED, max_iter=1000).fit_transform(X)

    pd.DataFrame({"set": sets, "x": emb[:, 0], "y": emb[:, 1]}).to_csv(
        out / "16_tsne.csv", index=False, encoding="utf-8")
    print("저장:", out / "16_tsne.csv", "/", out / "16_scores.csv")


# ================================================================ plot


def load_style():
    """06_figures.py 의 스타일·색을 재사용한다 (이름이 숫자로 시작해 import 불가)."""
    spec = importlib.util.spec_from_file_location("fig06", HERE / "06_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_style()
    return mod


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = load_style()
    src = ROOT / args.outdir
    tsne_p, score_p = src / "16_tsne.csv", src / "16_scores.csv"
    sim_p = ROOT / "experiment/15_reinvent/15_novelty_similarity.csv"

    for p in (tsne_p, score_p, sim_p):
        if not p.exists():
            print("입력이 없어 그림을 건너뛴다:", p)
            return

    tsne = pd.read_csv(tsne_p)
    scores = pd.read_csv(score_p)
    sim = pd.read_csv(sim_p)

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # ---- (a) 화학공간 t-SNE
    a = ax[0]
    style = {
        SET_GEN:  dict(c=S.MUTED,  s=5,  alpha=0.45, zorder=1),
        SET_DOCK: dict(c=S.BLUE,   s=9,  alpha=0.75, zorder=2),
        SET_ACT:  dict(c=S.ORANGE, s=9,  alpha=0.75, zorder=3),
    }
    for label in (SET_GEN, SET_DOCK, SET_ACT):
        d = tsne[tsne["set"] == label]
        a.scatter(d["x"], d["y"], label="%s (%s)" % (label, format(len(d), ",")),
                  linewidths=0, **style[label])
    a.set_title("(a) 화학공간 t-SNE")
    a.set_xlabel("t-SNE 1 (Morgan 2048 bit, jaccard)")
    a.set_ylabel("t-SNE 2")
    a.set_xticks([])
    a.set_yticks([])
    a.legend(loc="upper left", markerscale=1.8)

    # ---- (b) ML 점수 분포
    b = ax[1]
    bins = np.linspace(0, 1, 41)
    for label, color in ((SET_GEN, S.PURPLE), (SET_DOCK, S.BLUE)):
        v = scores.loc[scores["set"] == label, "score"].to_numpy(float)
        b.hist(v, bins=bins, density=True, histtype="stepfilled",
               color=color, alpha=0.35, linewidth=0)
        b.hist(v, bins=bins, density=True, histtype="step", color=color, linewidth=1.6,
               label="%s (n=%s, 중앙값 %.3f)" % (label, format(len(v), ","), np.median(v)))
    b.axvline(DOCK_MIN_SCORE, color=S.INK2, linestyle="--", linewidth=1.1)
    # 범례와 겹치지 않도록 축 중간 높이에 둔다
    b.annotate("3단계 선별의\n최저 점수 %.2f" % DOCK_MIN_SCORE,
               xy=(DOCK_MIN_SCORE, b.get_ylim()[1] * 0.55),
               xytext=(-8, 0), textcoords="offset points",
               ha="right", va="center", fontsize=8, color=S.INK2)
    b.set_title("(b) 2단계 ML 활성 예측 점수 분포")
    b.set_xlabel("Random Forest 활성 확률 (같은 모델·같은 split)")
    b.set_ylabel("밀도")
    b.grid(axis="y")
    b.legend(loc="upper right")

    # ---- (c) 최근접 Tanimoto
    c = ax[2]
    bins = np.linspace(0, 1, 41)
    pairs = (("max_tanimoto_docking996", SET_DOCK, S.BLUE),
             ("max_tanimoto_cdk2_actives", SET_ACT, S.ORANGE))
    for col, label, color in pairs:
        v = sim[col].to_numpy(float)
        v = v[~np.isnan(v)]
        c.hist(v, bins=bins, histtype="step", color=color, linewidth=1.6,
               label="vs %s\n   중앙값 %.3f · 최대 %.3f · ≥0.7 은 %d개"
                     % (label, np.median(v), v.max(), int((v >= 0.7).sum())))
    c.axvline(0.7, color=S.INK2, linestyle="--", linewidth=1.1)
    # 범례가 오른쪽 위를 쓰므로 선 설명은 축 중간 높이에 둔다
    c.annotate("Tanimoto 0.7", xy=(0.7, c.get_ylim()[1] * 0.45),
               xytext=(6, 0), textcoords="offset points", ha="left", va="center",
               fontsize=8, color=S.INK2)
    c.set_title("(c) 생성 분자의 최근접 Tanimoto")
    c.set_xlabel("최대 Tanimoto (Morgan 2048 bit)")
    c.set_ylabel("생성 분자 수")
    c.grid(axis="y")
    c.legend(loc="upper right")

    fig.suptitle("6단계 REINVENT prior sampling — 생성 분자와 기존 후보 집합 비교",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()

    outdir = ROOT / args.figdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "fig11_reinvent_space.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("저장:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="prep", choices=["prep", "plot"])
    ap.add_argument("--outdir", default="experiment/15_reinvent")
    ap.add_argument("--figdir", default="experiment/06_figures")
    args = ap.parse_args()
    (stage_prep if args.stage == "prep" else stage_plot)(args)


if __name__ == "__main__":
    main()
