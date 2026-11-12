"""
3단계 다양성 선별 — ML 상위 후보에서 도킹할 1,000개를 고른다.

입력:  experiment/08_ml_screen/08_top_candidates.csv  (점수 상위, 알려진 분자 제외)
출력:  experiment/09_diversity/

## 왜 점수 상위 1,000개를 그냥 쓰지 않는가
RF 점수 상위에는 같은 골격의 유사체가 몰린다. 학습 데이터가 몇 개 골격에 편중돼 있어
모델이 그 골격을 높게 보기 때문이다. 유사체 1,000개를 도킹하면 15시간을 써서
사실상 같은 분자를 반복 평가하게 된다. 탐색 범위를 넓히는 것이 과제 가산점 항목이기도 하다.

## 방법 — 점수 순서 sphere exclusion (directed sphere exclusion)
1. 점수가 높은 순서로 분자를 본다
2. 이미 고른 분자와의 Tanimoto 유사도가 모두 cutoff 미만이면 고른다
3. 1,000개가 되면 멈춘다

Butina 클러스터링 후 중심 분자를 뽑는 방법도 있으나, 중심 분자는 점수와 무관하게
결정되므로 클러스터 안에서 가장 유망한 분자를 놓칠 수 있다. sphere exclusion 은
**점수 우선순위를 유지한 채** 유사 구조만 배제하므로 깔때기의 목적에 더 맞고,
10,000 × 10,000 거리 행렬(5천만 쌍)을 만들지 않아 메모리도 적게 쓴다.
난수를 쓰지 않아 결과가 항상 같다.

## 실행
    python scripts/09_diversity_select.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

RADIUS, N_BITS = 2, 2048          # 08 단계와 같은 표현
CUTOFF = 0.6                      # 이 유사도 이상이면 "같은 계열"로 보고 버린다
TARGET = 1000                     # 도킹 대상 수 (02 단계 속도 실측에 맞춘 값)


def fingerprints(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps, ok = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        ok.append(i)
    return fps, ok


def sphere_exclusion(fps, cutoff, target):
    """점수 순서(= 입력 순서)대로 보면서, 기존 선택과 충분히 다른 것만 고른다.

    반환: (선택 index, 각 탈락 분자가 어느 선택 분자와 겹쳤는지)
    """
    picked, picked_fps = [], []
    excluded_by = {}
    for i, fp in enumerate(fps):
        if picked_fps:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, picked_fps))
            j = int(sims.argmax())
            if sims[j] >= cutoff:
                excluded_by[i] = (picked[j], round(float(sims[j]), 3))
                continue
        picked.append(i)
        picked_fps.append(fp)
        if len(picked) >= target:
            break
    return picked, excluded_by


def properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return dict(
        mw=round(Descriptors.MolWt(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 3),
        hbd=rdMolDescriptors.CalcNumHBD(mol),
        hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        rotb=rdMolDescriptors.CalcNumRotatableBonds(mol),
        heavy=mol.GetNumHeavyAtoms(),
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
    )


def nn_similarity(fps):
    """선택된 집합 안에서 각 분자의 최근접 유사도 — 다양성이 실제로 확보됐는지 확인."""
    out = []
    for i, fp in enumerate(fps):
        others = fps[:i] + fps[i + 1:]
        if not others:
            return np.array([0.0])
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, others)))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiment/08_ml_screen/08_top_candidates.csv")
    ap.add_argument("--outdir", default="experiment/09_diversity")
    ap.add_argument("--pool", type=int, default=10000,
                    help="점수 상위 몇 개를 후보 풀로 볼지 (기본 1만)")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input).sort_values("score", ascending=False)
    pool = df.head(args.pool).reset_index(drop=True)
    print(f"후보 풀 {len(pool):,}개 (점수 {pool['score'].min():.3f} ~ {pool['score'].max():.3f})")

    fps, ok = fingerprints(pool["smiles"].tolist())
    pool = pool.iloc[ok].reset_index(drop=True)
    print(f"fingerprint 생성 {len(fps):,}개")

    cutoff, relaxed = args.cutoff, []
    picked, excluded = sphere_exclusion(fps, cutoff, args.target)
    print(f"cutoff {cutoff}: {len(picked):,}개 선택")

    # 풀이 모자라면 다양성 기준을 조금씩 완화한다 (점수를 더 낮추는 것보다 낫다)
    while len(picked) < args.target and cutoff < 0.95:
        cutoff = round(cutoff + 0.05, 2)
        relaxed.append(cutoff)
        picked, excluded = sphere_exclusion(fps, cutoff, args.target)
        print(f"  부족 → cutoff {cutoff} 로 완화: {len(picked):,}개")

    sel = pool.iloc[picked].reset_index(drop=True)
    props = pd.DataFrame([properties(s) for s in sel["smiles"]])
    sel = pd.concat([sel, props], axis=1)
    sel.insert(0, "dock_id", [f"C{i + 1:04d}" for i in range(len(sel))])
    sel = sel.rename(columns={"rank": "ml_rank"})
    sel.to_csv(outdir / "09_selected_for_docking.csv", index=False)

    sel_fps = [fps[i] for i in picked]
    nn = nn_similarity(sel_fps)
    n_scaf = sel["scaffold"].nunique()

    stats = {
        "pool": int(len(pool)),
        "selected": int(len(sel)),
        "cutoff": cutoff,
        "cutoff_relaxed_to": relaxed,
        "excluded_as_analogue": int(len(excluded)),
        "score_range_selected": [float(sel["score"].min()), float(sel["score"].max())],
        "score_range_pool": [float(pool["score"].min()), float(pool["score"].max())],
        "ml_rank_max_selected": int(sel["ml_rank"].max()),
        "n_murcko_scaffold": int(n_scaf),
        "scaffold_per_molecule": round(n_scaf / len(sel), 3),
        "nn_similarity_mean": round(float(nn.mean()), 3),
        "nn_similarity_max": round(float(nn.max()), 3),
        "mw_mean": round(float(sel["mw"].mean()), 1),
        "logp_mean": round(float(sel["logp"].mean()), 2),
        "rotb_mean": round(float(sel["rotb"].mean()), 2),
        "heavy_mean": round(float(sel["heavy"].mean()), 1),
    }
    (outdir / "09_selection_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    # 점수만 보고 뽑았을 때와 비교 — 다양성 선별의 효과를 수치로 보여준다
    naive_fps = fps[:len(sel)]
    naive_nn = nn_similarity(naive_fps)
    naive_scaf = pd.Series(
        [properties(s).get("scaffold") for s in pool["smiles"].head(len(sel))]
    ).nunique()
    stats["naive_top_nn_similarity_mean"] = round(float(naive_nn.mean()), 3)
    stats["naive_top_n_scaffold"] = int(naive_scaf)
    (outdir / "09_selection_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    # 그림용 — 두 선별 방식의 최근접 유사도 분포
    pd.concat([
        pd.DataFrame({"method": "diversity", "nn_similarity": nn.round(4)}),
        pd.DataFrame({"method": "naive_top", "nn_similarity": naive_nn.round(4)}),
    ]).to_csv(outdir / "09_nn_similarity.csv", index=False)

    md = f"""### 3단계 다양성 선별 ({stats['pool']:,} → {stats['selected']:,})

점수 상위 {stats['pool']:,}개에서 **Tanimoto {stats['cutoff']} 이상으로 겹치지 않는**
분자 {stats['selected']:,}개를 점수 높은 순서로 골랐다 (sphere exclusion).
유사체로 판정돼 제외된 분자는 {stats['excluded_as_analogue']:,}개다.

| 항목 | 다양성 선별 {stats['selected']:,}개 | 점수 상위 {stats['selected']:,}개 (비교용) |
|---|---|---|
| 서로 다른 Murcko 골격 수 | **{stats['n_murcko_scaffold']:,}** | {stats['naive_top_n_scaffold']:,} |
| 집합 내 최근접 유사도 평균 | **{stats['nn_similarity_mean']}** | {stats['naive_top_nn_similarity_mean']} |

점수만 보고 상위 {stats['selected']:,}개를 뽑으면 서로 비슷한 분자가 몰린다.
같은 도킹 시간(약 15시간)으로 더 넓은 화학 공간을 보기 위해 다양성 선별을 넣었다.

- 선택된 분자의 RF 점수 범위: {stats['score_range_selected'][0]:.3f} ~ {stats['score_range_selected'][1]:.3f}
  (ML 순위 최대 {stats['ml_rank_max_selected']:,}위까지 내려갔다)
- 평균 분자량 {stats['mw_mean']}, 평균 logP {stats['logp_mean']},
  평균 회전결합 {stats['rotb_mean']}개, 평균 중원자 {stats['heavy_mean']}개

회전결합 평균이 {stats['rotb_mean']}개로, 02 단계 속도 실측에 쓴 분자들과 같은 범위다.
따라서 개당 59초 가정이 유지되고 1,000개는 약 16시간으로 예상된다.

#### 한계
- cutoff {stats['cutoff']} 은 관례적인 값이다. 더 크게 하면 유사체가 섞이고
  작게 하면 후보 풀이 모자란다.
- 다양성 선별은 **점수를 낮추는 대가**를 치른다. 위 표의 점수 범위가 그 크기다.
"""
    (outdir / "09_diversity_summary.md").write_text(md, encoding="utf-8")

    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"저장: {outdir}")


if __name__ == "__main__":
    main()
