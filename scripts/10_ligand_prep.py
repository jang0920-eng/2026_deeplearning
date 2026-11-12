"""
도킹용 리간드 3D 구조 생성 — 시드를 고정한다.

입력:  experiment/09_diversity/09_selected_for_docking.csv
출력:  experiment/10_ligands/

## 왜 이 스크립트가 따로 있는가
02 단계 속도 측정에서 Vina 점수가 **+1.10 kcal/mol** 로 나온 분자가 있었다.
처음에는 도킹 박스가 작아서라고 추정했으나, 박스를 16/20/24 Å 으로 바꿔 재도킹해 보니
차이가 0.13 kcal/mol 뿐이어서 그 가설은 기각됐다.
실제 원인은 **3D 구조 생성**이었다.
  - ETKDG 난수 시드를 고정하지 않았다 → 실행마다 다른 구조가 나온다
  - MMFF 최적화가 기본 200회였고 수렴 확인을 하지 않았다 → 뒤틀린 구조가 남는다
뒤틀린 구조는 단백질과 충돌하므로 점수가 양수가 된다. 분자 크기 문제가 아니었다.

그래서 이 단계를 독립된 스크립트로 분리하고 다음을 고정한다.
  1. `ETKDGv3`, `randomSeed = 42`  → 같은 입력이면 항상 같은 3D 구조
  2. MMFF94s 최적화, `maxIters=2000`, **수렴 코드 확인**
     한 번에 수렴하지 않으면 한 번 더 이어서 돌리고, 그래도 수렴하지 않으면 버린다
  3. 실패한 분자는 조용히 빠지지 않고 `10_ligand_prep_log.csv` 에 이유를 남긴다

## 실행
    python scripts/10_ligand_prep.py --batches 4
도킹을 나흘에 나눠 하므로 batch SDF 를 따로 쓴다. 같은 명령은 항상 같은 결과를 준다.
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

SEED = 42
MMFF_ITERS = 2000
MMFF_VARIANT = "MMFF94s"


def build_3d(smiles):
    """SMILES → 수소 붙은 3D 분자. 실패하면 (None, 이유)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "parse_fail", None

    mh = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = SEED          # 재현성 — 이 줄이 없으면 실행마다 구조가 달라진다
    if AllChem.EmbedMolecule(mh, params) != 0:
        return None, "embed_fail", None

    # MMFF 수렴 확인: 0 = 수렴, 1 = 반복 횟수 안에 수렴 못 함, -1 = 힘장 없음
    code = AllChem.MMFFOptimizeMolecule(mh, mmffVariant=MMFF_VARIANT,
                                        maxIters=MMFF_ITERS)
    if code == 1:
        code = AllChem.MMFFOptimizeMolecule(mh, mmffVariant=MMFF_VARIANT,
                                            maxIters=MMFF_ITERS)   # 이어서 한 번 더
    if code != 0:
        return None, ("mmff_no_converge" if code == 1 else "mmff_no_forcefield"), None

    props = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant=MMFF_VARIANT)
    ff = AllChem.MMFFGetMoleculeForceField(mh, props)
    energy = round(ff.CalcEnergy(), 3) if ff is not None else None

    # 03 단계 redocking B(RMSD 0.56 Å)와 **같은 형태로** 내보낸다.
    # 그 검증이 수소를 뗀 SDF 로 통과했으므로 입력 형식을 바꾸지 않는다.
    return Chem.RemoveHs(mh), "ok", energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiment/09_diversity/09_selected_for_docking.csv")
    ap.add_argument("--outdir", default="experiment/10_ligands")
    ap.add_argument("--batches", type=int, default=4,
                    help="도킹을 며칠에 나눌지 (batch SDF 개수)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"입력 {len(df):,}개")

    rows, mols = [], []
    for i, r in enumerate(df.itertuples(index=False), 1):
        mh, status, energy = build_3d(r.smiles)
        rows.append(dict(dock_id=r.dock_id, chembl_id=r.chembl_id,
                         smiles=r.smiles, ml_score=r.score,
                         status=status, mmff_energy=energy))
        if mh is not None:
            mh.SetProp("_Name", r.dock_id)
            mh.SetProp("dock_id", r.dock_id)
            mh.SetProp("chembl_id", r.chembl_id)
            mh.SetProp("ml_score", f"{r.score:.4f}")
            mh.SetProp("smiles", r.smiles)
            mols.append(mh)
        if i % 100 == 0:
            print(f"  {i:,}/{len(df):,}  성공 {len(mols):,}", flush=True)

    log = pd.DataFrame(rows)
    log.to_csv(outdir / "10_ligand_prep_log.csv", index=False)

    # 전체 SDF (gzip) — 도킹 환경으로 올릴 파일
    with gzip.open(outdir / "ligands_all.sdf.gz", "wt") as fh:
        w = Chem.SDWriter(fh)
        for m in mols:
            w.write(m)
        w.close()

    # 하루치씩 나눈 SDF — 세션이 끊겨도 배치 단위로 재개할 수 있다
    n = max(1, args.batches)
    size = -(-len(mols) // n)
    for b in range(n):
        part = mols[b * size:(b + 1) * size]
        if not part:
            continue
        w = Chem.SDWriter(str(outdir / f"ligands_batch{b + 1}.sdf"))
        for m in part:
            w.write(m)
        w.close()
        print(f"  batch{b + 1}: {len(part)}개")

    counts = log["status"].value_counts().to_dict()
    stats = dict(input=int(len(df)), prepared=int(len(mols)),
                 batches=n, batch_size=size, status=counts,
                 seed=SEED, mmff_variant=MMFF_VARIANT, mmff_iters=MMFF_ITERS)
    (outdir / "10_ligand_prep_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    fail_md = "\n".join(
        f"| {k} | {v} |" for k, v in counts.items() if k != "ok") or "| (없음) | 0 |"
    md = f"""### 리간드 3D 구조 생성 (도킹 입력)

- 입력 {stats['input']:,}개 → 생성 성공 **{stats['prepared']:,}개**
- `ETKDGv3`, `randomSeed = {SEED}` — 같은 입력이면 항상 같은 좌표가 나온다
- {MMFF_VARIANT} 최적화, `maxIters={MMFF_ITERS}`, 수렴 코드 확인
  (한 번에 수렴하지 않으면 한 번 더 이어서 돌리고, 그래도 안 되면 제외)
- {n}개 배치로 나눠 저장 (배치당 {size}개) — 하루에 한 배치씩 도킹한다

| 실패 사유 | 분자 수 |
|---|---|
{fail_md}

#### 이 단계를 따로 둔 이유
02 단계에서 Vina 점수가 +1.10 kcal/mol(= 결합 불성립)인 분자가 나왔다.
도킹 박스가 작아서라고 추정했지만 박스를 16/20/24 Å 으로 바꿔 보니 차이가
0.13 kcal/mol 뿐이라 가설이 기각됐다. 실제 원인은 시드를 고정하지 않은 ETKDG 와
수렴을 확인하지 않은 MMFF 최적화, 즉 **3D 구조 생성 품질**이었다.
추정으로 설정을 바꾸지 않고 실측으로 원인을 확인한 사례다.
"""
    (outdir / "10_ligand_prep_summary.md").write_text(md, encoding="utf-8")

    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"저장: {outdir}")


if __name__ == "__main__":
    main()
