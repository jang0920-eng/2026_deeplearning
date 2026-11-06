"""
1단계 물리화학 필터 — ChEMBL 전체 화합물에서 스크리닝 대상을 고른다.

입력:  chembl_37_chemreps.txt.gz (2,897,819개)
출력:  통과 분자 CSV + 탈락 사유별 집계 JSON

필터 순서 (앞에서 걸린 분자는 뒤 검사를 하지 않는다 — 집계는 "처음 걸린 사유" 기준)
  1. SMILES 파싱 실패
  2. 유기 원소가 아닌 것 포함 (금속 착물 등)
  3. 염·혼합물 → 가장 큰 조각만 사용 (탈락이 아니라 변환)
  4. 중원자 수 범위 밖 (10 미만은 조각, 50 초과는 너무 큼)
  5. lead-like: MW > 450 또는 회전결합 > 7
  6. Lipinski: logP > 5, HBD > 5, HBA > 10
  7. PAINS: 위양성을 자주 내는 구조

lead-like 를 넣은 이유는 두 가지다. 도킹 시간이 회전결합 수에 비례해 늘어나고,
신약 개발에서 출발 물질은 최적화 과정에서 분자량이 커지므로 작게 시작하는 편이 유리하다.

여러 번에 나눠 실행할 수 있다 (--start / --end 로 줄 범위 지정, 결과는 이어붙인다).
"""

import argparse
import gzip
import json
import os
from collections import Counter
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
MIN_HEAVY, MAX_HEAVY = 10, 50
MAX_MW, MAX_ROTB = 450.0, 7        # lead-like
MAX_LOGP, MAX_HBD, MAX_HBA = 5.0, 5, 10   # Lipinski

_CAT = None


def _catalog():
    global _CAT
    if _CAT is None:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _CAT = FilterCatalog(p)
    return _CAT


def screen(record):
    """통과하면 (행, None), 탈락하면 (None, 사유)."""
    cid, smi = record

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "parse_fail"

    # 염·혼합물은 가장 큰 조각만 남긴다
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None, "parse_fail"

    if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
        return None, "inorganic"

    heavy = mol.GetNumHeavyAtoms()
    if heavy < MIN_HEAVY or heavy > MAX_HEAVY:
        return None, "size"

    mw = Descriptors.MolWt(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if mw > MAX_MW or rotb > MAX_ROTB:
        return None, "lead_like"

    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    if logp > MAX_LOGP or hbd > MAX_HBD or hba > MAX_HBA:
        return None, "lipinski"

    if _catalog().HasMatch(mol):
        return None, "pains"

    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return (cid, Chem.MolToSmiles(mol), round(mw, 2), round(logp, 3),
            hbd, hba, round(tpsa, 2), rotb, heavy), None


def read_slice(path, start, end):
    with gzip.open(path, "rt") as f:
        f.readline()                      # 헤더
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                yield parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "07_filtered.csv")
    out_json = os.path.join(args.outdir, "07_filter_counts.json")

    counts = Counter()
    if os.path.exists(out_json):
        counts.update(json.load(open(out_json)))

    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a") as fo:
        if new_file:
            fo.write("chembl_id,smiles,mw,logp,hbd,hba,tpsa,rotb,heavy\n")

        with Pool(args.workers) as pool:
            stream = read_slice(args.input, args.start, args.end)
            for row, reason in pool.imap_unordered(screen, stream, chunksize=2000):
                counts["total"] += 1
                if reason:
                    counts[reason] += 1
                else:
                    counts["passed"] += 1
                    fo.write(",".join(str(x) for x in row) + "\n")

    json.dump(dict(counts), open(out_json, "w"), indent=1)
    print(f"구간 {args.start}~{args.end}")
    for k, v in counts.most_common():
        print(f"  {k:12s} {v:>9,}")


if __name__ == "__main__":
    main()
