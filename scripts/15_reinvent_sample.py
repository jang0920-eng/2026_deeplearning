"""6단계 — REINVENT prior 로 SMILES 를 생성하고 유효성·중복을 정리한다.

판정 기준은 `skill/06_reinvent.md` §3 에 **결과를 보기 전에** 적었다.
이 스크립트는 그 기준을 그대로 구현한다.

흐름
  1. reinvent.prior 를 안전하게 로드 (REINVENT4 설치하지 않는다, §2 참고)
  2. autoregressive sampling 으로 N 개 생성
  3. RDKit 유효성 검사 → canonical SMILES
  4. 중복 제거: 생성 내부 → 도킹 대상 996개 → CDK2 활성 데이터 → 142만 라이브러리
  5. Tanimoto 유사도는 **보고만** 한다 (탈락 기준으로 쓰지 않는다)
  6. 1단계 필터(`07_physchem_filter.py`)에 넣을 TSV 를 쓴다

왜 REINVENT4 를 설치하지 않는가 — `skill/06_reinvent.md` §2 에 적었다. 요약하면
(1) chemprop-env(5단계 ADMET 환경)의 torch 2.15.0 을 깨뜨리면 안 되고,
(2) 내려받은 pickle 을 `torch.load(weights_only=False)` 로 열면 임의 코드가
실행될 수 있다. 그래서 제한 Unpickler 로 읽고 텐서는 zip 안의 storage 에서
`torch.frombuffer` 로 직접 만든다. 실행되는 코드는 이 파일 안의 것뿐이다.

실행 (chemprop-env, 추가 설치 없음)

    PYTHONIOENCODING=utf-8 \
    ~/miniconda3/envs/chemprop-env/python.exe scripts/15_reinvent_sample.py \
        --prior "<바탕화면>/딥러닝/reinvent_prior/reinvent.prior" --n 10000

⚠ Windows 에서 `conda run` 은 출력을 cp949 로 재인코딩하다 UnicodeEncodeError 로
죽는다. 환경의 python 을 직접 부르고 PYTHONIOENCODING=utf-8 을 준다.
(`skill/04_admet.md` 의 admet-ai 와 같은 문제다.)
"""

import argparse
import builtins
import collections
import gzip
import importlib.util
import io
import json
import os
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ---------------------------------------------------------------- 체크포인트 로드

_DTYPES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class _StubBase:
    """체크포인트에 들어 있지만 우리가 쓰지 않는 클래스의 자리만 채운다."""

    def __setstate__(self, state):
        self._state = state


_stub_cache = {}


def _stub_class(module, name):
    key = (module, name)
    if key not in _stub_cache:
        _stub_cache[key] = type(name, (_StubBase,), {"_qn": "%s.%s" % (module, name)})
    return _stub_cache[key]


class SafePriorUnpickler(pickle.Unpickler):
    """허용 목록에 있는 것만 실제로 호출한다. 나머지는 데이터 보관용 Stub.

    `torch._utils._rebuild_tensor_v2` 도 torch 것을 쓰지 않고 우리 구현으로
    바꿔서, storage 를 zip 에서 직접 읽어 텐서를 만든다.
    """

    BUILTIN_OK = {
        "dict", "list", "tuple", "set", "frozenset",
        "int", "float", "str", "bool", "bytes", "complex", "object",
    }

    def __init__(self, fileobj, zf):
        super().__init__(fileobj)
        self._zf = zf

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self._rebuild_tensor
        if module == "builtins" and name in self.BUILTIN_OK:
            return getattr(builtins, name)
        return _stub_class(module, name)

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        if pid[0] != "storage":
            raise pickle.UnpicklingError("예상하지 못한 persistent id: %r" % (pid,))
        storage_cls, key = pid[1], pid[2]
        cls_name = getattr(storage_cls, "__name__", str(storage_cls)).split(".")[-1]
        if cls_name not in _DTYPES:
            raise pickle.UnpicklingError("모르는 storage 종류: %s" % cls_name)
        return {"raw": self._zf.read("reinvent/data/%s" % key), "dtype": _DTYPES[cls_name]}

    @staticmethod
    def _rebuild_tensor(storage, storage_offset, size, stride,
                        requires_grad=False, backward_hooks=None, metadata=None):
        flat = torch.frombuffer(bytearray(storage["raw"]), dtype=storage["dtype"])
        if not size:
            return flat[storage_offset].clone()
        return torch.as_strided(flat, tuple(size), tuple(stride), storage_offset).clone()


def load_prior(path):
    """체크포인트를 dict 로 돌려준다. 임의 코드를 실행하지 않는다."""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("reinvent/data.pkl")
        return SafePriorUnpickler(io.BytesIO(raw), zf).load()


# ---------------------------------------------------------------- 모델


class ReinventPrior(nn.Module):
    """REINVENT4 `Reinvent` 모델의 추론 경로만 다시 구현한 것.

    파라미터 이름을 체크포인트와 똑같이 맞췄다. `load_state_dict(strict=True)`
    가 통과하면 구조가 원본과 같다는 뜻이다.
    """

    def __init__(self, voc_size, embedding_layer_size=256, layer_size=512,
                 num_layers=3, dropout=0.0, cell_type="lstm"):
        super().__init__()
        if cell_type != "lstm":
            raise ValueError("lstm 만 구현했다: %r" % cell_type)
        self._embedding = nn.Embedding(voc_size, embedding_layer_size)
        self._rnn = nn.LSTM(embedding_layer_size, layer_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self._linear = nn.Linear(layer_size, voc_size)

    def forward(self, x, hidden=None):
        out, hidden = self._rnn(self._embedding(x), hidden)
        return self._linear(out), hidden


@torch.no_grad()
def sample_batch(model, bos, eos, n, max_len, temperature, generator):
    """autoregressive sampling. 반환: (token id 리스트들, 끝까지 안 끝난 개수)"""
    x = torch.full((n, 1), bos, dtype=torch.long)
    hidden = None
    done = torch.zeros(n, dtype=torch.bool)
    seqs = [[] for _ in range(n)]
    for _ in range(max_len):
        logits, hidden = model(x, hidden)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        live = (~done) & (nxt != eos)
        for i in torch.nonzero(live, as_tuple=False).flatten().tolist():
            seqs[i].append(int(nxt[i]))
        done |= nxt == eos
        if bool(done.all()):
            break
        x = nxt.unsqueeze(1)
    return seqs, int((~done).sum())


# ---------------------------------------------------------------- RDKit 쪽
# 표준화는 2단계 `08_ml_screen.py` 의 함수를 그대로 쓴다. 기준이 갈라지면 안 된다.

_ML = None


def _load_ml_module():
    """08_ml_screen.py 를 모듈로 불러온다 (이름이 숫자로 시작해 import 가 안 된다)."""
    global _ML
    if _ML is None:
        spec = importlib.util.spec_from_file_location(
            "ml_screen", HERE / "08_ml_screen.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._init_worker()           # _GEN(Morgan), _UNCHARGER 생성
        _ML = mod
    return _ML


def _init_rdkit_worker():
    _load_ml_module()


def canonical_plain(smi):
    """표준화 없이 RDKit canonical SMILES 로만 바꾼다.

    집합 사이 정확 일치 비교에 쓴다. 라이브러리·996개·활성 데이터도 같은 함수를
    통과시켜 **같은 RDKit 버전의 canonical 규칙**으로 맞춘다.
    (07 단계는 다른 환경·다른 RDKit 에서 돌았으므로 그 출력을 그대로 믿지 않는다.)
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


def canonical_std(smi):
    """2단계와 같은 표준화(Cleanup → FragmentParent → 중성화) 후 canonical."""
    from rdkit import Chem

    ml = _load_ml_module()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = ml.standardize_mol(mol)
    return None if mol is None else Chem.MolToSmiles(mol)


def _canon_pair(smi):
    """(plain, std) 를 한 번에. 워커에서 쓴다."""
    return canonical_plain(smi), canonical_std(smi)


def fingerprint(smi):
    """Morgan 지문. 실패하면 None.

    ⚠ 표준화(Cleanup → FragmentParent → 중성화) 뒤에도 ring 정보가 초기화되지
    않은 분자가 있어 `GetFingerprintAsNumPy` 가 `RingInfo not initialized` 로
    죽는다 (CDK2 활성 데이터에서 실제로 발생했다). 그래서 지문 계산 전에
    `SanitizeMol` 을 한 번 더 부른다. 그래도 실패하는 것은 None 으로 돌리고
    **호출한 쪽에서 개수를 세어 보고한다** (조용히 버리지 않는다).
    """
    from rdkit import Chem

    ml = _load_ml_module()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = ml.standardize_mol(mol)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return ml._GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------- 기준 집합


def read_csv_column(path, col, sep=","):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(sep)
        idx = header.index(col)
        for line in f:
            parts = line.rstrip("\n").split(sep)
            if len(parts) > idx and parts[idx]:
                yield parts[idx]


def canonical_set(smiles_iter, workers, desc, cache=None):
    """SMILES 묶음을 plain canonical 집합으로 만든다. cache 가 있으면 재사용한다."""
    if cache is not None and Path(cache).exists():
        print("  [%s] 캐시 사용: %s" % (desc, cache))
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return {l.rstrip("\n") for l in f if l.strip()}

    t0 = time.time()
    out = set()
    n_in = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_rdkit_worker) as ex:
        for i, c in enumerate(ex.map(canonical_plain, smiles_iter, chunksize=2000)):
            n_in += 1
            if c is None:
                n_fail += 1
            else:
                out.add(c)
            if (i + 1) % 200000 == 0:
                print("  [%s] %d개 처리 (%.0f초)" % (desc, i + 1, time.time() - t0))
    print("  [%s] 입력 %d, canonical 고유 %d, parse 실패 %d (%.0f초)"
          % (desc, n_in, len(out), n_fail, time.time() - t0))

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            for s in out:
                f.write(s + "\n")
        print("  [%s] 캐시 저장: %s" % (desc, cache))
    return out


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="reinvent.prior 경로")
    ap.add_argument("--n", type=int, default=10000, help="생성 개수 (기준: 10,000)")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=500, help="sampling batch 크기")
    ap.add_argument("--outdir", default="experiment/15_reinvent")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--docking-set",
                    default="experiment/09_diversity/09_selected_for_docking.csv")
    ap.add_argument("--activity", default="experiment/01_chembl_cdk2_activity.csv")
    ap.add_argument("--library", default=None,
                    help="142만 라이브러리 csv.gz. 생략하면 그 비교를 건너뛴다")
    ap.add_argument("--library-cache", default=None,
                    help="라이브러리 canonical 집합 캐시 (.txt.gz). 대용량이라 저장소 밖에 둔다")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir) if not os.path.isabs(args.outdir) else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. prior 로드
    print("=" * 70)
    print("1. prior 로드:", args.prior)
    ck = load_prior(args.prior)
    net_params = dict(ck["network_params"])
    voc_state = ck["vocabulary"]._state
    id2tok = {k: v for k, v in voc_state["_tokens"].items() if isinstance(k, int)}
    tok2id = {v: k for k, v in id2tok.items()}
    max_len = int(ck["max_sequence_length"])
    origin = ck["metadata"]._state.get("origina_data_source")

    model = ReinventPrior(len(id2tok), **net_params)
    load_msg = model.load_state_dict(ck["network"], strict=True)
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    print("   network_params:", net_params)
    print("   vocab %d / max_len %d / 사전학습 %s" % (len(id2tok), max_len, origin))
    print("   load_state_dict(strict=True):", load_msg)
    print("   파라미터 수:", n_param)

    bos, eos = tok2id["^"], tok2id["$"]

    # ---- 2. 생성
    print("=" * 70)
    print("2. sampling %d개 (temperature %.2f, seed %d)"
          % (args.n, args.temperature, args.seed))
    gen = torch.Generator().manual_seed(args.seed)
    raw = []
    n_truncated = 0
    t0 = time.time()
    while len(raw) < args.n:
        k = min(args.batch, args.n - len(raw))
        seqs, n_cut = sample_batch(model, bos, eos, k, max_len, args.temperature, gen)
        n_truncated += n_cut
        raw.extend("".join(id2tok[i] for i in s) for s in seqs)
        print("   %d / %d (%.0f초)" % (len(raw), args.n, time.time() - t0))
    t_sample = time.time() - t0
    print("   완료 %.1f초 (%.1f개/초), 길이 초과로 잘린 것 %d개"
          % (t_sample, args.n / t_sample, n_truncated))

    # ---- 3. 유효성 + canonical
    print("=" * 70)
    print("3. RDKit 유효성 검사")
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_rdkit_worker) as ex:
        pairs = list(ex.map(_canon_pair, raw, chunksize=200))

    valid_idx = [i for i, (p, _) in enumerate(pairs) if p is not None]
    n_valid = len(valid_idx)
    validity = n_valid / args.n
    # 표준화까지 통과한 것만 뒤로 보낸다 (2단계와 같은 전처리를 거쳐야 하므로)
    std_idx = [i for i in valid_idx if pairs[i][1] is not None]
    print("   유효 %d / %d = %.2f%%" % (n_valid, args.n, 100 * validity))
    print("   표준화까지 통과 %d개" % len(std_idx))

    if validity < 0.85:
        print()
        print("!! 유효율이 기준(85%)에 미달한다. skill/06_reinvent.md §3 에 따라")
        print("!! 기준을 낮추지 않고 구현 오류로 간주한다. 결과를 뒤로 넘기지 않는다.")
        _write_raw(outdir, raw, pairs)
        json.dump({"n_requested": args.n, "n_valid": n_valid, "validity": validity,
                   "aborted": "validity_below_0.85"},
                  open(outdir / "15_stats.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return 1

    _write_raw(outdir, raw, pairs)

    # ---- 4. 중복 제거
    print("=" * 70)
    print("4. 중복 제거 (canonical SMILES 정확 일치)")

    # 4a. 생성 내부
    seen, uniq = set(), []
    for i in std_idx:
        c = pairs[i][0]
        if c not in seen:
            seen.add(c)
            uniq.append(i)
    print("   생성 내부 중복 제거: %d → %d (고유율 %.2f%%)"
          % (len(std_idx), len(uniq), 100 * len(uniq) / max(1, len(std_idx))))

    # 4b. 기준 집합들
    dock_set = canonical_set(
        read_csv_column(ROOT / args.docking_set, "smiles"),
        args.workers, "도킹 996개")
    act_set = canonical_set(
        read_csv_column(ROOT / args.activity, "canonical_smiles"),
        args.workers, "CDK2 활성 데이터")

    lib_set = set()
    if args.library:
        lib_set = canonical_set(
            read_csv_column(args.library, "smiles"),
            args.workers, "142만 라이브러리", cache=args.library_cache)
    else:
        print("   [142만 라이브러리] --library 를 주지 않아 건너뛴다")

    novel, hit_dock, hit_act, hit_lib = [], 0, 0, 0
    for i in uniq:
        c = pairs[i][0]
        in_d, in_a, in_l = c in dock_set, c in act_set, c in lib_set
        hit_dock += in_d
        hit_act += in_a
        hit_lib += in_l
        if not (in_d or in_a or in_l):
            novel.append(i)
    print("   도킹 996개와 일치: %d" % hit_dock)
    print("   CDK2 활성 데이터와 일치: %d" % hit_act)
    print("   142만 라이브러리와 일치: %d" % hit_lib)
    print("   → 신규 %d개 (고유 %d개 중 %.2f%%)"
          % (len(novel), len(uniq), 100 * len(novel) / max(1, len(uniq))))

    # ---- 5. 유사도 (보고용. 탈락 기준으로 쓰지 않는다)
    print("=" * 70)
    print("5. Tanimoto 최근접 유사도 — 보고용 기술 통계")
    sim_dock = _max_tanimoto([pairs[i][1] for i in novel],
                             sorted(dock_set), args.workers, "도킹 996개")
    sim_act = _max_tanimoto([pairs[i][1] for i in novel],
                            sorted(act_set), args.workers, "CDK2 활성")

    # ---- 6. 출력
    print("=" * 70)
    print("6. 출력")
    novel_tsv = outdir / "15_novel.tsv.gz"
    with gzip.open(novel_tsv, "wt", encoding="utf-8", newline="\n") as f:
        f.write("gen_id\tsmiles\n")
        for rank, i in enumerate(novel, 1):
            f.write("GEN%05d\t%s\n" % (rank, pairs[i][1]))
    print("   1단계 필터 입력:", novel_tsv)

    sim_csv = outdir / "15_novelty_similarity.csv"
    with open(sim_csv, "w", encoding="utf-8", newline="\n") as f:
        f.write("gen_id,smiles,max_tanimoto_docking996,max_tanimoto_cdk2_actives\n")
        for rank, i in enumerate(novel, 1):
            f.write("GEN%05d,%s,%.4f,%.4f\n"
                    % (rank, pairs[i][1], sim_dock[rank - 1], sim_act[rank - 1]))
    print("   유사도:", sim_csv)

    stats = {
        "prior": {
            "path": str(args.prior),
            "pretrained_on": origin,
            "network_params": net_params,
            "vocab_size": len(id2tok),
            "max_sequence_length": max_len,
            "n_parameters": n_param,
        },
        "sampling": {
            "n_requested": args.n,
            "seed": args.seed,
            "temperature": args.temperature,
            "seconds": round(t_sample, 1),
            "per_second": round(args.n / t_sample, 1),
            "n_truncated_at_max_len": n_truncated,
        },
        "validity": {
            "n_valid": n_valid,
            "validity": round(validity, 4),
            "threshold": 0.85,
            "passed": True,
            "n_standardized": len(std_idx),
        },
        "dedup": {
            "n_unique_within_generated": len(uniq),
            "uniqueness": round(len(uniq) / max(1, len(std_idx)), 4),
            "match_docking996": hit_dock,
            "match_cdk2_actives": hit_act,
            "match_library": hit_lib,
            "n_novel": len(novel),
            "novelty": round(len(novel) / max(1, len(uniq)), 4),
            "library_compared": bool(args.library),
        },
        "similarity_report_only": {
            "docking996": _summary(sim_dock),
            "cdk2_actives": _summary(sim_act),
        },
    }
    with open(outdir / "15_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("   통계:", outdir / "15_stats.json")
    _write_summary(outdir, stats)
    print("   요약:", outdir / "15_summary.md")
    return 0


def _write_raw(outdir, raw, pairs):
    path = outdir / "15_generated_raw.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write("gen_idx,smiles_raw,valid,canonical_plain,canonical_std\n")
        for i, s in enumerate(raw):
            p, st = pairs[i]
            f.write("%d,%s,%d,%s,%s\n" % (i, s, p is not None, p or "", st or ""))
    print("   원본 저장:", path)


def _max_tanimoto(query_smiles, ref_smiles, workers, desc):
    """query 각각에 대해 ref 중 최대 Tanimoto. 반환: list[float]"""
    if not query_smiles or not ref_smiles:
        print("   [%s] 비교할 것이 없어 건너뛴다" % desc)
        return [float("nan")] * len(query_smiles)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_rdkit_worker) as ex:
        qf = [f for f in ex.map(fingerprint, query_smiles, chunksize=100)]
        rf = [f for f in ex.map(fingerprint, ref_smiles, chunksize=100)]
    n_ref_fail = sum(1 for f in rf if f is None)
    n_q_fail = sum(1 for f in qf if f is None)
    if n_ref_fail or n_q_fail:
        print("   [%s] 지문 실패 — 기준집합 %d/%d, 생성분자 %d/%d (제외하고 계산한다)"
              % (desc, n_ref_fail, len(rf), n_q_fail, len(qf)))
    R = np.array([f for f in rf if f is not None], dtype=np.uint8)
    out = []
    rsum = R.sum(1)
    for f in qf:
        if f is None or len(R) == 0:
            out.append(float("nan"))
            continue
        inter = R @ f
        union = rsum + f.sum() - inter
        out.append(float(np.max(inter / np.maximum(union, 1))))
    print("   [%s] query %d × ref %d, %.0f초" % (desc, len(qf), len(R), time.time() - t0))
    return out


def _summary(vals):
    a = np.array([v for v in vals if v == v])      # NaN 제거
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "max": round(float(a.max()), 4),
        "ge_0.4": int((a >= 0.4).sum()),
        "ge_0.7": int((a >= 0.7).sum()),
    }


def _write_summary(outdir, s):
    p, sa, v, d = s["prior"], s["sampling"], s["validity"], s["dedup"]
    lines = [
        "# 6단계 생성 결과 요약",
        "",
        "판정 기준은 `skill/06_reinvent.md` §3 에 결과를 보기 전에 적었다.",
        "",
        "## prior",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| 사전학습 데이터 | %s |" % p["pretrained_on"],
        "| 구조 | Embedding %d → LSTM %d층 × hidden %d → Linear |"
        % (p["network_params"]["embedding_layer_size"],
           p["network_params"]["num_layers"], p["network_params"]["layer_size"]),
        "| vocabulary | %d 토큰 |" % p["vocab_size"],
        "| 파라미터 수 | %s |" % format(p["n_parameters"], ","),
        "",
        "## 생성",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| 생성 개수 | %s |" % format(sa["n_requested"], ","),
        "| seed / temperature | %d / %.2f |" % (sa["seed"], sa["temperature"]),
        "| 소요 시간 | %.1f초 (%.1f개/초) |" % (sa["seconds"], sa["per_second"]),
        "| 최대 길이에서 잘린 것 | %d |" % sa["n_truncated_at_max_len"],
        "",
        "## 깔때기",
        "",
        "| 단계 | 개수 | 비율 |",
        "|---|---|---|",
        "| 생성 | %s | — |" % format(sa["n_requested"], ","),
        "| RDKit 유효 | %s | %.2f%% (기준 ≥ 85%%) |"
        % (format(v["n_valid"], ","), 100 * v["validity"]),
        "| 표준화 통과 | %s | — |" % format(v["n_standardized"], ","),
        "| 생성 내부 고유 | %s | %.2f%% |"
        % (format(d["n_unique_within_generated"], ","), 100 * d["uniqueness"]),
        "| 신규 (기존 집합과 정확 일치 없음) | %s | %.2f%% |"
        % (format(d["n_novel"], ","), 100 * d["novelty"]),
        "",
        "정확 일치로 제거된 것: 도킹 996개 %d, CDK2 활성 데이터 %d, 142만 라이브러리 %d"
        % (d["match_docking996"], d["match_cdk2_actives"], d["match_library"]),
        "" if d["library_compared"] else "⚠ 142만 라이브러리 비교를 하지 않았다 (--library 미지정).",
        "",
        "## Tanimoto 최근접 유사도 — 보고용",
        "",
        "**탈락 기준으로 쓰지 않는다.** 지문이 비슷하다고 활성이 비슷한 것이 아니다.",
        "",
        "| 기준 집합 | 중앙값 | 평균 | p90 | 최대 | ≥0.4 | ≥0.7 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in (("docking996", "도킹 996개"), ("cdk2_actives", "CDK2 활성")):
        t = s["similarity_report_only"][key]
        if t:
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %d | %d |"
                         % (label, t["median"], t["mean"], t["p90"], t["max"],
                            t["ge_0.4"], t["ge_0.7"]))
    lines += ["", "## 다음", "",
              "`07_physchem_filter.py` (1단계와 같은 기준) → "
              "`08_ml_screen.py` (2단계와 같은 모델) → 상위 50개 도킹."]
    (outdir / "15_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
