# CDK2 Virtual Screening — 딥러닝 기말 프로젝트

CDK4/6 억제제 내성 HR+/HER2− 전이성 유방암을 표적으로,
CDK2 저해 신약 후보 물질을 virtual screening으로 도출한다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `experiment/` | 보고서에 쓰이는 계산 결과 (CSV, 그림) |
| `scripts/` | 계산에 사용한 스크립트와 Colab 노트북 |
| `skill/` | 보고서 작성 AI(Solar Pro4)가 참고하는 문서 |
| `report/` | 최종 보고서 |

## 타겟

- 단백질: CDK2 (ChEMBL `CHEMBL301`)
- 구조: PDB `3RPR` (1.75 Å, 리간드 25Z, chain A)

## 선별 파이프라인

| 단계 | 방법 | 들어감 → 나옴 | 환경 |
|---|---|---|---|
| 단계 | 방법 | 들어감 → 나옴 | 환경 | 상태 |
|---|---|---|---|---|
| 1. 물리화학 필터 | RDKit (Lipinski, PAINS, lead-like) | 2,897,819 → 1,423,957 | 로컬 CPU | ✅ |
| 2. 활성 예측 | Morgan fingerprint + Random Forest | 1,423,957 → 20,000 | 로컬 CPU | ✅ |
| 3. 다양성 선별 | sphere exclusion (Tanimoto 0.6) | 10,000 → **996** | 로컬 CPU | ✅ |
| 4. 도킹 | GNINA v1.3.3 (CNN rescore) | 996 → **100** | Colab T4 | ✅ 995/996 (실패 1) |
| 5. ADMET | ADMET-AI (chemprop D-MPNN, TDC) | 100 → **30** | 로컬 CPU | ✅ |
| 6. 구조 확장 | REINVENT prior (SMILES-LSTM) sampling | 10,000 생성 → 신규 9,609 → 필터 5,581 → **50** | 로컬 CPU | ✅ 생성·선별 완료 |

3단계는 1,000개를 뽑았고 3D 구조 생성에서 4개가 실패해 **도킹 입력은 996개**다.

6단계는 1~3단계와 **반대 방향**이다. 1~5단계는 ChEMBL 이라는 기존 집합을 좁히고,
6단계는 그 집합에 없는 SMILES 를 새로 만든다. 1단계 필터와 2단계 모델은 **기준을
바꾸지 않고 그대로 재사용**했다. 생성 분자의 ML 점수 중앙값은 0.342 로 3단계가
선별한 1,000개(0.806)보다 낮다 — prior 가 CDK2 를 모르기 때문이고, 근거와 한계는
`skill/06_reinvent.md` 에 있다.

## 활성 예측 모델 — 분자 표현 3계열 비교

같은 scaffold split, 같은 평가 함수(`scripts/eval_utils.py`)로 비교했다.
hyperparameter 와 threshold 는 val 에서 고르고 test 는 마지막에 한 번만 평가했다.

| 계열 | 모델 | PR-AUC | ROC-AUC |
|---|---|---|---|
| 지문 | **Morgan + RF** (채택) | **0.8807** | **0.8231** |
| 트랜스포머 | ChemBERTa-10M-MTR + RF | 0.8637 | 0.7985 |
| 그래프 | Chemprop D-MPNN | 0.8577 | 0.8107 |

표현을 학습하는 두 계열이 고정된 지문에 졌다. 학습 데이터가 2,721개 규모라
표현을 배울 데이터가 부족하다. 전체 7개 조합은 `skill/03_screening.md` 참조.

## 재현 순서

저장소 루트에서 실행한다. `--library` 는 1단계 출력 파일 경로다.

```bash
python scripts/01_fetch_chembl_cdk2.py                  # ChEMBL 활성 데이터
python scripts/04_scaffold_split.py                     # scaffold split
python scripts/05_model_comparison.py                   # 지문·트랜스포머 6조합 비교
python scripts/07_physchem_filter.py --input chembl_37_chemreps.txt.gz \
                                     --outdir experiment/07_filter
python scripts/08_ml_screen.py --library experiment/07_filter/07_filtered_unique.csv.gz
python scripts/09_diversity_select.py                   # 도킹 1,000개 선별
python scripts/10_ligand_prep.py                        # 3D 구조 (seed 고정)

# 4단계 도킹은 Colab: scripts/11_dock_1000.ipynb → 11_top_for_admet.csv

# 아래 둘은 chemprop 환경에서 (CPU, GPU 불필요)
conda activate chemprop-env
python scripts/12_chemprop_compare.py                   # 그래프 계열 추가 비교
python scripts/13_admet.py --input experiment/11_docking/11_top_for_admet.csv \
                           --outdir experiment/13_admet

python scripts/06_figures.py && python scripts/12_figures_screen.py   # 그림
```

### 환경

| 환경 | 용도 | 주요 패키지 |
|---|---|---|
| `my-rdkit-env` | 1~3단계 계산 | rdkit, scikit-learn, pandas |
| `aizynth-env` | 그림 | matplotlib, scipy |
| `chemprop-env` | 그래프 모델·ADMET | chemprop 2.3.1, admet-ai 2.0.1, torch 2.15.0+cpu |

```bash
conda create -y -n chemprop-env python=3.11
conda activate chemprop-env && pip install chemprop admet-ai
```

난수는 전부 고정돼 있다 — RF `random_state=42`, ETKDGv3 `randomSeed=42`,
GNINA `--seed 1`. scaffold split 은 난수를 쓰지 않는다(묶음 크기 순 배정).

## 진행 상황

**2026-11-22 기준: 1~5단계 완료.** 도킹 995/996 성공(총 21.7시간, 실패 1개),
ADMET 하드 필터 통과 30개가 최종 우선순위다. 남은 것은 6단계 생성 분자 50개
도킹과 보고서 작성이다.

최종 후보 30개는 ADMET 통과분 중 상위 30개를 **고른 것이 아니라 통과한 전부**다.
문턱값(승인 약물 백분위 90)에 따라 통과 수가 2~56개로 변하므로 개수 자체에
의미를 두지 않는다. 경위는 [`skill/04_admet.md`](skill/04_admet.md) §6 에 있다.

작업 내역과 각 결정의 근거는 [`skill/00_log.md`](skill/00_log.md)에 날짜별로 기록한다.
지표를 그대로 쓰지 않고 적용 조건을 먼저 확인한 사례 두 건(EF 천장, ADMET 확률 문턱)도
거기에 경위까지 적혀 있다.

## 참고

대용량 중간 산출물(도킹 포즈 SDF, 화합물 라이브러리 원본)은 저장소에 포함하지 않는다.
재현에 필요한 스크립트와 정제된 결과만 추적한다. 자세한 규칙은 `.gitignore` 참조.
