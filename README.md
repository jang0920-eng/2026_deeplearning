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
| 1. 물리화학 필터 | RDKit (Lipinski, PAINS, lead-like) | 2,897,819 → 1,423,957 | CPU |
| 2. 활성 예측 | Morgan fingerprint + Random Forest | 1,423,957 → 20,000 | CPU |
| 3. 다양성 선별 | sphere exclusion (Tanimoto 0.6) | 10,000 → 1,000 | CPU |
| 4. 도킹 | GNINA v1.3.3 (CNN rescore) | 1,000 → 약 100 | Colab T4 |
| 5. ADMET | ADMET 예측 | 약 100 → 15~30 | — |
| 6. 구조 확장 | REINVENT (선택) | — | Colab T4 |

## 재현 순서

저장소 루트에서 실행한다. `--library` 는 1단계 출력 파일 경로다.

```bash
python scripts/01_fetch_chembl_cdk2.py                  # ChEMBL 활성 데이터
python scripts/04_scaffold_split.py                     # scaffold split
python scripts/05_model_comparison.py                   # 모델 6조합 비교
python scripts/07_physchem_filter.py --input chembl_37_chemreps.txt.gz \
                                     --outdir experiment/07_filter
python scripts/08_ml_screen.py --library experiment/07_filter/07_filtered_unique.csv.gz
python scripts/09_diversity_select.py                   # 도킹 1,000개 선별
python scripts/10_ligand_prep.py                        # 3D 구조 (seed 고정)
python scripts/06_figures.py && python scripts/12_figures_screen.py
# 4단계 도킹은 Colab: scripts/11_dock_1000.ipynb
```

난수는 전부 고정돼 있다 — RF `random_state=42`, ETKDGv3 `randomSeed=42`,
GNINA `--seed 1`. scaffold split 은 난수를 쓰지 않는다(묶음 크기 순 배정).

## 진행 상황

작업 내역과 각 결정의 근거는 [`skill/00_log.md`](skill/00_log.md)에 날짜별로 기록한다.

## 참고

대용량 중간 산출물(도킹 포즈 SDF, 화합물 라이브러리 원본)은 저장소에 포함하지 않는다.
재현에 필요한 스크립트와 정제된 결과만 추적한다. 자세한 규칙은 `.gitignore` 참조.
