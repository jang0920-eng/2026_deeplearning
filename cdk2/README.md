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

| 단계 | 방법 | 환경 |
|---|---|---|
| 1. 물리화학 필터 | RDKit (Lipinski, PAINS, lead-like) | CPU |
| 2. 활성 예측 | ChEMBL 2,724개로 학습한 ML 모델 | CPU |
| 3. 다양성 선별 | 클러스터링 | CPU |
| 4. 도킹 | GNINA v1.3.3 (CNN rescore) | Colab T4 |
| 5. ADMET | ADMET 예측 | — |
| 6. 구조 확장 | REINVENT | Colab T4 |

## 진행 상황

작업 내역과 각 결정의 근거는 [`skill/00_log.md`](skill/00_log.md)에 날짜별로 기록한다.

## 참고

대용량 중간 산출물(도킹 포즈 SDF, 화합물 라이브러리 원본)은 저장소에 포함하지 않는다.
재현에 필요한 스크립트와 정제된 결과만 추적한다. 자세한 규칙은 루트의 `.gitignore` 참조.
