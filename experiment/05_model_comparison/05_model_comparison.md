### 활성 예측 모델 비교 (scaffold split test, 마지막 1회 평가)

| 모델 | precision | recall | PR-AUC | AP | ROC-AUC | EF 5% | EF 10% | EF 20% |
|---|---|---|---|---|---|---|---|---|
| Morgan + LogReg | 0.7405 | 0.7358 | 0.7611 | 0.763 | 0.7345 | 1.349 | 1.208 | 1.405 |
| Morgan + RF | 0.7403 | 0.8428 | 0.8807 | 0.8811 | 0.8231 | 1.717 | 1.717 | 1.686 |
| ChemBERTa-10M-MTR + LogReg | 0.7643 | 0.7547 | 0.8355 | 0.8364 | 0.7875 | 1.594 | 1.653 | 1.53 |
| ChemBERTa-10M-MTR + RF | 0.7 | 0.8365 | 0.8637 | 0.8642 | 0.7985 | 1.717 | 1.717 | 1.655 |

- test 273개 중 활성 159개 (기준선 58.2%)
- EF 5% = 상위 5%의 활성 비율 ÷ 전체 활성 비율. **1이면 무작위와 같음**
- ⚠ **이 test 에서 EF 의 이론적 최대값은 1.717** (= 1 / 활성 비율 0.582). 상위를 전부 활성으로 맞혀도 이 값을 넘을 수 없다.
  활성 비율이 높아 EF 가 천장에 붙으므로, 모델 구분은 PR-AUC·ROC-AUC 로 한다.
  실제 스크리닝에서는 활성 비율이 훨씬 낮아 EF 의 천장이 높고 지표로서 의미가 있다.
- PR-AUC 는 사다리꼴 적분, AP 는 계단 합이라 값이 다를 수 있어 둘 다 적는다
- hyperparameter 와 threshold 는 val 에서 선택했고 test 는 한 번만 평가했다
- scaler 는 train 에서만 fit 했다 (누출 방지)
- ChemBERTa 는 **frozen**, encoder 출력을 masked mean pooling 한 384차원 벡터

| 모델 | val AP | 선택된 hyperparameter |
|---|---|---|
| Morgan + LogReg | 0.778 | `{"C": 0.1}` |
| Morgan + RF | 0.8484 | `{"n_estimators": 500, "max_depth": null, "min_samples_leaf": 1}` |
| ChemBERTa-10M-MTR + LogReg | 0.7941 | `{"C": 0.01}` |
| ChemBERTa-10M-MTR + RF | 0.8388 | `{"n_estimators": 500, "max_depth": null, "min_samples_leaf": 1}` |