"""
모델 비교에 쓰는 공통 평가 코드.

Morgan baseline(05)과 ChemBERTa(06)가 **같은 함수**로 평가해야
같은 기준에서 비교했다고 말할 수 있으므로 여기에 모아 둔다.

지표는 W09 수업에서 쓴 것 그대로:
  precision, recall, PR-AUC, AP, ROC-AUC, EF 5%

PR-AUC 와 AP 를 둘 다 계산하는 이유: AP 는 계단 합이고
PR-AUC 는 사다리꼴 적분이라 값이 다를 수 있다. 수업에서 명시하라고 한 부분.
"""

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def enrichment_factor(y_true, scores, frac=0.05):
    """상위 frac 안의 활성 비율 ÷ 전체 활성 비율. 1이면 무작위와 같다."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = max(1, int(round(len(scores) * frac)))
    top = np.argsort(-scores)[:n]
    base = y_true.mean()
    if base == 0:
        return float("nan")
    return float(y_true[top].mean() / base)


def ef_ceiling(y_true, frac=0.05):
    """EF 의 이론적 최대값.

    상위 n 개를 전부 활성으로 맞혀도 EF 는 (n/n)/base = 1/base 를 넘을 수 없다.
    활성 비율이 높은 데이터셋에서는 이 천장이 낮아 EF 가 모델을 구분하지 못한다.
    """
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(y_true) * frac)))
    base = y_true.mean()
    if base == 0:
        return float("nan")
    return float(min(n, y_true.sum()) / n / base)


def pick_threshold(y_val, scores_val):
    """val 에서 F1 이 가장 큰 threshold 를 고른다. test 는 쓰지 않는다."""
    prec, rec, thr = precision_recall_curve(y_val, scores_val)
    f1 = np.divide(
        2 * prec * rec, prec + rec,
        out=np.zeros_like(prec), where=(prec + rec) > 0,
    )
    # precision_recall_curve 는 마지막 점에 대응하는 threshold 가 없다
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def evaluate(y_true, scores, threshold, label=""):
    """threshold 기반 지표와 순위 기반 지표를 함께 반환."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    prec_c, rec_c, _ = precision_recall_curve(y_true, scores)
    return dict(
        model=label,
        n=len(y_true),
        actives=int(y_true.sum()),
        threshold=round(threshold, 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        pr_auc=round(auc(rec_c, prec_c), 4),            # 사다리꼴 적분
        ap=round(average_precision_score(y_true, scores), 4),  # 계단 합
        roc_auc=round(roc_auc_score(y_true, scores), 4),
        ef1=round(enrichment_factor(y_true, scores, 0.01), 3),
        ef5=round(enrichment_factor(y_true, scores, 0.05), 3),
        ef10=round(enrichment_factor(y_true, scores, 0.10), 3),
        ef20=round(enrichment_factor(y_true, scores, 0.20), 3),
        ef_max=round(ef_ceiling(y_true, 0.05), 3),
    )


def results_table(rows, title):
    """skill 문서에 붙일 마크다운 표."""
    md = [f"### {title}\n",
          "| 모델 | precision | recall | PR-AUC | AP | ROC-AUC | EF 5% | EF 10% | EF 20% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['precision']} | {r['recall']} | {r['pr_auc']} | "
            f"{r['ap']} | {r['roc_auc']} | {r['ef5']} | {r['ef10']} | {r['ef20']} |"
        )
    return "\n".join(md)
