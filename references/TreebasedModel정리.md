# [ESL] § 9.2 Tree-Based Methods --- 한국어 정리

**The Elements of Statistical Learning** *Hastie, Tibshirani, Friedman*

---

## 1. 배경 (Background)

트리 기반 방법(tree-based methods)은 **특징 공간(feature space)을 직사각형 영역들로 분할(partition)** 한 뒤, 각 영역에서 단순한 모형(예: 상수)을 적합합니다. 개념적으로 단순하지만 강력한 방법입니다. 본 절에서는 **CART**(Classification And Regression Tree)를 중심으로 다루고, 경쟁 알고리즘인 **C4.5**와 대조합니다.

### 재귀적 이진 분할 (Recursive Binary Partitioning)

좌표축에 평행한 임의의 분할은 표현이 복잡할 수 있으므로 **재귀적 이진 분할**로 제한합니다.

1. 먼저 공간을 두 영역으로 분할하고, 각 영역에서 $Y$의 평균으로 반응을 모형화.
2. 최적 적합을 주는 **분할 변수(splitting variable)**와 **분할점(split point)**을 선택.
3. 한쪽 또는 양쪽 영역을 다시 둘로 분할하며 **정지 규칙(stopping rule)**이 만족될 때까지 반복.

5개 영역 $R_1, R_2, \dots, R_5$로 분할된 경우의 회귀 모형:
$$\hat{f}(X) = \sum_{m=1}^{5} c_m \, I\{(X_1, X_2) \in R_m\} \tag{9.9}$$

이 모형은 **이진 트리(binary tree)**로 표현 가능하며, 트리의 **단말 노드(terminal nodes / leaves)**가 영역 $R_m$ 에 대응합니다.

> [!TIP]
> **핵심 장점**: 해석가능성(interpretability). 입력이 두 개 이상이어도 트리 표현은 동일하게 작동하며, 이는 의사가 사고하는 방식을 모방하기 때문에 의료 분야에서 특히 인기가 있습니다.

---

## 2. 회귀 트리 (Regression Trees)

데이터 $(x_i, y_i)$, $i = 1, 2, \dots, N$, $x_i = (x_{i1}, x_{i2}, \dots, x_{ip})$ 가 주어졌을 때, 알고리즘은 분할 변수, 분할점, 트리의 **위상(topology)** 을 자동으로 결정해야 합니다.

$M$ 개 영역 $R_1, \dots, R_M$ 으로 분할되었을 때 각 영역에서 상수 $c_m$ 으로 모형화하면:
$$f(x) = \sum_{m=1}^{M} c_m \, I(x \in R_m) \tag{9.10}$$

**잔차제곱합(sum of squares)** $\sum (y_i - f(x_i))^2$ 을 최소화하는 최적 $\hat{c}_m$ 은 영역 $R_m$ 내 $y_i$ 의 평균입니다:
$$\hat{c}_m = \text{ave}(y_i \mid x_i \in R_m) \tag{9.11}$$

### 탐욕 알고리즘 (Greedy Algorithm)

최소 잔차제곱합을 주는 최적 이진 분할을 찾는 것은 일반적으로 계산적으로 불가능(infeasible)하므로 **탐욕 알고리즘**을 사용합니다. 분할 변수 $j$ 와 분할점 $s$ 에 대해 다음 두 반평면(half-plane)을 정의합니다:
$$R_1(j, s) = \{X \mid X_j \leq s\}, \quad R_2(j, s) = \{X \mid X_j > s\} \tag{9.12}$$

다음을 푸는 $(j, s)$ 를 찾습니다:
$$\min_{j,\, s}\Bigg[\, \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 \;+\; \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \,\Bigg] \tag{9.13}$$

내부 최소화의 해:
$$\hat{c}_1 = \text{ave}(y_i \mid x_i \in R_1(j, s)), \quad \hat{c}_2 = \text{ave}(y_i \mid x_i \in R_2(j, s)) \tag{9.14}$$

### 비용 복잡도 가지치기 (Cost-Complexity Pruning)

**트리 크기는 모형 복잡도를 지배하는 튜닝 파라미터(tuning parameter)** 이며 데이터로부터 적응적으로 선택해야 합니다. 큰 트리 $T_0$ 를 키우되 **최소 노드 크기**(예: 5)에 도달할 때만 분할을 멈춘 뒤, **비용 복잡도 가지치기(cost-complexity pruning)** 로 가지치기합니다.

**비용 복잡도 기준(cost complexity criterion):**
> [!IMPORTANT]
> $$C_\alpha(T) = \sum_{m=1}^{|T|} N_m\, Q_m(T) + \alpha\, |T| \tag{9.16}$$
> - $\alpha \geq 0$ 는 트리 크기와 데이터 적합도 사이의 절충(tradeoff)을 지배합니다.
> - 각 $\alpha$ 에 대해 $C_\alpha(T)$ 를 최소화하는 **유일한 최소 부분트리** $T_\alpha$ 가 존재합니다.

---

## 3. 분류 트리 (Classification Trees)

목표가 $1, 2, \dots, K$ 값을 갖는 **분류 결과(classification outcome)** 인 경우, 알고리즘에서 변경되는 부분은 **노드 분할 기준**과 **가지치기 기준**뿐입니다.

### 노드 불순도 (Node Impurity)

노드 $m$ 의 **노드 불순도 측도** $Q_m(T)$:

- **오분류율 (Misclassification error):** $1 - \hat{p}_{m\,k(m)}$
- **지니 지수 (Gini index):** $\sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk}) \tag{9.17}$
- **교차 엔트로피 / 이탈도 (Cross-entropy / deviance):** $-\sum_{k=1}^{K} \hat{p}_{mk} \log \hat{p}_{mk}$

> [!NOTE]
> **어느 측도를 쓸 것인가?**
> - **성장 시**: 지니 지수나 교차 엔트로피는 노드 확률 변화에 더 민감하여 분할에 유리합니다.
> - **가지치기 시**: 보통 오분류율을 사용합니다.

---

## 4. 기타 사항 (Other Issues)

### 범주형 예측변수 (Categorical Predictors)
순서 없는 $q$ 개 값을 갖는 경우 가능한 분할 수는 $2^{q-1} - 1$입니다.

> [!CAUTION]
> **경고**: 레벨 수 $q$ 가 많은 범주형 예측변수는 데이터에 우연히 잘 맞는 분할을 찾을 가능성이 커서 **심각한 과적합**이 발생할 수 있습니다.

### 결측 예측값 (Missing Predictor Values)
- **대체 변수 (Surrogate variables)**: 주 분할의 결과를 가장 잘 모방하는 대체 변수 목록을 만들어 결측 시 사용합니다.

### 트리의 불안정성 (Instability)
트리의 주요 문제는 **높은 분산(high variance)**입니다. 데이터의 작은 변화가 상위 노드 분할을 바꾸면 아래 모든 구조가 변합니다. 이는 **배깅(Bagging)**이나 **랜덤 포레스트**로 보완 가능합니다.

---

## 5. 스팸 예제 (Spam Example)

- **혼동 행렬 (Confusion Matrix)**

| True \ Predicted | email | spam |
| :--- | :--- | :--- |
| **email** | 57.3% | 4.0% |
| **spam** | 5.3% | 33.4% |

- **ROC 곡선**: 민감도(Sensitivity) vs 특이도(Specificity)를 나타냅니다. 
- **c-통계량**: ROC 곡선 아래 면적(AUC)과 같으며, Mann-Whitney U 통계량과 동등합니다.

---

## 📝 핵심 요점 정리

| 개념 | 핵심 수식 / 정의 |
| :--- | :--- |
| **회귀 트리 모형** | $f(x) = \sum_{m=1}^{M} c_m I(x \in R_m)$ |
| **최적 상수** | $\hat{c}_m = \text{ave}(y_i \mid x_i \in R_m)$ |
| **비용 복잡도 기준** | $C_\alpha(T) = \text{Error} + \alpha \|T\|$ |
| **지니 지수** | $\sum_k \hat{p}_{mk}(1 - \hat{p}_{mk})$ |

> [!IMPORTANT]
> **핵심 메시지**:
> 트리는 특징 공간을 직사각형으로 분할하는 **재귀적 이진 분할**로 적합됩니다. **탐욕 알고리즘**으로 큰 트리를 키운 뒤 **비용 복잡도 가지치기**로 최적의 크기를 결정하며, 높은 분산과 매끄러움의 결여는 앙상블 기법으로 해결합니다.

---
_Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning._
