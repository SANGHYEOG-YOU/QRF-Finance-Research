# 랜덤 포레스트(Random Forests) 논문 핵심 정리

**원논문:** Leo Breiman (2001), *Random Forests*[cite: 1]
---

## 1. 서론 (Introduction)

랜덤 포레스트(Random forests)는 각 트리가 숲의 모든 트리에 대해 동일한 분포를 가지고 독립적으로 샘플링된 랜덤 벡터(random vector) $\Theta_k$의 값에 의존하는 트리 예측기(tree predictors)의 조합입니다[cite: 1]. 

*   랜덤 포레스트의 일반화 오차(generalization error)는 포레스트 내의 트리 수가 많아짐에 따라 거의 확실하게(almost surely) 극한으로 수렴합니다[cite: 1].
*   이러한 수렴성 덕분에 트리를 지속적으로 추가하더라도 과적합(overfitting) 문제가 발생하지 않습니다[cite: 1].

---

## 2. 분류 정확도의 특성 (Characterizing the Accuracy of Random Forests)

앙상블의 분류기 $h_1(x), h_2(x), \dots, h_K(x)$ 가 주어졌을 때, 마진 함수(margin function)는 올바른 클래스에 대한 평균 투표수가 다른 클래스에 대한 평균 투표수를 얼마나 초과하는지 측정하며 다음과 같이 정의됩니다[cite: 1]:

$$
mg(X,Y) = av_k I(h_k(X)=Y) - \max_{j \neq Y} av_k I(h_k(X)=j)
$$

여기서 $I(\cdot)$는 지시 함수(indicator function)입니다[cite: 1]. 일반화 오차(generalization error)는 다음과 같습니다[cite: 1]:

$$
PE^* = P_{X,Y}(mg(X,Y) < 0)
$$

### 2.1 랜덤 포레스트의 수렴 (Random Forests Converge)

**정리 1.2 (Theorem 1.2)**
트리의 수가 증가함에 따라, 거의 모든 시퀀스 $\Theta_1, \dots$ 에 대하여 $PE^*$는 다음으로 수렴합니다[cite: 1]:

$$
P_{X,Y}(P_\Theta(h(X,\Theta)=Y) - \max_{j \neq Y} P_\Theta(h(X,\Theta)=j) < 0)
$$

*   **증명 요약:** 강한 대수의 법칙(Strong Law of Large Numbers)에 의해 $\frac{1}{N} \sum_{n=1}^N I(h(\Theta_n, x)=j)$ 가 $P_\Theta(h(\Theta, x)=j)$ 로 거의 확실하게 수렴함을 통해 증명됩니다[cite: 1]. 이는 트리를 추가할수록 오차가 한계값에 도달하여 과적합되지 않음을 수학적으로 설명합니다[cite: 1].

### 2.2 강도와 상관관계 (Strength and Correlation)

랜덤 포레스트의 예측 성능은 개별 트리 분류기의 **강도(strength)**와 트리 간의 **상관관계(correlation)**에 의해 결정됩니다[cite: 1].

*   **강도(Strength):** 개별 분류기가 얼마나 정확한지를 나타내는 척도로서, 마진 함수의 기댓값입니다[cite: 1].
    $$s = E_{X,Y} mr(X,Y)$$
*   **원시 마진 함수(Raw margin function):**
    $$rmg(\Theta,X,Y) = I(h(X,\Theta)=Y) - I(h(X,\Theta)=\hat{j}(X,Y))$$

**정리 2.3 (Theorem 2.3)**
일반화 오차에 대한 상한(upper bound)은 다음과 같이 주어집니다[cite: 1]:

$$
PE^* \le \bar{\rho}(1-s^2)/s^2
$$

여기서 $\bar{\rho}$는 개별 분류기 간 원시 마진 함수의 평균 상관관계(mean value of the correlation)입니다[cite: 1].

---

## 3. OOB 추정을 통한 모니터링 (Using Out-Of-Bag Estimates)

배깅(bagging)을 사용할 때, 각 부트스트랩 훈련 세트에는 전체 데이터의 약 1/3이 누락됩니다[cite: 1].
*   이렇게 제외된 데이터를 **OOB(out-of-bag)** 인스턴스라고 부르며, 이를 이용하여 별도의 테스트 세트 없이도 일반화 오차, 분류기의 강도, 그리고 상관관계를 내부적으로 편향 없이 추정할 수 있습니다[cite: 1].

---

## 4. 회귀를 위한 랜덤 포레스트 (Random Forests for Regression)

수치형 값을 출력하는 회귀(regression) 문제에서도 랜덤 포레스트는 동일한 원리로 작동합니다[cite: 1]. 수치형 예측기 $h(x)$에 대한 평균 제곱 일반화 오차는 다음과 같습니다[cite: 1]:

$$
E_{X,Y}(Y-h(X))^2
$$

**정리 11.2 (Theorem 11.2)**
모든 $\Theta$에 대해 $EY = E_X h(X,\Theta)$ 라고 가정할 때, 포레스트의 일반화 오차 $PE^*(forest)$는 다음의 상한을 가집니다[cite: 1]:

$$
PE^*(forest) \le \bar{\rho} PE^*(tree)
$$

---

## 핵심 용어 정리 (Glossary)

| 한국어 | 영어 |
| :--- | :--- |
| **일반화 오차** | Generalization error[cite: 1] |
| **마진 함수** | Margin function[cite: 1] |
| **지시 함수** | Indicator function[cite: 1] |
| **강한 대수의 법칙** | Strong Law of Large Numbers[cite: 1] |
| **평균 상관관계** | Mean value of the correlation[cite: 1] |
