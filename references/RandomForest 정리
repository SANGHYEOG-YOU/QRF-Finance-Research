# 랜덤 포레스트(Random Forest) 가이드 투어 — 한국어 정리

**원논문**: Biau & Scornet (2015), *A Random Forest Guided Tour*, arXiv:1511.05741

---

## 1. 서론 (Introduction)

랜덤 포레스트(Random Forest)는 Breiman(2001)이 제안한 알고리즘으로, **분할정복(divide and conquer)** 원리에 따라 작동한다: 데이터의 일부를 샘플링하고, 각 조각에 무작위화된 결정트리(randomized decision tree)를 학습시킨 뒤, 이 예측기들을 결합(aggregate)한다.

핵심 구성요소:
- **배깅(Bagging, Bootstrap Aggregating)**: Breiman(1996)이 제안한 일반적 결합 기법
- **CART 분할 기준(CART-split criterion)**: Breiman et al.(1984)의 영향력 있는 CART에서 유래

이 두 요소가 핵심이지만 수학적으로 분석하기 어려워서, 이론 연구는 주로 단순화된 버전을 다룬다.

---

## 2. 랜덤 포레스트 추정치 (The Random Forest Estimate)

### 2.1 기본 원리 (Basic Principles)

**회귀(regression) 설정**:
- 입력 확률벡터 $X \in \mathcal{X} \subset \mathbb{R}^p$
- 제곱적분가능한 응답변수 $Y \in \mathbb{R}$
- 회귀함수 $m(x) = \mathbb{E}[Y|X=x]$ 추정이 목표
- 학습 표본 $\mathcal{D}_n = ((X_1, Y_1), \ldots, (X_n, Y_n))$

**일치성(consistency) 정의**: $\mathbb{E}[m_n(X) - m(X)]^2 \to 0$ as $n \to \infty$

**$j$번째 트리 추정값**:
$$m_n(x; \Theta_j, \mathcal{D}_n) = \sum_{i \in \mathcal{D}_n^*(\Theta_j)} \frac{\mathbf{1}_{X_i \in A_n(x; \Theta_j, \mathcal{D}_n)} Y_i}{N_n(x; \Theta_j, \mathcal{D}_n)}$$

여기서:
- $\mathcal{D}_n^*(\Theta_j)$: 트리 구성 전 선택된 데이터 포인트 집합
- $A_n(x; \Theta_j, \mathcal{D}_n)$: $x$를 포함하는 셀(cell)
- $N_n(x; \Theta_j, \mathcal{D}_n)$: 해당 셀에 속한 (사전 선택된) 점의 수
- $\Theta_1, \ldots, \Theta_M$: 독립이고 동일 분포를 따르는 무작위 변수

**유한 포레스트 추정치 (finite forest estimate)**:
$$m_{M,n}(x; \Theta_1, \ldots, \Theta_M, \mathcal{D}_n) = \frac{1}{M} \sum_{j=1}^{M} m_n(x; \Theta_j, \mathcal{D}_n) \quad (1)$$

**무한 포레스트 추정치 (infinite forest estimate)**: $M \to \infty$ 극한에서
$$m_{\infty, n}(x; \mathcal{D}_n) = \mathbb{E}_\Theta [m_n(x; \Theta, \mathcal{D}_n)]$$

R 패키지 `randomForest`의 기본값은 `ntree = 500`이다.

### 2.2 알고리즘 (Algorithm)

**핵심 파라미터 3개**:
1. $a_n \in \{1, \ldots, n\}$: 각 트리에서 샘플링되는 데이터 포인트 수
2. $\texttt{mtry} \in \{1, \ldots, p\}$: 각 노드에서 분할 후보 방향 수
3. $\texttt{nodesize} \in \{1, \ldots, a_n\}$: 셀이 분할되지 않는 최소 샘플 수

**기본값 (회귀, R 패키지 randomForest)**:
- $\texttt{mtry} = \lceil p/3 \rceil$
- $a_n = n$
- $\texttt{nodesize} = 5$

**CART 분할 기준 (회귀)**:

셀 $A$에서 가능한 분할 쌍 $(j, z)$ ($j$는 차원, $z$는 분할 위치)에 대해:

$$L_{\text{reg},n}(j, z) = \frac{1}{N_n(A)} \sum_{i=1}^{n} (Y_i - \bar{Y}_A)^2 \mathbf{1}_{X_i \in A}$$
$$- \frac{1}{N_n(A)} \sum_{i=1}^{n} (Y_i - \bar{Y}_{A_L} \mathbf{1}_{X_i^{(j)} < z} - \bar{Y}_{A_R} \mathbf{1}_{X_i^{(j)} \geq z})^2 \mathbf{1}_{X_i \in A} \quad (2)$$

여기서 $A_L = \{x \in A : x^{(j)} < z\}$, $A_R = \{x \in A : x^{(j)} \geq z\}$.

최적 분할:
$$(j_n^*, z_n^*) \in \arg\max_{\substack{j \in \mathcal{M}_{\text{try}} \\ (j,z) \in \mathcal{C}_A}} L_{\text{reg},n}(j, z)$$

**Breiman의 트리와 일반 CART의 3가지 차이**:
1. 기준 (2)가 전체 좌표가 아닌 무작위 선택된 부분집합 $\mathcal{M}_{\text{try}}$ 위에서 평가됨
2. 개별 트리는 가지치기(pruning)되지 않으며, 최종 셀은 $\texttt{nodesize}$ 이하의 관측치를 가짐
3. 각 트리는 전체 표본이 아닌 $a_n$개 부분표본으로 구성됨
   - $a_n = n$이고 복원 추출이면 **부트스트랩(bootstrap) 모드**
   - $a_n < n$이면 **서브샘플링(subsampling) 모드**

### 2.3 지도 분류 (Supervised Classification)

이진 분류에서 $Y \in \{0, 1\}$. **베이즈 분류기(Bayes classifier)**:
$$m^*(x) = \begin{cases} 1 & \text{if } \mathbb{P}[Y=1|X=x] > \mathbb{P}[Y=0|X=x] \\ 0 & \text{otherwise} \end{cases}$$

**랜덤 포레스트 분류기**: 트리들의 다수결(majority vote)
$$m_{M,n}(x; \Theta_1, \ldots, \Theta_M, \mathcal{D}_n) = \begin{cases} 1 & \text{if } \frac{1}{M} \sum_{j=1}^{M} m_n(x; \Theta_j, \mathcal{D}_n) > 1/2 \\ 0 & \text{otherwise} \end{cases}$$

**분류용 CART 분할 기준**:
$$L_{\text{class},n}(j, z) = p_{0,n}(A) p_{1,n}(A) - \frac{N_n(A_L)}{N_n(A)} \times p_{0,n}(A_L) p_{1,n}(A_L) - \frac{N_n(A_R)}{N_n(A)} \times p_{0,n}(A_R) p_{1,n}(A_R)$$

이는 **지니 불순도(Gini impurity)** $2 p_{0,n}(A) p_{1,n}(A)$에 기반한다.

분류에서 권장값: $\texttt{nodesize} = 1$, $\texttt{mtry} = \sqrt{p}$.

### 2.4 파라미터 튜닝 (Parameter Tuning)

- **$M$ (트리 수)**: 클수록 분산이 감소하며 과적합되지 않는다. 다만 계산 비용이 $M$에 선형적으로 증가:
  $$\lim_{n \to \infty} \mathbb{E}[m_{M,n}(X; \Theta_1, \ldots, \Theta_M) - m(X)]^2 = \mathbb{E}[m_{\infty,n}(X) - m(X)]^2$$
- **$\texttt{nodesize}$**: 분류 1, 회귀 5가 일반적으로 좋은 선택
- **$\texttt{mtry}$**: 영향이 작지만, 보수적으로 $\texttt{mtry} = p$로 설정하기도 함
- **OOB 오차(Out-of-bag error)**: 부트스트랩 표본에 포함되지 않은 약 1/3의 관측치를 이용해 검증 표본 없이 파라미터 조정 가능

---

## 3. 단순화된 모델과 국소 평균 추정 (Simplified Models and Local Averaging Estimates)

### 3.1 단순화된 모델 (Simplified Models)

**순수 무작위 포레스트 (Purely Random Forests)**: 트리가 학습 표본 $\mathcal{D}_n$과 독립적으로 설계됨.

**중심 포레스트(Centered Forest)** ($\mathcal{X} = [0, 1]^d$):
1. 리샘플링 단계 없음
2. 각 노드에서 좌표를 $\{1, \ldots, p\}$에서 균등 무작위 선택
3. 선택된 좌표를 따라 셀의 **중심**에서 분할
4. $k$번 재귀 반복 → $2^k$개의 잎(leaf)을 가진 완전 이진 트리

**일치성(Consistency)** (Breiman 2004; Biau et al. 2008; Scornet 2015a):
$k \to \infty$ 및 $n / 2^k \to \infty$이면 분류·회귀 모두 일치.

**수렴률(Rates of Convergence)**:

희소성(sparsity) 가정: $m(x) = \mathbb{E}[Y | X_{\mathcal{S}} = x_{\mathcal{S}}]$, 즉 회귀함수가 강(strong) 변수 부분집합 $\mathcal{S}$에만 의존. 그러면:

$$\mathbb{E}[m_{\infty, n}(X) - m(X)]^2 = O\left( n^{-\frac{0.75}{|\mathcal{S}| \log 2 + 0.75}} \right)$$

이 속도는 $|\mathcal{S}|$에만 의존하며 $p$에 무관하다. $|\mathcal{S}| \leq \lfloor 0.54 p \rfloor$이면 일반적인 $n^{-2/(p+2)}$보다 엄격하게 빠르다.

**순수 균등 무작위 포레스트(Purely Uniform Random Forest, PURF)** (Genuer 2012, $p=1$):
립시츠(Lipschitz) 가정 하에서
$$\mathbb{E}[m_{\infty, n}(X) - m(X)]^2 = O(n^{-2/3})$$
이는 립시츠 함수 클래스에서 미니맥스(minimax) 속도이다.

### 3.2 포레스트, 최근접 이웃, 커널 (Forests, Neighbors, Kernels)

**계층 최근접 이웃(Layered Nearest Neighbor, LNN)**: 점 $x$와 $X_i$로 정의되는 초직사각형(hyperrectangle)에 다른 데이터 포인트가 없는 경우 $X_i$를 $x$의 LNN이라 함.

리샘플링이 없고 잎에 정확히 한 점만 남는 포레스트의 추정값은 LNN의 가중 평균이다:

$$m_{\infty, n}(x) = \sum_{i=1}^{n} W_{ni}(x) Y_i \quad (3)$$

가중치 $W_{ni}(x) \geq 0$, $\sum_i W_{ni} = 1$, $X_i$가 LNN이 아니면 $W_{ni}(x) = 0$.

Lin & Jeon(2006): $X$가 $[0, 1]^p$에서 균등이고 트리 성장이 $Y_1, \ldots, Y_n$과 독립(non-adaptive)이면
$$\mathbb{E}[m_{\infty, n}(X) - m(X)]^2 = O\left( \frac{1}{n_{\max}(\log n)^{p-1}} \right)$$

**커널 표현(Kernel Representation)**:
$M \to \infty$일 때 (잔차항 무시):
$$m_{\infty, n}(x) \approx \frac{\sum_{i=1}^n Y_i K_n(X_i, x)}{\sum_{j=1}^n K_n(X_j, x)} \quad (4)$$

**커널(kernel)**: $K_n(x, z) = \mathbb{P}_\Theta[z \in A_n(x, \Theta)]$ — $x$와 $z$가 한 셀에 속할 확률.

중심 포레스트의 경우 (Scornet 2015b):
$$K_{n,k}(x, z) = \sum_{\substack{k_1, \ldots, k_p \\ \sum_{j=1}^p k_j = k}} \frac{k!}{k_1! \cdots k_p!} \left(\frac{1}{p}\right)^k \prod_{j=1}^p \mathbf{1}_{\lceil 2^{k_j} x_j \rceil = \lceil 2^{k_j} z_j \rceil}$$

$K_n$은 일반적으로 Nadaraya-Watson 형식의 평행이동 불변(translation-invariant) 동차 커널이 아니다.

---

## 4. Breiman 포레스트 이론 (Theory for Breiman's Forests)

### 4.1 리샘플링 메커니즘 (Resampling Mechanism)

부트스트랩(bootstrap)은 $n$개에서 $n$번 복원 추출. 이론 분석 시 대부분 부트스트랩을 **서브샘플링(subsampling, $a_n < n$ 비복원 추출)**으로 대체하고, 일반적으로 $a_n / n \to 0$을 가정한다.

**중앙값 포레스트(Median Forest)** (Scornet 2015a): 분할 방향 선택 후 셀의 경험적 중앙값에서 분할. $a_n / n \to 0$이면 개별 트리가 비일치라도 포레스트는 일치한다.

**서브배깅된 1-NN(1-Nearest Neighbor) 추정량** (Biau & Devroye 2010):
$$r_n^*(x) = \mathbb{E}^*[r_{a_n}(x)]$$
$a_n \to \infty$ 및 $a_n / n \to 0$이면 보편적(universally) 평균제곱 일치(mean squared consistent).

### 4.2 결정 분할 (Decision Splits)

**분할 위치의 극한 분포** (Banerjee & McKeague 2007): $Y = m(X) + \varepsilon$ 모형, $X$가 실수값, $\varepsilon$는 독립 가우시안 잡음일 때:

$$n^{1/3} \begin{pmatrix} \hat{\beta}_{\ell, n} - \beta_\ell^* \\ \hat{\beta}_{r, n} - \beta_r^* \\ \hat{d}_n - d^* \end{pmatrix} \xrightarrow{D} \begin{pmatrix} c_1 \\ c_2 \\ 1 \end{pmatrix} \arg\max_t (a W(t) - b t^2) \quad (5)$$

여기서 $W$는 양방향 표준 브라운 운동(two-sided Brownian motion).

**ECP(End-Cut Preference, 끝-자르기 선호)** (Ishwaran 2013): CART 분할이 비정보적 변수에 대해서는 셀 가장자리 근처에서 잘리는 경향. 한쪽 자식 셀의 표본 수가 절반으로 줄어드는 것을 막아 **나무가 다운스트림에서 회복할 가능성**을 보존한다는 점에서 바람직한 성질.

**희소성 적응** (Scornet et al. 2015): 정규성 조건 하에서 모든 방향이 분할 후보로 사전선택되는 변형 모델에서, 확률 $1 - \xi$ 이상으로 충분히 큰 $n$과 $1 \leq q \leq k$에 대해
$$j_{n,q}(X) \in \{1, \ldots, S\}$$
즉 분할이 주로 정보적(informative) 변수를 따라 일어남.

### 4.3 일치성, 점근 정규성 (Consistency, Asymptotic Normality)

**Breiman(2001)의 일반화 오차 상한**:
$$\mathbb{E}_{X,Y}[Y - m_{\infty,n}(X)]^2 \leq \bar{\rho} \, \mathbb{E}_{\Theta, X, Y}[Y - m_n(X; \Theta)]^2$$

$$\bar{\rho} = \frac{\mathbb{E}_{\Theta, \Theta'}[\rho(\Theta, \Theta') g(\Theta) g(\Theta')]}{\mathbb{E}_\Theta[g(\Theta)]^2}$$

$\rho(\Theta, \Theta') = \text{Corr}_{X,Y}(Y - m_n(X; \Theta), Y - m_n(X; \Theta'))$, $g(\Theta) = \sqrt{\mathbb{E}_{X,Y}[Y - m_n(X; \Theta)]^2}$.

**분산 분해** (Friedman et al. 2009):
$$\text{Var}[m_{\infty, n}(x)] = \rho(x) \sigma(x)$$

**유한-무한 포레스트 오차 차이** (Scornet 2015a):
$$0 \leq \mathbb{E}[m_{M,n}(X; \Theta_1, \ldots, \Theta_M) - m(X)]^2 - \mathbb{E}[m_{\infty,n}(X) - m(X)]^2 \leq \frac{8}{M} \times (\|m\|_\infty^2 + \sigma^2(1 + 4 \log n))$$

**점근 정규성** (Mentch & Hooker 2015): $a_n = o(\sqrt{n})$, $\lim_{n \to \infty} n / M_n = 0$이면
$$\frac{\sqrt{n}(m_{M,n}(x; \Theta_1, \ldots, \Theta_M) - \mathbb{E}[m_{\infty,n}(x)])}{\sqrt{a_n^2 \zeta_{1, a_n}}} \xrightarrow{D} \mathcal{N}$$

$\zeta_{1, a_n} = \text{Cov}(m_n(X_1, X_2, \ldots, X_{a_n}; \Theta), m_n(X_1, X_2', \ldots, X_{a_n}'; \Theta'))$.

**비일치성 반례** (Biau et al. 2008): 그림 4의 분포에서 $k$ 분할 횟수가 고정되고 $\texttt{mtry} = 1$이며 각 트리가 이론적 오차를 최소화하더라도, 중앙 체커보드(checkerboard) 영역이 결코 분할되지 않아 오차가 최소 $1/6$.

---

## 5. 변수 중요도 (Variable Importance)

### 5.1 변수 중요도 지표

**MDI (Mean Decrease Impurity, 평균 불순도 감소)** (Breiman 2003a):
$$\widehat{\text{MDI}}(X^{(j)}) = \frac{1}{M} \sum_{\ell=1}^{M} \sum_{\substack{t \in \mathcal{T}_\ell \\ j_{n,t}^* = j}} p_{n,t} L_{\text{reg},n}(j_{n,t}^*, z_{n,t}^*)$$

여기서 $p_{n,t}$는 노드 $t$에 떨어진 관측치 비율.

**MDA (Mean Decrease Accuracy, 평균 정확도 감소)** (Breiman 2001): OOB 표본에서 변수 $X^{(j)}$의 값을 무작위 순열하여 OOB 오차의 변화를 측정.

$$\widehat{\text{MDA}}(X^{(j)}) = \frac{1}{M} \sum_{\ell=1}^{M} \left[ R_n(m_n(\cdot; \Theta_\ell), \mathcal{D}_{\ell, n}^j) - R_n(m_n(\cdot; \Theta_\ell), \mathcal{D}_{\ell, n}) \right] \quad (6)$$

$$R_n(m_n(\cdot; \Theta_\ell), \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{i: (X_i, Y_i) \in \mathcal{D}} (Y_i - m_n(X_i; \Theta_\ell))^2 \quad (7)$$

**MDA의 모집단 버전(population version)**:
$$\text{MDA}^*(X^{(j)}) = \mathbb{E}[Y - m_n(X_j'; \Theta)]^2 - \mathbb{E}[Y - m_n(X; \Theta)]^2$$

여기서 $X_j' = (X^{(1)}, \ldots, X'^{(j)}, \ldots, X^{(p)})$이고 $X'^{(j)}$는 $X^{(j)}$의 독립 복제본.

### 5.2 이론적 결과

**완전 무작위·완전 성장 트리에서의 MDI 분해** (Louppe et al. 2013):
$$\text{MDI}^*(X^{(j)}) = \sum_{k=0}^{p-1} \frac{1}{\binom{p}{k}(p-k)} \sum_{B \in \mathcal{P}_k(V^{-j})} I(X^{(j)}; Y | B)$$

$V^{-j} = \{1, \ldots, j-1, j+1, \ldots, p\}$, $\mathcal{P}_k(V^{-j})$는 $V^{-j}$의 크기 $k$ 부분집합 모음, $I(X^{(j)}; Y | B)$는 조건부 상호정보(conditional mutual information).

전체 합:
$$\sum_{j=1}^{p} \text{MDI}^*(X^{(j)}) = I(X^{(1)}, \ldots, X^{(p)}; Y)$$

**무관 변수(irrelevant variable)**: $I(X^{(j)}; Y | B) = 0$이면 $X^{(j)}$가 $B$에 대해 무관. 무한 표본에서 무관 변수를 추가해도 다른 변수의 중요도는 변하지 않는다.

**상관 가우시안 모형에서의 MDA** (Gregorutti et al. 2013): $C = (1-c)I_p + c \mathbf{1} \mathbf{1}^\top$, $\text{Cov}(X^{(j)}, Y) = \tau_0$이면
$$\text{MDA}^*(X^{(j)}) = 2 \left( \frac{\tau_0}{1 - c + p c} \right)^2$$

상관변수가 많을수록 변수 중요도는 $p$의 제곱의 역수로 감소한다.

### 5.3 관련 연구

여러 실증 연구(Archer & Kimes 2008; Strobl et al. 2008 등)는 **상관변수가 MDA 성능을 떨어뜨림**을 지적한다. Strobl et al.(2008)은 MDA가 변수의 한계(marginal) 중요도를 평가할 뿐 조건부 효과를 고려하지 않는다고 지적한다. 해결책으로 재귀 변수 제거(Recursive Feature Elimination) 결합, 가설 검정 활용 등이 제안된다.

---

## 6. 확장 (Extensions)

- **가중치 포레스트(Weighted Forests)**: 트리별 가중치 부여 (Winham et al. 2013); Dynamic Random Forest (Bernard et al. 2012)
- **온라인 포레스트(Online Forests)**: 스트리밍 환경 (Saffari et al. 2009; Denil et al. 2013); **Mondrian Forests** (Lakshminarayanan et al. 2014)
- **생존 포레스트(Survival Forests)**: 우측 절단(right-censored) 데이터; **Random Survival Forests, RSF** (Ishwaran et al. 2008)
- **순위 포레스트(Ranking Forests)** (Clémençon et al. 2013): AUC 기준의 ROC 최적화
- **클러스터링 포레스트(Clustering Forests)**: Cluster Forests (Yan et al. 2013)
- **분위수 포레스트(Quantile Forests)** (Meinshausen 2006): 조건부 분포 전체 정보
- **결측치 처리(Missing Data)**: 근접도 행렬(proximity matrix) 활용
- **단일 클래스 데이터(Single Class Data)**: One Class Random Forests (Désir et al. 2013)
- **불균형 데이터(Unbalanced Data)**: 다수 클래스 다운샘플링 (Chen et al. 2004)

---

## 7. 결론과 전망 (Conclusion and Perspectives)

- 분할정복, 리샘플링, 결합, 무작위 특징 공간 탐색은 단순하지만 근본적인 아이디어로 새 알고리즘에 영향을 줄 수 있다.
- 트리 결합 모델은 표준 희소성·평활성 조건으로 특징지을 수 없는 복잡한 패턴을 추정하는 것으로 보이며, 이는 아직 수학적으로 규명되지 않았다.
- 잎 노드 식별자의 튜플은 트리 수에 지수적인 영역 분할을 만들 수 있어 **심층 신경망(deep network) 구조**와의 연결 가능성이 제기된다 (Welbl 2014).
- **튜닝 파라미터의 최적 선택**, 특히 사전 리샘플링 크기 $a_n$에 대해 부트스트랩($a_n = n$)을 지지하는 이론은 부재하다.
- 트리 깊이가 통계적 성능에 미치는 영향도 미해결 문제로 남아 있다.

---

## 핵심 용어 정리 (Glossary)

| 한국어 | 영어 |
|---|---|
| 배깅 | Bagging (Bootstrap Aggregating) |
| 서브샘플링 | Subsampling |
| 부트스트랩 | Bootstrap |
| 일치성 | Consistency |
| 셀 / 잎 | Cell / Leaf |
| 회귀함수 | Regression function |
| 베이즈 분류기 | Bayes classifier |
| 지니 불순도 | Gini impurity |
| OOB 오차 | Out-of-bag error |
| 순수 무작위 포레스트 | Purely Random Forest |
| 중심 포레스트 | Centered Forest |
| 계층 최근접 이웃 | Layered Nearest Neighbor (LNN) |
| 조건부 상호정보 | Conditional mutual information |
| 평균 불순도 감소 | Mean Decrease Impurity (MDI) |
| 평균 정확도 감소 | Mean Decrease Accuracy (MDA) |
| 끝-자르기 선호 | End-Cut Preference (ECP) |
