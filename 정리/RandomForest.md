# 랜덤 포레스트 안내 투어 (A Random Forest Guided Tour) — §2–§5 정리

> 원논문: Gérard Biau & Erwan Scornet, *A Random Forest Guided Tour*, arXiv:1511.05741v1 (2015).
>
> 본 문서는 §2 (랜덤 포레스트 추정량), §3 (단순화된 모형), §4 (Breiman 포레스트 이론), §5 (변수 중요도) 의 핵심 내용과 수식을 정리한 것이다.

---

## 목차

- §2. 랜덤 포레스트 추정량
  - 2.1 기본 원리
  - 2.2 알고리즘과 CART 분할 기준
  - 2.3 지도 분류
  - 2.4 매개변수 조정
- §3. 단순화된 모형과 국소 평균 추정량
  - 3.1 단순화된 모형 (Centered / Uniform / PURF)
  - 3.2 포레스트, 이웃, 커널
- §4. Breiman 포레스트에 대한 이론
  - 4.1 재표집 메커니즘
  - 4.2 결정 분할
  - 4.3 일관성, 점근 정규성
- §5. 변수 중요도
  - 5.1 변수 중요도 측도 (MDI, MDA)
  - 5.2 이론적 결과
  - 5.3 관련 연구

---

## §2. 랜덤 포레스트 추정량 (The Random Forest Estimate)

### 2.1 기본 원리 (Basic Principles)

**문제 설정 (비모수 회귀, nonparametric regression).** 입력 무작위 벡터 $X \in \mathcal{X} \subset \mathbb{R}^p$ 와 제곱적분가능한 반응 변수 $Y \in \mathbb{R}$ 의 쌍 $(X,Y)$ 가 주어지며, 회귀 함수

$$m(\mathbf{x}) = \mathbb{E}[Y \mid X = \mathbf{x}]$$

를 추정하는 것이 목표이다. 훈련 표본을 $\mathcal{D}_n = ((X_1,Y_1), \ldots, (X_n,Y_n))$ 라 하자. 추정량 $m_n$ 이

$$\mathbb{E}\bigl[m_n(X) - m(X)\bigr]^2 \xrightarrow{n \to \infty} 0$$

를 만족하면 **(평균 제곱) 일관(consistent)** 이라 한다.

**랜덤 포레스트의 정의.** 랜덤 포레스트는 $M$ 개의 무작위화된 회귀 트리들의 모음이다. $j$ 번째 트리의 질의점 $\mathbf{x}$ 에서의 예측은

$$m_n(\mathbf{x}; \Theta_j, \mathcal{D}_n) = \sum_{i \in \mathcal{D}_n^*(\Theta_j)} \frac{\mathbf{1}_{X_i \in A_n(\mathbf{x};\Theta_j,\mathcal{D}_n)} \, Y_i}{N_n(\mathbf{x};\Theta_j,\mathcal{D}_n)}$$

이며, 여기서

- $\Theta_1, \ldots, \Theta_M$ 은 $\mathcal{D}_n$ 과 독립이고 동일하게 분포된 무작위 변수 (재표집·분할 방향 결정에 사용)
- $\mathcal{D}_n^*(\Theta_j)$ 는 트리 구축 이전 사전 선택된 데이터점 집합
- $A_n(\mathbf{x}; \Theta_j, \mathcal{D}_n)$ 은 $\mathbf{x}$ 를 포함하는 셀(cell)
- $N_n(\mathbf{x}; \Theta_j, \mathcal{D}_n)$ 은 그 셀에 들어가는 사전 선택된 점의 수

**식 (1) — 유한 포레스트 (finite forest):**

$$m_{M,n}(\mathbf{x}; \Theta_1, \ldots, \Theta_M, \mathcal{D}_n) = \frac{1}{M} \sum_{j=1}^{M} m_n(\mathbf{x}; \Theta_j, \mathcal{D}_n)$$

R 패키지 `randomForest` 의 기본값은 `ntree` $= M = 500$ 이다.

**무한 포레스트 (infinite forest).** $M \to \infty$ 의 극한에서 큰 수의 법칙에 의해 거의 확실히

$$m_{\infty,n}(\mathbf{x}) = \mathbb{E}_{\Theta}\bigl[m_n(\mathbf{x}; \Theta, \mathcal{D}_n)\bigr] = \lim_{M \to \infty} m_{M,n}(\mathbf{x}; \Theta_1, \ldots, \Theta_M, \mathcal{D}_n)$$

가 성립하며, 이론적 분석에서는 이 극한 형태를 흔히 다룬다.

---

### 2.2 알고리즘과 CART 분할 기준 (Algorithm)

**세 가지 핵심 매개변수.**

| 매개변수 | 의미 | 회귀 기본값 | 분류 기본값 |
|---|---|---|---|
| $a_n \in \{1, \ldots, n\}$ | 각 트리에서 사전 표집되는 데이터점 수 | $n$ (부트스트랩) | $n$ |
| $\mathrm{mtry} \in \{1, \ldots, p\}$ | 각 노드에서 분할 후보 좌표 수 | $\lceil p/3 \rceil$ | $\lceil \sqrt{p} \, \rceil$ |
| $\mathrm{nodesize} \in \{1, \ldots, a_n\}$ | 분할 멈춤 기준 (셀 내 최소 점 수) | $5$ | $1$ |

**식 (2) — 회귀 CART 분할 기준.** 셀 $A$ 에서 분할은 쌍 $(j, z)$ 로 정의되며, 가능한 모든 분할 집합을 $\mathcal{C}_A$ 라 하자. 임의의 $(j,z) \in \mathcal{C}_A$ 에 대해 분할 전 노드 분산에서 분할 후 자식들의 분산을 뺀 값이 분할 기준이다:

$$L_{\text{reg},n}(j,z) = \frac{1}{N_n(A)} \sum_{i=1}^{n} (Y_i - \bar{Y}_A)^2 \, \mathbf{1}_{X_i \in A} - \frac{1}{N_n(A)} \sum_{i=1}^{n} \bigl(Y_i - \bar{Y}_{A_L} \mathbf{1}_{X_i^{(j)} < z} - \bar{Y}_{A_R} \mathbf{1}_{X_i^{(j)} \geq z}\bigr)^2 \, \mathbf{1}_{X_i \in A}$$

여기서 $A_L = \{ \mathbf{x} \in A : x^{(j)} < z \}$, $A_R = \{ \mathbf{x} \in A : x^{(j)} \geq z \}$ 이고 $\bar{Y}_A, \bar{Y}_{A_L}, \bar{Y}_{A_R}$ 는 해당 영역 내 $Y_i$ 의 평균이다. 직관적으로 (2)는 **분할 전후 경험적 분산의 (재정규화된) 차이** 이다.

**최적 분할.** 각 노드에서 균일하게 무작위 선택된 좌표 부분집합 $\mathcal{M}_{\text{try}} \subset \{1, \ldots, p\}$ ($|\mathcal{M}_{\text{try}}| = \mathrm{mtry}$) 위에서

$$(j_n^*, z_n^*) \in \arg\max_{j \in \mathcal{M}_{\text{try}}, \; (j,z) \in \mathcal{C}_A} L_{\text{reg},n}(j,z)$$

를 선택한다.

**알고리즘 1: $\mathbf{x}$ 에서의 Breiman 랜덤 포레스트 예측.**

```
입력: 훈련 집합 D_n, 트리 수 M, a_n, mtry, nodesize, 질의점 x.
출력: m_{M,n}(x; Theta_1,...,Theta_M, D_n).

for j = 1, ..., M do
    D_n 으로부터 (복원/비복원으로) a_n 개의 점을 균일 무작위 추출.
    P       <- (X);    P_final <- {}
    while P != {} do
        A <- P 의 첫 원소
        if (A 의 점 수 < nodesize) 또는 (A 의 모든 X_i 가 같음) then
            P 에서 A 제거;  P_final <- P_final + {A}
        else
            {1,..,p} 의 균일 무작위 부분집합 M_try (|M_try|=mtry) 선택
            식 (2) 를 M_try 좌표 위에서 최대화하여 최적 분할 (j*, z*) 결정
            A 를 (j*, z*) 로 자르면 자식 셀 A_L, A_R 생성
            P 에서 A 제거;  P <- P + {A_L, A_R}
        end if
    end while
    m_n(x; Theta_j, D_n) <- P_final 내 x 의 셀에 속하는 Y_i 의 평균
end for
return  (1/M) * sum_{j=1..M} m_n(x; Theta_j, D_n)         # 식 (1)
```

**Breiman 포레스트와 단일 CART의 본질적 차이.**

1. (2) 는 전체 $\{1, \ldots, p\}$ 가 아닌 무작위 부분집합 $\mathcal{M}_{\text{try}}$ 위에서 평가됨.
2. 개별 트리는 가지치기(pruning) 되지 않음 — 잎이 `nodesize` 미만이 될 때까지 자람.
3. 각 트리는 $a_n$ 개의 부분 표본 위에서만 구축됨 ($a_n = n$ 복원이면 부트스트랩, $a_n < n$ 이면 부분 표집).

---

### 2.3 지도 분류 (Supervised Classification)

이진 분류 ($Y \in \{0,1\}$) 에서 분류기 $m_n$ 의 일관성은

$$L(m_n) = \mathbb{P}[m_n(X) \neq Y] \xrightarrow{n \to \infty} L^*$$

이며, $L^*$ 는 베이즈 분류기

$$m^*(\mathbf{x}) = \begin{cases} 1, & \mathbb{P}[Y=1 \mid X=\mathbf{x}] > \mathbb{P}[Y=0 \mid X=\mathbf{x}] \\ 0, & \text{otherwise} \end{cases}$$

의 오차이다.

**랜덤 포레스트 분류기 (다수결, majority vote).**

$$m_{M,n}(\mathbf{x}; \Theta_1, \ldots, \Theta_M, \mathcal{D}_n) = \begin{cases} 1, & \dfrac{1}{M} \sum_{j=1}^{M} m_n(\mathbf{x}; \Theta_j, \mathcal{D}_n) > \dfrac{1}{2} \\ 0, & \text{otherwise} \end{cases}$$

**분류용 CART 분할 기준 (지니 불순도, Gini impurity 기반).** $p_{0,n}(A)$, $p_{1,n}(A)$ 를 셀 $A$ 내 레이블 0, 1 의 경험적 비율이라 할 때

$$L_{\text{class},n}(j,z) = p_{0,n}(A) \, p_{1,n}(A) - \frac{N_n(A_L)}{N_n(A)} \, p_{0,n}(A_L) \, p_{1,n}(A_L) - \frac{N_n(A_R)}{N_n(A)} \, p_{0,n}(A_R) \, p_{1,n}(A_R)$$

이 기준의 기반은 **지니 지수** $2 \, p_{0,n}(A) \, p_{1,n}(A)$ 이다.

> **참고.** $Y \in \{0,1\}$ 일 때 $L_{\text{class},n}$ 와 $L_{\text{reg},n}$ 의 최적화는 동등하므로 같은 트리가 얻어지나, 예측 전략이 다르다 (분류: 국소 다수결, 회귀: 국소 평균).

---

### 2.4 매개변수 조정 (Parameter Tuning)

**트리 수 $M$.** $M$ 이 커지면 분산은 감소하며, **과적합으로 이어지지 않는다**:

$$\lim_{n \to \infty} \mathbb{E}\bigl[m_{M,n}(X; \Theta_1, \ldots, \Theta_M) - m(X)\bigr]^2 = \mathbb{E}\bigl[m_{\infty,n}(X) - m(X)\bigr]^2$$

비용은 $M$ 에 선형이므로 정확도-계산량 절충에 따라 선택한다.

**mtry.** 성능에 큰 영향은 없으나, 너무 크면 예측 성능이 감소할 수 있음 (Díaz-Uriarte & de Andrés, 2006). Genuer et al. (2010) 은 기본값이 너무 작다고 주장하며 보수적으로 $\mathrm{mtry} = p$ 도 권한다.

**nodesize.** 분류 1, 회귀 5 가 기본값. 이론적 보장은 없으나 경험적으로 좋은 선택.

**자루 외부 오차 (Out-Of-Bag, OOB error).** 부트스트랩 표본에서 제외된 약 $1/3$ 의 관측치가 자연스러운 검증 집합 역할을 한다. 트리 별로 OOB 점에서 예측 오차를 계산하고 평균하면 별도의 검증 집합 없이 매개변수를 조정할 수 있다.

---

## §3. 단순화된 모형과 국소 평균 추정량 (Simplified Models)

### 3.1 단순화된 모형 (Centered / Uniform / PURF Forests)

이론 분석을 다루기 쉽도록 트리가 $\mathcal{D}_n$ 과 무관하게 설계되는 모형군을 **순수 랜덤 포레스트(purely random forests)** 라 하며, $\mathcal{X} = [0,1]^p$ 위에서 정의된다.

**중심 포레스트 (Centered Forest).**

1. 재표집 단계 없음.
2. 각 노드에서 $\{1, \ldots, p\}$ 중 한 좌표를 균일 무작위로 선택.
3. 선택된 좌표를 따라 **셀의 중심에서** 분할.
4. 위 (2)–(3) 을 $k$ 회 재귀 반복하여 정확히 $2^k$ 개의 잎을 만듦.

**균일 랜덤 포레스트 (Uniform Random Forest).** 위 (3) 만 "선택된 좌표 범위 위에서 **균일 무작위**" 분할로 바뀐 변종.

**일관성 (Breiman, 2004; Biau et al., 2008; Scornet, 2015a).** $k \to \infty$ 이고 $n / 2^k \to \infty$ 이면 중심 포레스트는 일관됨.

**수렴 속도 (희소성 시나리오, Breiman 2004; Biau 2012).** $X^{(j)}$ 들이 독립이고 회귀 함수가 부분집합 $\mathcal{S}$ 에만 의존, 즉 $m(\mathbf{x}) = \mathbb{E}[Y \mid X_{\mathcal{S}} = \mathbf{x}_{\mathcal{S}}]$ 이라 하자. 분할 확률 $p_{j,n} \to 1 / |\mathcal{S}|$ 이고 $m$ 이 립시츠 형이면

$$\mathbb{E}\bigl[m_{\infty,n}(X) - m(X)\bigr]^2 = O\!\left(n^{-\frac{0.75}{|\mathcal{S}| \log 2 + 0.75}}\right)$$

이 속도는 $|\mathcal{S}| \leq \lfloor 0.54 p \rfloor$ 인 한 통상의 비모수 속도 $n^{-2/(p+2)}$ 보다 **엄격하게 빠르다**. 즉, 랜덤 포레스트는 본질적 차원 $|\mathcal{S}|$ 에 적응한다.

**순수 균일 랜덤 포레스트 (PURF, Genuer 2012).** $p = 1$ 에 한정. $[0,1]$ 위 $k$ 개의 점을 균일 추출하여 부분 구간을 만든다. 립시츠 가정 하에

$$\mathbb{E}\bigl[m_{\infty,n}(X) - m(X)\bigr]^2 = O(n^{-2/3})$$

이며, 이 속도는 립시츠 클래스에 대해 **미니맥스 최적** (Stone, 1980).

---

### 3.2 포레스트, 이웃, 커널 (Forests, Neighbors and Kernels)

**계층화된 최근접 이웃 (Layered Nearest Neighbor, LNN).** 점 $X_i$ 가 $\mathbf{x}$ 와 함께 정의하는 초직사각형 안에 다른 데이터점이 없으면 $X_i$ 는 $\mathbf{x}$ 의 LNN.

**식 (3) — LNN-포레스트 연결 (Lin & Jeon, 2006).** 잎에 정확히 한 점이 남고 재표집이 없다면, $\mathbf{x}$ 에서의 포레스트 예측은 $\mathbf{x}$ 의 LNN 들의 가중 평균:

$$m_{\infty,n}(\mathbf{x}) = \sum_{i=1}^{n} W_{ni}(\mathbf{x}) \, Y_i, \qquad \sum_i W_{ni}(\mathbf{x}) = 1, \quad W_{ni} \geq 0$$

비적응형(non-adaptive) 포레스트에서 $X$ 가 $[0,1]^p$ 위 균일이면

$$\mathbb{E}\bigl[m_{\infty,n}(X) - m(X)\bigr]^2 = O\!\left(\frac{1}{n_{\max} (\log n)^{p-1}}\right)$$

여기서 $n_{\max}$ 는 말단 셀 최대 점 수이다.

**커널 표현 (kernel representation).** 재표집이 없는 유한 포레스트에 대해

$$m_{M,n}(\mathbf{x}; \Theta_1, \ldots, \Theta_M) = \sum_{i=1}^{n} W_{ni}(\mathbf{x}) \, Y_i$$

이고 가중치는

$$W_{ni}(\mathbf{x}) = \frac{1}{M} \sum_{j=1}^{M} \frac{\mathbf{1}_{X_i \in A_n(\mathbf{x}; \Theta_j)}}{N_n(\mathbf{x}; \Theta_j)}$$

이다.

**식 (4) — 무한 포레스트의 커널 표현.** $M \to \infty$ 의 극한에서 (무시할 수 있는 항을 제외하고)

$$m_{\infty,n}(\mathbf{x}) \approx \frac{\sum_{i=1}^{n} Y_i \, K_n(X_i, \mathbf{x})}{\sum_{j=1}^{n} K_n(X_j, \mathbf{x})}$$

여기서 커널은

$$K_n(\mathbf{x}, \mathbf{z}) = \mathbb{P}_{\Theta}\bigl[\mathbf{z} \in A_n(\mathbf{x}, \Theta)\bigr]$$

으로 정의되며, 이는 $\mathbf{x}$ 와 $\mathbf{z}$ 가 같은 셀에 떨어질 확률, 즉 포레스트 내 근접도이다.

> **주의.** $K_n$ 은 일반적으로 나다라야-왓슨(Nadaraya–Watson) 형 평행이동 불변 커널이 아니므로 분석은 모형마다 달라진다.

**중심 포레스트의 커널 (Scornet, 2015b).** 다항 분포 가중을 가진 닫힌 형태로 표현된다:

$$K_{n,k}(\mathbf{x}, \mathbf{z}) = \sum_{k_1 + \cdots + k_p = k} \frac{k!}{k_1! \cdots k_p!} \left(\frac{1}{p}\right)^{k} \prod_{j=1}^{p} \mathbf{1}_{\lceil 2^{k_j} x_j \rceil = \lceil 2^{k_j} z_j \rceil}$$

여기서 합은 $k_1, \ldots, k_p \geq 0$, $\sum_{j=1}^{p} k_j = k$ 인 모든 다중지수에 대해 이루어진다.

---

## §4. Breiman 포레스트에 대한 이론 (Theory for Breiman's Forests)

### 4.1 재표집 메커니즘 (The Resampling Mechanism)

**부트스트랩 (bootstrap).** Breiman 의 원래 알고리즘은 $n$ 개에서 $n$ 회 **복원 추출**한다. 단순하나 분석은 어렵다 — 부트스트랩 표본 $\mathcal{D}_n^*$ 의 분포는 원본 $\mathcal{D}_n$ 의 분포와 다르며, $X$ 가 밀도를 가져도 $\mathcal{D}_n^*$ 는 양의 확률로 중복된 점을 포함하므로 절대 연속일 수 없다.

**부분 표집 (subsampling) 으로의 우회.** 대부분의 이론 연구는 부트스트랩을 비복원 부분 표집으로 대체하고 $a_n / n \to 0$ 을 가정한다 (Mentch & Hooker 2015; Wager 2014; Scornet et al. 2015).

**중앙값 랜덤 포레스트 (Median Random Forest, Scornet 2015a).** 분할 위치를 셀 내 $X_i$ 의 **경험적 중앙값**으로 잡고 각 셀이 단 하나의 점을 가질 때까지 자란다. 개별 트리는 **비일관**(Györfi 등, 2002)이지만, $a_n / n \to 0$ 이면 **포레스트는 일관**된다.

> **직관.** $a_n / n \to 0$ 은 어떤 단일 쌍 $(X_i, Y_i)$ 가 $j$ 번째 트리에 사용될 확률을 작게 하여 트리 사이의 분할들이 충분히 비유사해지도록 한다.

**서브배깅된 1-NN (Biau & Devroye, 2010).** 1-NN 은 일반적으로 비일관이지만, $\mathcal{D}_n$ 에서 크기 $a_n$ 의 부분 표본을 추출하여 1-NN 추정한 뒤 무한히 평균한 추정량

$$r_n^*(\mathbf{x}) = \mathbb{E}^*\bigl[r_{a_n}(\mathbf{x})\bigr]$$

은, $a_n \to \infty$ 이고 $a_n / n \to 0$ 이면 **보편적으로** (즉 $(X,Y)$ 의 분포에 무관하게) 평균 제곱 일관됨. 이는 사실 가중치

$$W_{ni}(\mathbf{x}) = \mathbb{P}\bigl[X_i \text{ is 1-NN of } \mathbf{x} \text{ in a random selection of size } a_n\bigr]$$

을 가진 국소 평균 추정량이다.

---

### 4.2 결정 분할 (Decision Splits)

**식 (5) — Banerjee–McKeague (2007) 의 분할 위치 극한 법칙.** 회귀 모형 $Y = m(X) + \varepsilon$ ($X \in \mathbb{R}$, $\varepsilon$ 가우스 잡음) 에서 이론적 CART 최적 분할을 $d^*$, 양쪽 자식의 이론값을 $\beta_\ell^* = \mathbb{E}[Y \mid X \leq d^*]$, $\beta_r^* = \mathbb{E}[Y \mid X > d^*]$ 라 하고 그 경험적 추정을

$$(\hat{\beta}_{\ell,n}, \hat{\beta}_{r,n}, \hat{d}_n) \in \arg\min_{\beta_\ell, \beta_r, d} \sum_{i=1}^{n} \bigl(Y_i - \beta_\ell \mathbf{1}_{X_i \leq d} - \beta_r \mathbf{1}_{X_i > d}\bigr)^2$$

라 하자. 적당한 정칙 가정 ($X$ 의 밀도 $f$ 와 $m$ 이 연속 미분 가능) 하에서

$$n^{1/3} \begin{pmatrix} \hat{\beta}_{\ell,n} - \beta_\ell^* \\ \hat{\beta}_{r,n} - \beta_r^* \\ \hat{d}_n - d^* \end{pmatrix} \xrightarrow{\mathcal{D}} \begin{pmatrix} c_1 \\ c_2 \\ 1 \end{pmatrix} \arg\max_{t} \bigl(a W(t) - b t^2\bigr)$$

여기서 $W$ 는 표준 양측 브라운 운동, $a, b > 0$ 은 모형 의존 상수이다. 이로부터 CART 분할 점에 대한 **신뢰 구간 구성**이 가능하다.

**끝-분할 선호 (End-Cut Preference, ECP) (Ishwaran 2013).** 비정보 변수에 대한 분할은 셀의 가장자리 근처에서 일어나는 경향이 있다. 이는 $[0,1]^d$ 위에서 자식 셀의 표본 크기를 보존하는 **바람직한 성질**로 해석된다.

**희소성 적응 (Scornet et al., 2015).** 정칙 조건 하에서 $X$ 의 셀 구축에 사용된 분할 방향들 $j_{n,1}(X), \ldots, j_{n,k}(X)$ 이 확률 $1 - \xi$ 로 충분히 큰 $n$ 에 대해

$$j_{n,q}(X) \in \{1, \ldots, |\mathcal{S}|\}, \qquad 1 \leq q \leq k$$

즉 알고리즘이 점근적으로 **정보적 변수만 따라 분할**한다.

**CART 변종.**

- **Extra-Trees** (Geurts et al., 2006): 분할 점을 무작위 추출 후 CART 기준으로 선택.
- **PERT** (Cutler & Zhao, 2001): 무작위 분할 + 완벽 적합.
- **Oblique trees** (Breiman 2001): 특징의 선형 결합으로 분할.
- **Enriched RF** (Amaratunga et al., 2008), **RLT** (Zhu et al., 2015): 정보적 변수 가중 선택.
- **RRF / GRRF** (Deng & Runger, 2012, 2013): 분할 변수에 정규화 벌점.

---

### 4.3 일관성, 점근 정규성, 그리고 그 이상

**Breiman (2001) 의 상한.** 어떤 종류의 포레스트든

$$\mathbb{E}_{X,Y}\bigl[Y - m_{\infty,n}(X)\bigr]^2 \leq \bar{\rho} \, \mathbb{E}_{\Theta,X,Y}\bigl[Y - m_n(X; \Theta)\bigr]^2$$

여기서

$$\bar{\rho} = \frac{\mathbb{E}_{\Theta, \Theta'}\bigl[\rho(\Theta, \Theta') \, g(\Theta) \, g(\Theta')\bigr]}{\mathbb{E}_{\Theta}[g(\Theta)]^2}$$

이고

$$\rho(\Theta, \Theta') = \mathrm{Corr}_{X,Y}\bigl(Y - m_n(X; \Theta), \, Y - m_n(X; \Theta')\bigr), \qquad g(\Theta) = \sqrt{\mathbb{E}_{X,Y}[Y - m_n(X; \Theta)]^2}$$

이다. 즉 트리들의 **상관관계가 낮고 강도가 높을수록** 포레스트가 정확하다.

**Friedman et al. (2009) 의 분산 분해.**

$$\mathrm{Var}\bigl[m_{\infty,n}(\mathbf{x})\bigr] = \rho(\mathbf{x}) \, \sigma(\mathbf{x})$$

여기서 $\rho(\mathbf{x}) = \mathrm{Corr}\bigl[m_n(\mathbf{x}; \Theta), m_n(\mathbf{x}; \Theta')\bigr]$, $\sigma(\mathbf{x}) = \mathrm{Var}\bigl[m_n(\mathbf{x}; \Theta)\bigr]$.

**유한–무한 포레스트 오차 차이 (Scornet, 2015a).** 정칙 조건 하에

$$0 \leq \mathbb{E}\bigl[m_{M,n}(X; \Theta_1, \ldots, \Theta_M) - m(X)\bigr]^2 - \mathbb{E}\bigl[m_{\infty,n}(X) - m(X)\bigr]^2 \leq \frac{8}{M}\bigl(\|m\|_\infty^2 + \sigma^2 (1 + 4 \log n)\bigr)$$

이로부터 $M$ 을 충분히 크게 잡으면 유한 포레스트는 무한 포레스트와 임의로 가까워진다.

**점근 정규성 (Mentch & Hooker, 2015).** $a_n = o(\sqrt{n})$, $n / M_n \to 0$ 일 때 고정된 $\mathbf{x}$ 에 대해

$$\frac{\sqrt{n} \bigl(m_{M,n}(\mathbf{x}; \Theta_1, \ldots, \Theta_M) - \mathbb{E}[m_{\infty,n}(\mathbf{x})]\bigr)}{\sqrt{\dfrac{a_n^2}{n} \, \zeta_{1, a_n}}} \xrightarrow{\mathcal{D}} \mathcal{N}(0,1)$$

여기서

$$\zeta_{1, a_n} = \mathrm{Cov}\bigl(m_n(X_1, X_2, \ldots, X_{a_n}; \Theta), \; m_n(X_1, X_2', \ldots, X_{a_n}'; \Theta')\bigr)$$

이며 $X_i'$ 와 $\Theta'$ 는 독립 사본. Wager et al. (2014) 와 Mentch & Hooker (2015) 모두 $\zeta_{1, a_n}$ 을 위한 보정을 제공한다.

**가법 모형에서의 일관성 (Scornet et al., 2015).** Breiman 포레스트의 가지치기된 버전은 가법 회귀 모형에서 일관됨. 가지치기되지 않은 버전의 일관성은 CART 행동에 대한 추가 가정에 의존한다.

**Biau et al. (2008) 의 반례.** $\mathrm{mtry} = 1$, 분할 수 $k$ 고정인 환경에서, $X$ 가 세 정사각형 $[0,1]^2 \cup [1,2]^2 \cup [2,3]^2$ 위에 균일 분포하고, 좌하단 정사각형은 가산 무한 수직 띠 (값 $0/1$ 교대), 우상단은 수평 띠, 중간은 $2 \times 2$ 체커보드인 경우 — 어떤 무작위 분할 수열로도 중간 직사각형은 절대 잘리지 않으며, 따라서 분류기 오차 확률은 $\geq 1/6$ 으로 베이즈 오차 $L^* = 0$ 에 도달할 수 없다. 즉, 탐욕적으로 자란 RF 의 일관성은 미묘한 문제이다.

---

## §5. 변수 중요도 (Variable Importance)

### 5.1 변수 중요도 측도 (MDI, MDA)

랜덤 포레스트는 두 가지 측도로 변수의 중요도를 순위 매긴다.

#### 평균 불순도 감소 (Mean Decrease Impurity, MDI)

$$\widehat{\mathrm{MDI}}(X^{(j)}) = \frac{1}{M} \sum_{\ell=1}^{M} \sum_{t \in \mathcal{T}_\ell : \, j_{n,t}^* = j} p_{n,t} \, L_{\text{reg},n}(j_{n,t}^*, z_{n,t}^*)$$

여기서

- $\{\mathcal{T}_\ell\}_{1 \leq \ell \leq M}$: 포레스트 내 트리 모음
- $p_{n,t}$: 노드 $t$ 에 떨어지는 관측치의 비율
- $(j_{n,t}^*, z_{n,t}^*)$: 노드 $t$ 의 최적 분할

즉, **$X^{(j)}$ 를 따른 분할에 의한 가중 불순도 감소를 모든 트리에 대해 평균**한다. (분류에서는 $L_{\text{reg},n}$ 을 $L_{\text{class},n}$ 으로 대체.)

#### 평균 정확도 감소 (Mean Decrease Accuracy, MDA)

OOB 오차 추정량에 기반. 자루 외부 데이터셋 $\mathcal{D}_{\ell,n}$ 에서 변수 $X^{(j)}$ 의 값을 무작위 **치환(permute)** 한 데이터셋 $\mathcal{D}_{\ell,n}^j$ 를 만든 뒤, 치환 전후 오차 차이를 모든 트리에 대해 평균한다.

**식 (6):**

$$\widehat{\mathrm{MDA}}(X^{(j)}) = \frac{1}{M} \sum_{\ell=1}^{M} \Bigl[ R_n\bigl(m_n(\cdot; \Theta_\ell), \mathcal{D}_{\ell,n}^j\bigr) - R_n\bigl(m_n(\cdot; \Theta_\ell), \mathcal{D}_{\ell,n}\bigr) \Bigr]$$

**식 (7) — 오차 함수 정의:**

$$R_n\bigl(m_n(\cdot; \Theta_\ell), \mathcal{D}\bigr) = \frac{1}{|\mathcal{D}|} \sum_{i : (X_i, Y_i) \in \mathcal{D}} \bigl(Y_i - m_n(X_i; \Theta_\ell)\bigr)^2$$

**모집단 버전:**

$$\mathrm{MDA}^*(X^{(j)}) = \mathbb{E}\bigl[Y - m_n(X_j'; \Theta)\bigr]^2 - \mathbb{E}\bigl[Y - m_n(X; \Theta)\bigr]^2$$

여기서 $X_j' = (X^{(1)}, \ldots, X'^{(j)}, \ldots, X^{(p)})$ 이고 $X'^{(j)}$ 는 $X^{(j)}$ 의 독립 사본.

> **직관.** $X^{(j)}$ 가 무관하면 그 값을 치환해도 정확도가 떨어지지 않아야 한다.

---

### 5.2 이론적 결과

**Louppe et al. (2013) — 범주형 변수의 MDI 분해.** $X = (X^{(1)}, \ldots, X^{(p)})$ 이 유한 모달리티의 범주형이고, 각 트리는 부모 노드에 사용되지 않은 변수 중에서 균일 무작위 선택, 모달리티 수만큼의 자식으로 분할되며 완전히 자란다고 하자. 이때 무한 앙상블의 모집단 MDI 는

$$\mathrm{MDI}^*(X^{(j)}) = \sum_{k=0}^{p-1} \frac{1}{\binom{p}{k} (p - k)} \sum_{B \in \mathcal{P}_k(V^{-j})} I\bigl(X^{(j)}; \, Y \mid B\bigr)$$

여기서 $V^{-j} = \{1, \ldots, j-1, j+1, \ldots, p\}$, $\mathcal{P}_k(V^{-j})$ 는 카디널리티 $k$ 의 부분집합 모음, $I(X^{(j)}; Y \mid B)$ 는 조건부 상호 정보 (conditional mutual information) 이다.

**총합 항등식.**

$$\sum_{j=1}^{p} \mathrm{MDI}^*(X^{(j)}) = I\bigl(X^{(1)}, \ldots, X^{(p)}; \, Y\bigr)$$

즉, 출력 $Y$ 와 입력 전체의 상호 정보는 변수별 중요도의 합으로 분해된다.

**무관 변수 정의.** $X^{(j)}$ 가 모든 $B \subset V$ 에 대해 $I(X^{(j)}; Y \mid B) = 0$ 이면 **무관(irrelevant)** 이라 한다. 그러면 $\mathrm{MDI}^*(X^{(j)}) = 0$ 동치이며, 무관 변수를 추가해도 다른 변수의 중요도는 변하지 않는다.

**Ishwaran (2007) — MDA 의 잡음화 변형.** 치환 대신 트리 아래로 내려보낼 때 변수 $X^{(j)}$ 분할에서 $1/2$ 확률로 좌·우 자식 중 하나를 무작위 선택 ("특징 잡음화"). 회귀 함수가 조각별 상수일 때 표본 크기 $n \to \infty$ 의 점근 행동은 $X^{(j)}$ 를 따라 분할되는 부분 트리 집합과 밀접히 연관된다.

**Gregorutti et al. (2013) — 가우스 환경에서 MDA 의 닫힌 식.** 모형

$$Y = m(X) + \varepsilon, \qquad (X, \varepsilon) \text{ Gaussian}$$

이 다음 상관 구조

$$C = \bigl[\mathrm{Cov}(X^{(j)}, X^{(k)})\bigr]_{1 \leq j, k \leq p} = (1 - c) I_p + c \, \mathbf{1} \mathbf{1}^\top, \qquad c \in (0, 1)$$

와 모든 $j$ 에 대해 $\mathrm{Cov}(X^{(j)}, Y) = \tau_0$ 을 만족한다고 하자. 그러면 모든 $j$ 에 대해

$$\mathrm{MDA}^*(X^{(j)}) = 2 \left(\frac{\tau_0}{1 - c + p c}\right)^{2}$$

가우스 환경에서 상관된 변수의 수 $p$ 가 늘어나면 변수 중요도는 $p$ 의 제곱의 역수로 감소한다.

---

### 5.3 관련 연구

**상관된 변수의 부정 영향.** Archer & Kimes (2008), Strobl et al. (2008), Nicodemus & Malley (2009), Auret & Aldrich (2011), Tolosi & Lengauer (2011), Genuer et al. (2010) 등은 변수 사이 상관관계가 강할수록 MDA·MDI 모두 관련 변수 식별 능력이 떨어짐을 경험적으로 보였다.

**편향의 원인 (Strobl et al., 2008).** 알고리즘이 변수의 한계 중요도(marginal importance)를 평가하기 때문 (변수 사이 조건이 아님).

**개선 절차.**

- Gregorutti et al. (2013): RF + Guyon et al. (2002) 의 재귀적 특징 제거 (Recursive Feature Elimination).
- Mentch & Hooker (2015): 가설 검정 기반 관련 특징 감지.
- Mentch & Hooker (2014): 가법성(additivity) 등 회귀 함수 구조 감지에도 응용.

---

## 핵심 식 요약

| 식 번호 | 내용 |
|---|---|
| (1) | 유한 포레스트 추정량 $m_{M,n}(\mathbf{x}) = \frac{1}{M} \sum_j m_n(\mathbf{x}; \Theta_j, \mathcal{D}_n)$ |
| (2) | 회귀 CART 분할 기준 $L_{\text{reg},n}(j,z)$ — 분할 전후 경험적 분산 차이 |
| (3) | LNN 표현 $m_{\infty,n}(\mathbf{x}) = \sum_i W_{ni}(\mathbf{x}) Y_i$ |
| (4) | 무한 포레스트의 커널 표현 — 가중치는 $K_n(\mathbf{x}, \mathbf{z}) = \mathbb{P}_\Theta[\mathbf{z} \in A_n(\mathbf{x}, \Theta)]$ |
| (5) | Banerjee–McKeague 의 $n^{1/3}$ 분할 점 극한 법칙 |
| (6) | $\widehat{\mathrm{MDA}}(X^{(j)})$ 의 정의 (OOB 오차 차이의 트리 평균) |
| (7) | OOB 오차 함수 $R_n$ 의 정의 |

---

*요약 작성: 2026. 원논문은 모든 상세 증명과 실험 결과를 포함하므로, 정확한 진술은 원문을 참고할 것.*
