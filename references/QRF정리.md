\documentclass[11pt,a4paper]{article}
\usepackage{kotex}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}
\geometry{margin=2.5cm}

\newtheorem{theorem}{정리(Theorem)}
\newtheorem{lemma}{보조정리(Lemma)}
\newtheorem{assumption}{가정(Assumption)}

\title{분위 회귀 포레스트(Quantile Regression Forests) 핵심 정리}
\author{원논문: Meinshausen, N. (2006), \emph{Journal of Machine Learning Research}, 7, 983--999}
\date{}

\begin{document}
\maketitle

\section{개요(Introduction)}

표준 회귀 분석은 조건부 평균(conditional mean) $\mathbb{E}(Y \mid X = x)$의 추정량 $\hat{\mu}(x)$를 다음과 같이 얻는다:
\[
\mathbb{E}(Y \mid X = x) = \arg\min_z \mathbb{E}\bigl\{(Y - z)^2 \mid X = x\bigr\}.
\]
그러나 조건부 평균은 반응 변수(response variable)의 조건부 분포(conditional distribution)의 한 단면일 뿐이다. \textbf{분위 회귀(quantile regression)}는 분포 전체에 대한 정보를 제공한다. 조건부 분포 함수(conditional distribution function)는
\[
F(y \mid X = x) = \mathbb{P}(Y \le y \mid X = x),
\]
이고 $\alpha$-분위수($\alpha$-quantile) $Q_\alpha(x)$는
\begin{equation}
Q_\alpha(x) = \inf\{y : F(y \mid X = x) \ge \alpha\}.
\label{eq:quantile}
\end{equation}

\textbf{핵심 기여:} 랜덤 포레스트(random forests, Breiman 2001)는 조건부 평균뿐 아니라 \textbf{전체 조건부 분포}에 대한 정보를 담고 있다. 이를 활용하여 \textbf{분위 회귀 포레스트(quantile regression forests, QRF)}를 제안하고, \textbf{일치성(consistency)}을 증명한다.

\textbf{주요 응용:}
\begin{itemize}
    \item \textbf{예측 구간(prediction intervals):} 95\% 예측 구간은
    \begin{equation}
    I(x) = [Q_{.025}(x),\, Q_{.975}(x)].
    \label{eq:pi}
    \end{equation}
    구간 폭이 $x$에 따라 달라지므로 예측 신뢰도(reliability) 판단에 유용.
    \item \textbf{이상치 탐지(outlier detection):} 조건부 중앙값 절대 편차(conditional median absolute deviation) 또는 조건부 사분위 범위(conditional interquartile range)에 비춘 극단성 평가.
\end{itemize}

\section{분위 회귀의 손실 함수}

분위수는 \textbf{체크 손실(check loss / 가중 절대 편차)} $L_\alpha$를 최소화한다:
\begin{equation}
L_\alpha(y, q) = 
\begin{cases}
\alpha\, |y - q| & y > q, \\
(1 - \alpha)\, |y - q| & y \le q,
\end{cases}
\label{eq:loss}
\end{equation}
\[
Q_\alpha(x) = \arg\min_q \mathbb{E}\bigl\{L_\alpha(Y, q) \mid X = x\bigr\}.
\]
선형 분위 회귀는 볼록 최적화(convex optimization, Portnoy and Koenker 1997)로 해결되며, 비모수(non-parametric) 접근으로 분위 평활 스플라인(quantile smoothing splines)이나 분위 회귀 트리(quantile regression trees, Chaudhuri and Loh 2002)가 있다.

본 논문은 손실 함수 \eqref{eq:loss}를 직접 최소화하지 않고, \textbf{Lin and Jeon(2002)의 적응적 최근접 이웃(adaptive nearest neighbor) 관점}에서 랜덤 포레스트를 활용한다.

\section{랜덤 포레스트 표기와 가중치}

$n$개의 독립 표본 $(Y_i, X_i)$, $i = 1, \ldots, n$, 예측 변수 차원 $X : \Omega \to \mathcal{B} \subseteq \mathbb{R}^p$. $\theta$는 트리 성장 방식을 결정하는 무작위 모수 벡터(random parameter vector), $T(\theta)$는 해당 트리, $\ell(x, \theta)$는 $x$가 떨어지는 잎(leaf), $R_\ell \subseteq \mathcal{B}$는 잎 $\ell$의 직사각형 부분공간(rectangular subspace).

\textbf{단일 트리의 가중치}:
\begin{equation}
w_i(x, \theta) = \frac{\mathbf{1}\{X_i \in R_{\ell(x, \theta)}\}}{\#\{j : X_j \in R_{\ell(x, \theta)}\}}.
\label{eq:tree_weight}
\end{equation}
가중치 합은 1이며, 단일 트리 예측은 $\hat{\mu}(x) = \sum_{i=1}^{n} w_i(x, \theta) Y_i$.

\textbf{포레스트 가중치($k$개 트리의 평균):}
\begin{equation}
w_i(x) = k^{-1} \sum_{t=1}^{k} w_i(x, \theta_t).
\label{eq:forest_weight}
\end{equation}
랜덤 포레스트 예측: $\hat{\mu}(x) = \sum_{i=1}^{n} w_i(x) Y_i$.

\textbf{단일 조정 모수(tuning parameter):} \texttt{mtry} (각 노드에서 분할 후보로 고려할 변수 수). 결과는 광범위한 범위에서 거의 최적이며 OOB(out-of-bag) 예측으로 미세 조정 가능.

\section{분위 회귀 포레스트(QRF) 알고리즘}

핵심 통찰: $F(y \mid X = x) = \mathbb{E}(\mathbf{1}\{Y \le y\} \mid X = x)$이므로, 평균 추정과 동일한 가중치를 사용해 \textbf{지시 함수(indicator)}의 가중 평균을 취하면 분포 함수가 추정된다:
\begin{equation}
\hat{F}(y \mid X = x) = \sum_{i=1}^{n} w_i(x)\, \mathbf{1}\{Y_i \le y\}.
\label{eq:Fhat}
\end{equation}

\textbf{알고리즘:}
\begin{enumerate}
    \item[a)] 랜덤 포레스트와 동일하게 $k$개의 트리 $T(\theta_t)$, $t = 1, \ldots, k$를 성장시킨다. 단, \textbf{각 잎의 모든 관측치를 보존}(평균만 보존하는 RF와의 핵심 차이).
    \item[b)] 새 점 $X = x$를 모든 트리에 떨어뜨려 \eqref{eq:tree_weight}와 \eqref{eq:forest_weight}로 가중치 $w_i(x)$ 계산.
    \item[c)] 모든 $y \in \mathbb{R}$에 대해 \eqref{eq:Fhat}로 $\hat{F}(y \mid X = x)$ 계산.
\end{enumerate}
조건부 분위수 추정량 $\hat{Q}_\alpha(x)$는 \eqref{eq:Fhat}를 \eqref{eq:quantile}에 대입.

\textbf{RF와의 핵심 차이:} 랜덤 포레스트는 각 노드의 평균만 보존하지만, QRF는 \textbf{각 노드의 모든 관측치 값}을 보존하여 분포 추정에 사용한다.

R 패키지 \texttt{quantregForest}로 구현 (\texttt{randomForest}에 기반).

\section{일치성(Consistency)}

\subsection{가정(Assumptions)}

잎 $\ell$의 노드 크기(node-size): $k_\theta(\ell) = \#\{i : X_i \in R_{\ell(x, \theta)}\}$.

\begin{assumption}
$\mathcal{B} = [0, 1]^p$이고 $X$는 $[0, 1]^p$ 위의 균등 분포(uniform). (표기 편의용 가정.)
\end{assumption}

\begin{assumption}
$\max_{\ell, \theta} k_\theta(\ell) = o(n)$ (잎 비율 0으로 수렴) 이고 $1 / \min_{\ell, \theta} k_\theta(\ell) = o(1)$ (잎 최소 크기 발산).
\end{assumption}

\begin{assumption}
각 노드에서 변수 $m = 1, \ldots, p$가 분할 변수로 선택될 확률은 양의 상수로 유계(bounded from below). 또한 분할 시 각 자식 노드는 부모 노드 관측치의 비율 $\gamma$ 이상($0 < \gamma \le 0.5$)을 포함.
\end{assumption}

\begin{assumption}[Lipschitz 연속성]
어떤 상수 $L$에 대해, 모든 $x, x' \in \mathcal{B}$에서
\[
\sup_y |F(y \mid X = x) - F(y \mid X = x')| \le L \|x - x'\|_1.
\]
\end{assumption}

\begin{assumption}
모든 $x \in \mathcal{B}$에서 $F(y \mid X = x)$는 $y$에 대해 \textbf{엄격하게 단조 증가(strictly monotonously increasing)}.
\end{assumption}

\subsection{주요 정리}

\begin{theorem}
가정 1--5 하에서, 모든 $x \in \mathcal{B}$에 대해 점별로(pointwise)
\[
\sup_{y \in \mathbb{R}} |\hat{F}(y \mid X = x) - F(y \mid X = x)| \to_p 0, \quad n \to \infty.
\]
\end{theorem}

\subsection{증명 핵심 흐름(Proof Sketch)}

확률 적분 변환(probability integral transform): $U_i = F(Y_i \mid X = X_i)$로 정의하면 $U_i$는 i.i.d. $\mathrm{Uniform}[0, 1]$. 가정 5에 의해 $\{Y_i \le y\}$와 $\{U_i \le F(y \mid X_i)\}$는 동치.

추정 오차를 두 항으로 분해:
\[
|F(y|x) - \hat{F}(y|x)| \le \underbrace{\Bigl| F(y|x) - \sum_{i=1}^{n} w_i(x) \mathbf{1}\{U_i \le F(y|x)\} \Bigr|}_{\text{분산형 항(variance-type)}} + \underbrace{\Bigl| \sum_{i=1}^{n} w_i(x)\bigl(\mathbf{1}\{U_i \le F(y|X_i)\} - \mathbf{1}\{U_i \le F(y|x)\}\bigr) \Bigr|}_{\text{분포 변화 항(bias-type)}}.
\]

\textbf{(i) 분산형 항:} $0 \le w_i(x) \le (\min_{\ell, \theta} k_\theta(\ell))^{-1}$, $\sum w_i(x) = 1$, 가정 2로부터
\begin{equation}
\sum_{i=1}^{n} w_i(x)^2 \to 0, \quad n \to \infty,
\label{eq:weight_squared}
\end{equation}
따라서 $\sup_{z \in [0,1]} |z - \sum w_i(x) \mathbf{1}\{U_i \le z\}| = o_p(1)$ (Bonferroni 부등식 + Kolmogorov-Smirnov형 논변).

\textbf{(ii) 분포 변화 항:} 가정 4 (Lipschitz)에 의해
\[
\sum_{i=1}^{n} w_i(x)\, \|x - X_i\|_1 = o_p(1)
\]
임을 보이면 충분. 이는 보조정리 2로 환원된다.

\begin{lemma}
정리 1의 조건 하에서, 모든 $x \in \mathcal{B}$에 대해
\[
\max_m |I(x, m, \theta)| = o_p(1), \quad n \to \infty,
\]
여기서 $I(x, m, \theta) \subseteq [0, 1]$은 $\ell(x, \theta)$의 $m$번째 좌표 구간이다.
\end{lemma}

\textbf{보조정리 2 증명 요지:} $S(x, m, \theta)$를 $x$의 경로에서 변수 $m$에 대한 분할 횟수, $S_{\min}(\theta) = \min_{x \in \mathcal{B}} \sum_m S(x, m, \theta)$. 가정 3에 의해 $\max_\ell k_\theta(\ell) \ge n \gamma^{S_{\min}(\theta)}$이고, 가정 2에 의해 $\max_\ell k_\theta(\ell) = o(n)$이므로 $\gamma^{S_{\min}(\theta)} = o(1)$, 즉 $S_{\min}(\theta) \to \infty$. 변수 선택 확률이 양의 상수로 유계이므로 $\min_m S(x, m, \theta) \to_p \infty$. 따라서
\[
n^{-1} \#\{i : X_{i,m} \in I(x, m, \theta)\} \le (1 - \gamma)^{S(x, m, \theta)} \to_p 0,
\]
$X$의 균등성으로부터 Kolmogorov-Smirnov형 논변에 의해 $\max_m |I(x, m, \theta)| = o_p(1)$. \hfill$\square$

\textbf{한계점:} 신호/잡음 변수 구분 부재, 수렴 속도(convergence rate) 미분석, 정리 1이 트리 수 $k$에 무관(앙상블 안정화 효과 미반영).

\section{수치 실험(Numerical Examples)}

\subsection{설정}

\textbf{비교 방법:}
\begin{itemize}
    \item \textbf{QRF}: 트리 수 $k = 1000$, \texttt{mtry} = 변수 수의 1/3 (기본값), 잎당 최소 10개 관측치, 배깅(bagging) 사용.
    \item \textbf{LQR}: 상호작용 없는 선형 분위 회귀.
    \item \textbf{QQR}: 5-fold 교차검증(cross-validation)으로 전진 선택(forward selection)된 상호작용 항 추가.
    \item \textbf{TRC, TRM, TRP}: 분위 회귀 트리 (조각별 상수/다중 선형/2차 다항, GUIDE 패키지).
\end{itemize}

\textbf{데이터셋:} Boston Housing ($p=13, n=506$), Ozone ($p=12, n=366$), Abalone ($p=8, n=500$), BigMac ($p=9, n=69$), Fuel ($p=5, n=51$).

\textbf{평가:} 5-fold 교차검증으로 손실 \eqref{eq:loss}를 $\alpha \in \{.005, .025, .05, .5, .95, .975, .995\}$에서 평가. 부트스트랩(bootstrap) 95\% 신뢰구간으로 QRF 대비 평균 손실 차이의 유의성 판정.

\subsection{주요 결과}
\begin{itemize}
    \item \textbf{어떤 데이터셋에서도} 다른 방법이 QRF보다 \textbf{유의하게} 더 우수하지 않음. QRF가 유의하게 더 우수한 경우는 자주 관찰됨.
    \item QRF는 단일 트리 집합으로 모든 분위를 추정하므로 \textbf{단조성(monotonicity)} 보장: $\alpha \ge \beta \Rightarrow \hat{Q}_\alpha(x) \ge \hat{Q}_\beta(x)$. 다른 방법들은 이를 보장하지 않음.
    \item 상호작용 포함 선형 분위 회귀(QQR)는 중간 분위($\alpha$가 0이나 1에서 멀 때)에서 잘 작동하나, 극단 분위에서는 QRF가 더 우수.
    \item \textbf{잡음 변수 추가에 강건(robust)}: 각 예측 변수를 무작위 순열한 사본을 추가했을 때도 QRF 성능 유지.
\end{itemize}

\subsection{예측 구간 결과}
\begin{itemize}
    \item Boston Housing: 95\% 예측 구간이 506개 관측치 중 \textbf{10개를 제외하고 모두 포함} (목표 피복률 95\% 부합).
    \item BigMac, Fuel, Ozone에서는 예측 구간 폭이 관측치마다 \textbf{크게 변동}. 일부 새 관측치는 다른 관측치보다 훨씬 정확하게 예측 가능.
\end{itemize}

\section{결론(Conclusions)}

분위 회귀 포레스트는:
\begin{itemize}
    \item 반응 변수의 \textbf{전체 조건부 분포}를 추론하는 비모수 방법.
    \item 합리적 가정 하에 \textbf{일치성(consistency)} 보장.
    \item 예측 구간 구축 및 이상치 탐지에 활용 가능.
    \item 머신 러닝 벤치마크에서 선형/트리 기반 방법 대비 \textbf{경쟁력 있는 성능}.
    \item 예측 구간 폭으로 \textbf{예측 신뢰도}를 정량화: 폭이 좁으면 신뢰할 만한 예측, 넓으면 부정확.
\end{itemize}

\end{document}
