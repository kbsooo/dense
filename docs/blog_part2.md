# 밀도 높은 문장은 복원이 더 어렵다 — Diffusion 관점

> 이 글은 [Part 1: 트랜스포머는 글의 밀도를 느끼는가]의 후속편이다.

Part 1에서 우리는 트랜스포머의 forward pass를 관찰했다. Encoder(BERT)에서는 고밀도 문장의 Layer Delta가 더 크고, Decoder(Qwen3)에서는 오히려 작다는 것을 발견했다.

이번에는 관점을 바꾼다. **문장을 완전히 파괴한 다음, 처음부터 복원해본다.**

---

## Diffusion Language Model이란

이미지 생성 AI(Stable Diffusion, DALL-E)를 써봤다면 원리를 안다. 노이즈에서 시작해서 점점 깨끗한 이미지를 만들어간다. 텍스트에도 같은 걸 할 수 있다.

**Discrete Diffusion** (D3PM; Austin et al., 2021)은 이산 상태 공간에서의 확산 과정이다. Absorbing state 변형에서, forward process는 각 토큰을 독립적으로 absorbing state $[\text{MASK}]$로 전이시킨다:

$$q(x_t \mid x_0) = \text{Cat}\bigl(x_t;\; (1-\beta_t)\,\mathbf{e}_{x_0} + \beta_t\,\mathbf{e}_{[\text{MASK}]}\bigr)$$

여기서 $\beta_t$는 noise schedule, $\mathbf{e}$는 one-hot 벡터. $t \to T$이면 모든 토큰이 $[\text{MASK}]$가 된다.

Reverse process는 각 step에서 모델 $p_\theta(x_0 \mid x_t)$를 사용하여 마스킹된 위치의 원래 토큰을 예측한다:

$$p_\theta(x_{t-1} \mid x_t) = \sum_{x_0} q(x_{t-1} \mid x_t, x_0)\, p_\theta(x_0 \mid x_t)$$

이 과정에서 모델의 확신도(confidence)와 불확실성(entropy)을 매 스텝마다 기록하면, 문장이 "결정화"되는 궤적을 볼 수 있다.

**결정화 가설:** 밀도 높은 문장은 복원하기 더 어렵다. 더 많은 스텝 동안 불확실성이 높게 유지된다.

---

## BERT를 Diffusion 모델로 쓴다

이상적으로는 MDLM 같은 전용 Diffusion Language Model을 쓰고 싶었다. 하지만 MDLM 코드베이스는 `flash_attn`과 `triton` — NVIDIA GPU 전용 라이브러리 — 에 하드코딩되어 있어서 Apple Silicon에서 돌릴 수 없었다.

대안: **BERT를 discrete diffusion denoiser로 쓴다.**

이건 편법이 아니다. Absorbing state D3PM에서 reverse step의 핵심 연산은 $p_\theta(x_0 \mid x_t)$ — 마스킹된 토큰의 원래 값을 예측하는 것이다. 이것이 정확히 BERT의 Masked Language Modeling objective:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{i \in \mathcal{M}} \bigl[\log P_\theta(x_i \mid \mathbf{x}_{\backslash \mathcal{M}})\bigr]$$

여기서 $\mathcal{M}$은 마스킹된 위치의 집합. BERT의 [MASK] 예측이 곧 denoising step이다.

MDLM(Sahoo et al., 2024)도 이 구조다 — 학습된 continuous-time noise schedule $\beta(t)$를 쓰지만, 핵심 메커니즘은 같다.

그래서 klue/bert-base(한국어)와 bert-base-uncased(영어)를 iterative unmasking engine으로 사용했다.

---

## 결정화 궤적

문장 $s = (t_1, \ldots, t_S)$에 대해:

1. 모든 content 토큰을 $[\text{MASK}]$로 치환: $x^{(0)} = ([\text{MASK}], \ldots, [\text{MASK}])$
2. 매 step $k$에서, 모델이 각 위치 $i$의 확률 분포 $P_\theta(v \mid x^{(k)})$를 예측
3. 각 위치의 **confidence** (확신도)와 **entropy** (불확실성)를 기록:

$$c_i^{(k)} = \max_v P_\theta(v \mid x^{(k)}) \qquad H_i^{(k)} = -\sum_v P_\theta(v \mid x^{(k)}) \log_2 P_\theta(v \mid x^{(k)})$$

4. 아직 마스킹된 위치 중 $c_i^{(k)}$가 가장 높은 위치를 unmask: $x_i^{(k+1)} = \arg\max_v P_\theta(v \mid x^{(k)})$
5. 모든 토큰이 복원될 때까지 반복 ($k = 1, \ldots, K$ where $K = S$)

이렇게 하면 문장마다 "denoising heatmap"이 생긴다. X축은 토큰 위치 $i$, Y축은 denoising step $k$. 색은 entropy $H_i^{(k)}$ (밝은색 = 높은 불확실성, 어두운색 = 확정됨).

---

## 결과: 영어에서 가설이 맞았다

두 가지 지표를 측정했다:

**Mean Crystallization** — 토큰 $i$의 결정화 시점 $\kappa_i$는 해당 토큰이 unmask되는 step 번호다. 문장 수준의 결정화:

$$\bar{\kappa} = \frac{1}{S}\sum_{i=1}^{S} \frac{\kappa_i}{K}$$

$\bar{\kappa} \approx 0$이면 대부분의 토큰이 즉시 확정, $\bar{\kappa} \approx 1$이면 끝까지 불확실.

**Convergence Area** — 평균 entropy 곡선 아래 면적. 전체 denoising 과정의 총 불확실성:

$$A = \int_0^1 \bar{H}(t)\, dt, \qquad \bar{H}(t) = \frac{1}{S}\sum_{i=1}^{S} H_i^{(\lfloor tK \rfloor)}$$

$A$가 크면 전체 과정에서 불확실성이 오래 높게 유지됨 = 복원이 어려운 문장.

| 모델 | 지표 | 방향 | p값 |
|---|---|---|---|
| klue/bert-base (KO) | Mean crystallization | 차이 없음 | 1.00 ns |
| klue/bert-base (KO) | Convergence area | 차이 없음 | 0.67 ns |
| bert-base-uncased (EN) | Mean crystallization | 차이 없음 | 0.67 ns |
| **bert-base-uncased (EN)** | **Convergence area** | **High > Low** | **0.011 \*** |

**영어 고밀도 문장의 convergence area가 유의미하게 더 크다.**

모델이 고밀도 영어 문장을 복원할 때, 전체 과정에 걸쳐 더 많은 불확실성을 경험한다. 문장 안에 담긴 정보가 많으니, "이 자리에 뭐가 올까?"를 결정하기가 더 어려운 것이다.

---

## 한국어는 왜 안 보이는가

한국어에서는 두 지표 모두 유의미하지 않았다. Part 1에서 klue/bert-base는 forward pass의 Layer Delta에서 p=0.032*를 보여줬는데, diffusion 관점에서는 침묵.

몇 가지 가능한 이유:

**1. SOV 구조의 문제.** 한국어는 동사가 문장 끝에 온다. Iterative unmasking에서 모델은 "가장 확신하는 위치"부터 복원하는데, 한국어에서는 내용어(명사, 형용사)가 먼저 복원되고 조사/어미가 나중에 온다. 이 순서가 밀도와 무관할 수 있다.

**2. BERT의 masked LM head 차이.** klue/bert-base의 MLM head가 한국어 토큰 예측에서 밀도 민감도가 낮을 수 있다. Forward pass(내부 표현)에서는 차이가 보이지만, 출력층(토큰 예측 확률)에서는 안 보이는 것.

**3. 표본 크기.** 여전히 n=5. 효과가 있어도 검출 못 할 만큼 작을 수 있다.

---

## Part 1과 연결

Part 1의 Surprisal 분석과 Part 2의 Convergence Area가 같은 이야기를 하고 있다:

| 분석 | 관점 | 영어 결과 |
|---|---|---|
| Surprisal (Qwen3) | 토큰별 예측 난이도 | High > Low, p=0.008** |
| Convergence Area (BERT) | 전체 복원 난이도 | High > Low, p=0.011* |

**"예측이 어렵다" ≈ "복원이 어렵다."** Decoder가 다음 토큰을 예측하는 것과, Diffusion 모델이 마스킹된 토큰을 복원하는 것은 — 같은 난이도를 다른 각도에서 보는 것이다.

---

## 전체 실험을 돌아보며

세 가지 패러다임으로 텍스트 밀도를 관찰했다.

**Encoder (BERT, bidirectional):**
문장 전체를 동시에 보는 모델. 고밀도 문장에서 **Layer Delta가 더 크다** (p=0.032*).
해석: 압축된 의미를 펼치려면 레이어마다 더 많은 변환이 필요하다.

**Decoder (Qwen3, causal):**
토큰을 왼쪽에서 오른쪽으로 처리하는 모델. 고밀도 문장에서 **surprisal이 더 균일하다** (CV p=0.008**). Layer Delta는 오히려 작아지는 경향.
해석: 잘 쓰여진 밀도 높은 문장은 정보를 토큰들에 고르게 배분한다 (UID 가설).

**Diffusion (BERT as denoiser):**
전체를 파괴한 뒤 복원하는 모델. 고밀도 영어 문장에서 **총 불확실성이 더 높다** (p=0.011*).
해석: 밀도 높은 문장은 복원 과정에서 더 오래 "불확실한 상태"를 유지한다.

세 관점이 모두 같은 방향을 가리킨다: **밀도는 모델에게 "더 어렵다."**

하지만 "어렵다"의 의미가 패러다임마다 다르다:
- Encoder한테는 "더 많은 변환이 필요하다"
- Decoder한테는 "예측이 균일하게 어렵다"
- Diffusion한테는 "복원에 더 많은 불확실성이 남는다"

---

## 이 실험이 답하지 못한 것

**표본 크기.** n=5 per group. 유의미한 결과가 나온 것 자체가 다행이지만, 효과 크기의 신뢰구간은 넓다.

**밀도 vs 문체.** 고밀도 문장 = 격언, 저밀도 = 일상문. 밀도를 통제하면서 문체를 일치시킨 실험이 아직 없다. Probing에서 PCA(1)=100%가 이걸 보여줬다.

**진짜 Diffusion LM.** BERT iterative unmasking은 이론적으로 D3PM과 동치이지만, MDLM처럼 학습된 noise schedule을 쓴 모델과의 직접 비교는 CUDA GPU가 필요하다.

**인과관계.** "밀도가 높으니까 Layer Delta가 크다"인지, "어려운 문장이니까 Layer Delta가 크다"인지는 구분하지 못했다. 밀도와 난이도를 교차시킨 2x2 실험이 필요하다.

---

## 마지막 생각

사람도 밀도 높은 글을 한 번에 이해하지 못한다. 부르크하르트의 "혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다"를 읽으면, 잠시 멈추고, 몇 번 다시 읽고, 자기 경험에 비춰서 해석한다.

트랜스포머도 비슷한 일을 하는 걸까? Layer Delta가 큰 것은 "각 레이어에서 더 많이 생각한다"는 뜻이고, Surprisal CV가 낮은 것은 "문장 전체에 걸쳐 고르게 집중한다"는 뜻이다.

물론 모델이 "이해"하는 건 아니다. 하지만 밀도라는 속성이 모델의 내부 처리에 흔적을 남긴다는 건 확인했다. 그 흔적의 모양이 패러다임마다 다르다는 것도.

글에 밀도가 있다는 건 — 아마도 글쓴이가 그만큼 많은 생각을 압축했다는 뜻일 것이다. 그리고 그 압축을 풀려면, 읽는 쪽에서도 그만큼의 노력이 필요하다. 사람이든 모델이든.
