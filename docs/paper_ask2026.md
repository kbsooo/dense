# ASK 2026 학부생 논문 — 원고 (양식 붙여넣기용)

> 양식: KIPS 학술발표대회 논문양식_v1_word.doc (A4 2~3p, 학부생)
> 제출 마감: 2026년 4월 6일(월) 자정

---

## 제목 (한글)

Paraphrase Expansion Ratio 기반 텍스트 밀도 측정 및 트랜스포머 패러다임별 내부 처리 차이 분석

## 제목 (영문)

Text Density Measurement via Paraphrase Expansion Ratio and Cross-Paradigm Analysis of Transformer Internal Processing

## 저자

(본인 정보로 교체)

홍길동
한국대학교 컴퓨터공학과 학부생

gildong@university.ac.kr

Gil-Dong Hong
Dept. of Computer Science, Han-Kook University

---

## 요약

텍스트 밀도(text density)는 동일한 토큰 수에 담긴 명제적 정보량의 차이를 의미하지만, 이를 정량화하는 표준적 방법은 아직 없다. 본 연구는 LLM 기반의 새로운 밀도 측정 지표인 PER(Paraphrase Expansion Ratio)을 제안하고, 6개 LLM을 통해 교차 검증한다. 또한 한국어·영어 20개 문장에 대해 Encoder(klue/bert-base), Decoder(Qwen3-0.6B), Diffusion(BERT iterative unmasking) 세 가지 트랜스포머 패러다임에서 밀도에 따른 내부 처리 차이를 분석한다. 실험 결과, Encoder에서는 고밀도 문장의 Layer Delta가 유의미하게 높고($p=0.032$), Decoder에서는 surprisal 변동계수가 유의미하게 낮으며($p=0.008$), Diffusion에서는 convergence area가 유의미하게 높았다($p=0.011$). 이는 동일한 밀도 속성이 패러다임에 따라 서로 다른 내부 신호로 발현됨을 보여준다.

---

## 1. 서론

자연어 텍스트에는 밀도(density)라는 속성이 존재한다. "혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다"(부르크하르트)와 "오늘 아침에 일찍 일어나서 따뜻한 물로 세수를 하고 밥을 먹었다"는 비슷한 토큰 수를 가지지만, 전자는 풀어쓰면 7배 이상 길어진다. 이 차이는 단순한 어휘 난이도가 아닌 명제적 압축도(propositional compression)에 기인한다.

기존의 텍스트 복잡도 측정법은 이 차이를 포착하지 못한다. 퍼플렉서티(perplexity)는 예측 난이도를 측정하지만 "어려운 단어"와 "압축된 의미"를 구분하지 못하며, 명제 밀도(propositional density)[1]는 동사·형용사 수를 세는 구조적 접근으로 암묵적 명제를 놓친다.

본 연구의 기여는 세 가지다. 첫째, LLM의 paraphrase 능력을 활용한 새로운 밀도 측정 지표 PER을 제안한다. 둘째, Encoder/Decoder/Diffusion 세 패러다임에서 밀도에 따른 내부 신호 차이를 체계적으로 비교한다. 셋째, 한국어·영어 이중 언어 환경에서 밀도 인식의 언어별 차이를 보인다.

## 2. PER: Paraphrase Expansion Ratio

### 2.1 정의

문장 $s$의 PER은 LLM이 생성한 풀어쓴 버전과 원문의 토큰 수 비율로 정의된다:

$$\text{PER}(s) = \frac{|\text{tokens}(\text{paraphrase}(s))|}{|\text{tokens}(s)|}$$

Paraphrase 프롬프트는 "모든 암묵적 전제와 함축을 명시적으로 풀어 초등학생이 이해할 수 있게 재작성하라"로 통일하였다.

### 2.2 교차 검증

PER의 측정 도구가 LLM이므로 모델 간 일관성 검증이 필수적이다. 6개 LLM(Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.4-pro, GPT-5.4-thinking, Gemini Flash, Gemini 3.1 Pro)에 대해 한국어 10문장, 영어 10문장의 PER을 산출하고 Spearman 순위 상관을 계산하였다.

GPT·Claude 4개 모델 간 상관은 $\rho=0.74\sim0.88$ ($p<0.001$)로 높은 합의를 보인 반면, Gemini 계열과의 상관은 $\rho=-0.14\sim0.19$ (n.s.)로 유의미하지 않았다. 이는 PER이 절대값이 아닌 상대적 순위로 사용되어야 함을 시사한다. 한국어 고밀도 문장이 가장 안정적으로 인식되었다 (CV $= 0.27\sim0.28$).

## 3. 트랜스포머 내부 신호 분석

### 3.1 실험 설계

고밀도·저밀도 각 5문장씩, 한국어와 영어 총 20문장을 사용하였다. 4개 모델을 언어별로 매칭하여 사용하였다:

| 모델 | 유형 | 담당 언어 | 파라미터 |
|---|---|---|---|
| klue/bert-base | Encoder | 한국어 | 110M |
| bert-base-uncased | Encoder | 영어 | 110M |
| gpt2 | Decoder | 영어 | 124M |
| Qwen/Qwen3-0.6B | Decoder | 한/영 | 600M |

5개 내부 신호를 측정하였다. 본 논문에서는 유의미한 결과를 보인 신호를 중심으로 보고한다.

**Layer Delta** — 연속된 레이어 $l$과 $l-1$ 사이의 표현 변화량:

$$\delta_l = 1 - \cos(\bar{\mathbf{h}}_l, \bar{\mathbf{h}}_{l-1})$$

여기서 $\bar{\mathbf{h}}_l$은 토큰 차원에 대한 mean-pooled hidden state이다.

**Surprisal** — causal LM에서 토큰 $t_i$의 정보량:

$$s(t_i) = -\log_2 P(t_i \mid t_{<i})$$

문장 수준의 변동계수 $\text{CV}(s) = \sigma(s) / \bar{s}$는 UID(Uniform Information Density) 가설[2]의 검증 지표로 사용한다.

### 3.2 Encoder 결과

klue/bert-base에서 한국어 문장의 Layer Delta가 고밀도 그룹에서 유의미하게 높았다 ($U$-test, $p=0.032$). 나머지 4개 신호(attention entropy, hidden state norm, effective rank, attention distance)는 유의미하지 않았다. 이는 밀도가 높은 문장일수록 레이어 간 표현 변환이 더 활발하게 일어남을 의미한다.

### 3.3 Decoder 결과

Qwen3-0.6B의 surprisal 분석에서 한국어 고밀도 문장의 평균 surprisal이 유의미하게 높았고 ($\bar{s}_H=6.08$ vs $\bar{s}_L=4.27$ bits, $p=0.032$), 변동계수는 유의미하게 낮았다 ($\text{CV}_H=0.68$ vs $\text{CV}_L=0.95$, $p=0.008$). 이는 고밀도 문장의 정보가 토큰 간에 더 균일하게 배분되어 있음을 나타내며, UID 가설을 부분적으로 지지한다.

Decoder의 Layer Delta는 Encoder와 반대 방향의 경향을 보였다 (High < Low, $p=0.076$). 이는 surprisal의 균일성으로 설명된다: 토큰별 정보량이 균일하면 레이어마다 비슷한 양의 상태 변화가 일어나 누적 Layer Delta가 오히려 낮아진다.

### 3.4 Diffusion 결과

Masked LM을 D3PM[3]의 absorbing-state discrete diffusion denoiser로 사용하여 iterative unmasking 궤적을 추출하였다. 모든 토큰을 [MASK]로 치환한 후, 모델의 confidence가 높은 위치부터 순차적으로 unmask하며, 매 step의 entropy 궤적을 기록하였다.

Convergence area (평균 entropy 곡선의 적분값)를 밀도 그룹 간 비교한 결과, 영어에서 고밀도 문장의 convergence area가 유의미하게 높았다 ($A_H=5.73$ vs $A_L=5.11$, $p=0.011$). 이는 고밀도 문장이 denoising 과정에서 더 높은 총 불확실성을 유지함을 의미한다.

### 3.5 표면 특성 통제

Probing classifier (logistic regression, LOO-CV)를 적용한 결과, 모든 레이어에서 100% 정확도를 보였으나, PCA 1개 주성분만으로 완벽 분리가 가능하였다. 이는 고밀도(격언)와 저밀도(일상문)의 어휘·문체 차이에 기인하며, Layer Delta나 Surprisal CV처럼 "처리 방식의 차이"를 측정하는 신호와는 구별되어야 한다.

## 4. 결론

(표 1) 패러다임별 유의미한 밀도 신호 요약

| 패러다임 | 신호 | 방향 | $p$값 | 언어 |
|---|---|---|---|---|
| Encoder | Layer Delta | High > Low | 0.032* | KO |
| Decoder | Surprisal CV | High < Low | 0.008** | KO |
| Decoder | Mean Surprisal | High > Low | 0.008** | EN |
| Diffusion | Convergence Area | High > Low | 0.011* | EN |

본 연구는 텍스트 밀도가 트랜스포머의 내부 처리에 패러다임별로 서로 다른 흔적을 남긴다는 것을 보였다. Encoder에서는 Layer Delta 증가로, Decoder에서는 surprisal의 균일성으로, Diffusion에서는 복원 불확실성 증가로 발현된다. 이 세 관점은 동일한 현상 — 밀도 높은 텍스트가 모델에게 "더 어렵다" — 을 서로 다른 각도에서 포착한다.

한계로는 그룹당 5문장의 소규모 표본, 밀도와 문체의 교락(confounding), 진정한 Diffusion LM(MDLM 등)이 아닌 BERT 기반 근사를 사용한 점이 있다. 향후 대규모 밀도 통제 코퍼스 구축과 실제 Diffusion LM을 사용한 검증이 필요하다.

## 참고문헌

[1] S. Kemper, A. Rash, D. Kynette, and S. Norman, "Telling stories: The structure of adults' narratives," European Journal of Cognitive Psychology, vol. 2, no. 3, pp. 205-228, 1990.
[2] R. Levy and T. Jaeger, "Speakers optimize information density through syntactic reduction," Advances in Neural Information Processing Systems, vol. 20, 2007.
[3] J. Austin, D. Johnson, J. Ho, D. Tarlow, and R. van den Berg, "Structured denoising diffusion models in discrete state-spaces," Advances in Neural Information Processing Systems, vol. 34, pp. 17981-17993, 2021.
[4] S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov, "Simple and effective masked diffusion language models," Advances in Neural Information Processing Systems, vol. 37, 2024.
[5] 한국어 BERT: klue/bert-base, https://huggingface.co/klue/bert-base
[6] Qwen Team, "Qwen3 Technical Report," 2025.
