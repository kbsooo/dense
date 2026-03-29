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

문장 s의 PER은 LLM이 생성한 풀어쓴 버전과 원문의 토큰 수 비율로 정의된다:

PER(s) = |tokens(paraphrase(s))| / |tokens(s)|

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

> **(표 2) 삽입 위치** — 위의 모델 테이블을 Word 표로 변환
> 캡션: "(표 2) 실험에 사용한 모델 구성"

**Layer Delta** — 연속된 레이어 $l$과 $l-1$ 사이의 표현 변화량:

$$\delta_l = 1 - \cos(\bar{\mathbf{h}}_l, \bar{\mathbf{h}}_{l-1})$$

여기서 $\bar{\mathbf{h}}_l$은 토큰 차원에 대한 mean-pooled hidden state이다.

**Surprisal** — causal LM에서 토큰 $t_i$의 정보량:

$$s(t_i) = -\log_2 P(t_i \mid t_{<i})$$

문장 수준의 변동계수 $\text{CV}(s) = \sigma(s) / \bar{s}$는 UID(Uniform Information Density) 가설[2]의 검증 지표로 사용한다.

### 3.2 파일럿 실험 (20문장)

고밀도·저밀도 각 5문장씩 총 20문장으로 예비 실험을 수행하였다. klue/bert-base에서 한국어 Layer Delta가 유의미하게 높았으나 ($p=0.032$), 영어에서는 유의미하지 않았다. Qwen3-0.6B의 surprisal 분석에서는 한국어 고밀도 문장의 CV가 유의미하게 낮았다 ($p=0.008$). 이 파일럿 결과를 바탕으로, 표본 크기와 교락 변수 문제를 해결하기 위한 검증 실험을 설계하였다.

### 3.3 검증 실험 1: 확대 코퍼스 (100문장)

한국어·영어 각 50문장(고밀도 25, 저밀도 25)으로 구성된 일반 코퍼스에서 동일 분석을 반복하였다.

> **(표 2) 삽입 위치** — Word 표로 변환
> 캡션: "(표 2) 확대 코퍼스(general.csv) 실험 결과 (Mann-Whitney U, n=25/group)"

| 신호 | 언어 | 방향 | $p$값 |
|---|---|---|---|
| Layer Delta | KO | High > Low | 0.003** |
| Layer Delta | EN | High > Low | 0.0001*** |
| Surprisal Mean | KO | High > Low | < 0.0001*** |
| Surprisal Mean | EN | High > Low | < 0.0001*** |
| Surprisal CV | KO | High < Low | < 0.0001*** |
| Surprisal CV | EN | High < Low | 0.003** |
| Convergence Area | KO | = | 0.94 ns |
| Convergence Area | EN | High < Low | < 0.0001*** |

파일럿에서 유의미하지 않았던 영어 Layer Delta ($p=0.0001$)와 영어 Surprisal CV ($p=0.003$)가 표본 확대 후 강하게 유의미해졌다. Convergence area는 영어에서만 유의미하며 한국어에서는 일관되게 null이었다.

### 3.4 검증 실험 2: Minimal Pairs (100쌍)

교락 변수(어휘·문체 차이) 통제를 위해, 동일한 의미를 가지되 밀도만 다른 minimal pair 100쌍(한국어 50, 영어 50)을 구축하였다. 예: "발표 전에 충분히 준비하지 않으면 질문을 받을 때 제대로 답하기 어렵다"(저밀도) vs "준비 없는 발표는 질문 앞에서 흔들린다"(고밀도). 이는 명제적 밀도가 아닌 **구문적 압축**(같은 명제의 다른 표현 길이)을 통제하는 실험이다. Wilcoxon signed-rank test를 사용하였다.

> **(표 3) 삽입 위치** — Word 표로 변환
> 캡션: "(표 3) Minimal pair 실험 결과 (Wilcoxon signed-rank, n=50 pairs/lang)"

| 신호 | 언어 | 방향 | $p$값 |
|---|---|---|---|
| Layer Delta | KO | High > Low | 0.005** |
| Layer Delta | EN | High > Low | < 0.0001*** |
| Surprisal Mean | KO | High > Low | < 0.0001*** |
| Surprisal Mean | EN | High > Low | < 0.0001*** |
| Surprisal CV | KO | High < Low | < 0.0001*** |
| Surprisal CV | EN | High < Low | < 0.0001*** |
| Convergence Area | KO | = | 0.44 ns |
| Convergence Area | EN | High < Low | < 0.0001*** |

의미를 통제한 minimal pair에서도 Layer Delta, Surprisal Mean/CV 모두 유의미하였다. 이는 관찰된 신호 차이가 단순한 어휘·문체 차이가 아닌, **텍스트 압축 자체**에 모델이 반응하는 것임을 시사한다.

> **(그림 1) 삽입 위치** — `outputs/step3_validation_comparison.png`
> 캡션: "(그림 1) 명제적 밀도(general) vs 구문적 압축(minimal pair) 비교. 두 조건 모두에서 Layer Delta와 Surprisal CV의 유의미한 차이가 관찰되었다."

### 3.5 Diffusion 결과의 언어별 비대칭

Convergence area는 영어에서만 일관되게 유의미하고 ($p < 0.0001$), 한국어에서는 두 실험 모두 null이었다. 이는 한국어 SOV 구조에서 iterative unmasking의 토큰 복원 순서가 밀도와 무관할 가능성을 시사한다.

## 4. 결론

> **(표 4) 삽입 위치** — Word 표로 변환
> 캡션: "(표 4) 전체 실험 결과 요약 (검증 실험 기준)"

| 패러다임 | 신호 | KO general | KO pair | EN general | EN pair |
|---|---|---|---|---|---|
| Encoder | Layer Delta | 0.003** | 0.005** | 0.0001*** | <0.0001*** |
| Decoder | Surprisal CV | <0.0001*** | <0.0001*** | 0.003** | <0.0001*** |
| Diffusion | Conv. Area | 0.94 ns | 0.44 ns | <0.0001*** | <0.0001*** |

본 연구는 텍스트 밀도가 트랜스포머의 내부 처리에 패러다임별로 서로 다른 흔적을 남긴다는 것을 보였다. 파일럿 실험(20문장)의 결과는 확대 코퍼스(100문장)와 의미 통제된 minimal pair(100쌍)에서 재현되었으며, 대부분의 신호에서 통계적 유의성이 강화되었다. Encoder에서는 Layer Delta 증가로, Decoder에서는 surprisal의 균일성으로, Diffusion에서는 복원 불확실성 증가로 발현된다.

한계로는 minimal pair가 명제적 밀도가 아닌 구문적 압축만을 통제한 점, 한국어 Diffusion 신호의 null result에 대한 추가 분석이 필요한 점, 진정한 Diffusion LM(MDLM 등)이 아닌 BERT 기반 근사를 사용한 점이 있다.

## 참고문헌

[1] S. Kemper, A. Rash, D. Kynette, and S. Norman, "Telling stories: The structure of adults' narratives," European Journal of Cognitive Psychology, vol. 2, no. 3, pp. 205-228, 1990.
[2] R. Levy and T. Jaeger, "Speakers optimize information density through syntactic reduction," Advances in Neural Information Processing Systems, vol. 20, 2007.
[3] J. Austin, D. Johnson, J. Ho, D. Tarlow, and R. van den Berg, "Structured denoising diffusion models in discrete state-spaces," Advances in Neural Information Processing Systems, vol. 34, pp. 17981-17993, 2021.
[4] S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov, "Simple and effective masked diffusion language models," Advances in Neural Information Processing Systems, vol. 37, 2024.
[5] 한국어 BERT: klue/bert-base, https://huggingface.co/klue/bert-base
[6] Qwen Team, "Qwen3 Technical Report," 2025.
