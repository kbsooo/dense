# 블로그 글 재료 노트

> 이 파일은 블로그 초안 작성을 위한 원재료. 실험이 진행될수록 계속 추가할 것.
> 정제된 글이 아닌, 나중에 꺼내 쓸 수 있는 날 것의 관찰과 생각들.

---

## 글의 핵심 아이디어 (처음 떠올린 직관)

"글에는 밀도라는 게 있어. 같은 단어 수, 글자 수를 가지더라도 훨씬 긴 내용을 표현할 수 있거나, 여러 가지의 뜻을 내포할 수 있거든. 이건 단순한 어려운 어휘를 쓰거나 하는 것과는 달라."

이게 시작이었다. NLP 모델이 이 차이를 '느끼는지' 궁금했다.

---

## 동기 부여가 되는 예시 문장들 (본문에 쓸 것)

**고밀도 문장 예시:**
- "상부에서 자신들의 권위로 균형 잡힌 정의를 조성해주지 않는데도 다른 사람들의 권리를 자발적으로 존중하는 것은 그리 쉬운 일이 아니다" — 하이젠베르크, 『부분과 전체』
- "혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다" — 야코프 부르크하르트
- "자유란 책임을 의미한다. 그것이 대부분의 사람들이 자유를 두려워하는 이유다" — 버나드 쇼

**저밀도 문장 예시 (대조용):**
- "오늘 아침에 일찍 일어나서 따뜻한 물로 세수를 하고 밥을 먹은 다음에 학교에 갔다가 저녁에 집으로 돌아왔다"

→ 두 문장의 토큰 수는 비슷하다. 그런데 첫 번째는 풀어쓰면 4~7배가 된다.

---

## PER (Paraphrase Expansion Ratio) — 개념 설명용

**직관적 설명:**
"초등학생이 이해할 수 있게 풀어써" 라고 했을 때 토큰이 몇 배로 늘어나는지.
늘어나는 배수 = 그 문장이 얼마나 압축되어 있었는가.

**실제 예시 (GPT-5.4-thinking 기준):**
- 부르크하르트 "혁명이 숙적을..." (20토큰) → 풀어쓴 버전 (154토큰) → **PER = 7.70**
- "나는 어제 친구를 만나서 카페에서 커피를 마셨다" (23토큰) → 풀어쓴 버전 (39토큰) → **PER = 1.70**

4.5배 차이. 같은 토큰 수의 문장이 아니라, 내부에 담긴 정보량이 4.5배 다른 것.

**왜 기존 측정법과 다른가:**
- 퍼플렉서티(perplexity): 모델이 얼마나 놀라는가 → 어려운 어휘도 높게 나옴
- 명제 밀도(propositional density): 동사/형용사 수 카운트 → 구조적 측정
- PER: "이 문장을 완전히 이해하려면 얼마나 많은 말이 필요한가" → 의미론적 측정

---

## Step 0 실험 결과 — 흥미로운 발견들

### 발견 1: 모델 패밀리가 "밀도"를 다르게 본다

6개 모델(Claude Opus/Sonnet, GPT-5.4-pro/thinking, Gemini Flash/3.1 Pro)에게 같은 풀어쓰기 과제를 줬더니, 두 개의 독립적인 클러스터가 나타났다.

| 클러스터 | 모델 | 서로 간 Spearman ρ |
|---|---|---|
| **GPT+Claude** | Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.4-pro, GPT-5.4-thinking | 0.74~0.88 *** |
| **Gemini** | Gemini Flash, Gemini 3.1 Pro | 0.54* |
| **GPT+Claude vs Gemini** | — | -0.14~0.19 (ns) |

Gemini와 GPT/Claude는 **어떤 문장이 더 dense한지 서로 동의하지 않는다.**
이건 오류가 아니다 — 모델마다 "풀어쓰기"의 기준이 근본적으로 다른 것이다.

블로그 포인트: LLM 자체가 밀도 측정 도구인데, LLM마다 밀도를 다르게 정의한다. 이건 "밀도"가 객관적인가, 아니면 해석자에 따라 달라지는가 라는 더 깊은 질문을 제기한다.

### 발견 2: 한국어는 잘 작동, 영어는 문제

| 모델 | 한국어 High/Low ratio | 영어 High/Low ratio |
|---|---|---|
| gpt-5.4-thinking | **3.82x** | 1.59x |
| claude-sonnet-4.6 | **2.85x** | 1.04x |
| claude-opus-4.6 | 1.78x | **0.81x (역전!)** |

왜 영어가 문제인가? 우리가 "저밀도"라고 골랐던 영어 문장들이 생각보다 내용이 많았다.

예시: "This morning I took the bus to the library, returned three books that were due, checked out two new ones, and came back home before lunch."
- 겉으로 보면 단순한 일상 서술
- 하지만 "도서관", "반납 기한(due date)", "대출 절차" 같은 암묵적 컨텍스트가 숨어있음
- Claude Opus는 이걸 다 펼쳐서 썼다 → PER이 높아짐

→ 데이터 설계 문제. "저밀도"의 기준을 더 엄격하게 잡아야 한다.

### 발견 3: 가장 "밀도"가 일관되게 인식된 문장

모든 모델에서 공통적으로 높은 PER을 받고, 모델 간 분산도 낮은 문장:

1. KO_H02: "혁명이 숙적을 주인으로..." — mean PER=6.54, CV=0.28
2. KO_H03: "자유란 책임을 의미한다..." — mean PER=5.74, CV=0.28
3. KO_H01: "상부에서 자신들의 권위로..." — mean PER=3.28, CV=0.27

모두 한국어 고밀도 문장. 상위 5개가 전부 한국어. 이 문장들은 어떤 LLM을 써도 "이건 확실히 dense하다"고 느끼는 것.

### 발견 4: PER의 절대값보다 순위가 중요하다

Claude Sonnet은 전반적으로 높은 PER을 줬다 (한국어 High 평균 6.56). GPT-5.4-thinking은 낮은 PER (한국어 High 평균 6.23이지만 Low는 1.63). 절대값은 다르지만, 어떤 문장이 더 dense한지 순위는 비슷하다(ρ=0.57).

→ PER은 상대적 측정으로 써야 한다. "이 문장의 PER은 X다" 가 아니라 "이 문장은 저 문장보다 N배 더 dense하다."

---

## 실험 설계에서 배운 것 (방법론적 교훈)

1. **밀도 ≠ 어려움.** 어려운 전문 용어 문장이 반드시 dense한 게 아님. "양자역학적 비국소성 현상" — 어렵지만 명제 하나. dense하지 않다.

2. **밀도와 surprisal을 분리해야 한다.** 자연어에서 dense 문장은 보통 surprisal도 높다. 이걸 통제 안 하면 "밀도 때문인지, 어려운 단어 때문인지" 모른다. → 2×2 설계 필요.

3. **언어마다 밀도의 "패턴"이 다르다.** 한국어 SOV + 좌분기 관계절은 문장 끝에 동사가 오기 때문에, 끝까지 읽어야 전체 구조가 잡힌다. 이게 영어와 근본적으로 다른 처리를 요구할 수 있다.

4. **PER을 재현하려면 같은 모델 패밀리 내에서 비교해야 한다.** 다른 패밀리 간 PER 비교는 의미 없을 수 있다.

---

## Step 1 결과 — 언어-모델 매칭이 핵심이었다

### v1 (실패): mBERT + GPT-2

**세팅:** mBERT(encoder) + GPT-2(decoder), 한국어+영어 혼합
**결과:** 모든 신호 p > 0.05 (전부 null)

**원인 분석:**
1. **모델-언어 불일치** — GPT-2는 영어 전용, 한국어 문장을 byte-level 처리 → 의미 없음
2. **mBERT** — 한국어를 다루지만 밀도를 구분하도록 학습된 적 없음
3. 흥미로운 단서: mBERT Layer Delta에서 방향 High < Low (저밀도가 오히려 층간 변화가 더 큼) — 반직관적

### v2 (개선): 언어별 전담 모델 + Qwen3

**세팅:**
- `klue/bert-base` → KO 전담 encoder (한국어 사전학습)
- `bert-base-uncased` → EN 전담 encoder
- `gpt2` → EN decoder baseline
- `Qwen/Qwen3-0.6B` → KO+EN multilingual modern decoder

**결과:**

| 모델 | 신호 | 방향 | p값 |
|---|---|---|---|
| klue/bert-base (KO) | **Layer Delta** | **High > Low** | **0.032 \*** |
| bert-base-uncased (EN) | 전부 | - | ns |
| GPT-2 (EN) | 전부 | - | ns |
| Qwen3-0.6B (KO+EN) | Layer Delta | High < Low | 0.076 (marginal) |

### 핵심 발견

**방향이 뒤집혔다.**
- mBERT(v1): High < Low (저밀도가 층간 변화 더 큼) — 반직관적
- klue/bert-base(v2): **High > Low (고밀도가 층간 변화 더 큼)** — 직관에 부합

해석: 한국어 전용 모델을 쓰니 의미론적 압축을 실제로 감지한다.
고밀도 문장은 레이어를 거치며 더 많은 표현 변환이 필요 — 압축된 의미를 점진적으로 펼치는 것.

**Qwen3 marginal(p=0.076, High < Low):**
decoder(causal)는 encoder(bidirectional)와 반대 방향 경향.
→ 처리 패러다임(동시 처리 vs 순차 처리)이 밀도 반응 방식을 바꾼다는 단서.

### Step 1 Extension: Surprisal / UID 분석

**Qwen3-0.6B로 토큰별 surprisal (-log2 P) 계산:**

| 지표 | 언어 | 방향 | p값 |
|---|---|---|---|
| Mean surprisal | KO | High > Low | 0.032 * |
| Mean surprisal | EN | High > Low | 0.008 ** |
| Surprisal CV (변동계수) | KO | **High < Low** | **0.008 \*\*** |
| Surprisal CV | EN | High < Low | ns |

**해석:**
1. 고밀도 문장은 토큰당 평균 정보량이 더 높음 (KO: 6.08 vs 4.27 bits, EN: 6.10 vs 4.36 bits)
2. **한국어 고밀도 문장은 정보가 더 균일하게 배분됨** (CV: 0.68 vs 0.95, p=0.008**)
   → UID(Uniform Information Density) 가설을 부분 지지

**Qwen3 Layer Delta 역방향 설명:**
- 고밀도 문장 = 높은 mean surprisal + **낮은 CV** (균일한 surprisal)
- 균일한 surprisal → 각 토큰에서 모델의 상태 변화가 비슷 → 층간 누적 변화(Layer Delta)가 오히려 낮아짐
- 반면 저밀도 문장 = 낮은 mean surprisal + **높은 CV** (들쭉날쭉) → 특정 토큰에서 큰 변화 → Layer Delta 높아짐

### 블로그에 쓸 수 있는 서사
"encoder(BERT)와 decoder(Qwen3)는 같은 문장을 정반대 방향으로 느낀다.
BERT 계열: 고밀도 문장에서 레이어마다 표현이 더 많이 변한다 (내용을 펼치는 과정).
Qwen3: 고밀도 문장에서 레이어마다 표현이 오히려 안정적이다. 왜?
surprisal 분석이 답을 준다 — 고밀도 문장은 토큰당 정보량이 균일하게 분포되어 있다(UID).
균일한 surprisal = 예측이 고르게 어려움 = 어느 한 토큰에서 '놀라는' 일이 없음 = 층간 표현 변화가 고른 것."

### 기술적 메모
- transformers 5.x에서 SDPA가 기본값 → `output_attentions=True`는 `attn_implementation="eager"` 필요
- Qwen3-0.6B는 BFloat16 → SVD 전 `.float()` 캐스팅 필요
- Qwen3.5-0.8B는 멀티모달 (image-text-to-text) — 텍스트 전용 실험에 사용 불가

---

## 앞으로 나올 내용 (나중에 추가할 재료)

### Step 1: 트랜스포머 내부 관측 (예정)
- Attention entropy: dense 문장에서 attention이 더 분산되는가?
- Effective rank: dense 문장의 hidden state가 더 고차원적인가?
- Layer delta: dense 문장에서 층간 표현 변화가 더 큰가?
- 예상 그림: BERT/GPT-2에서 층별 신호 변화 곡선 (고밀도 vs 저밀도)

### Step 2: Diffusion LM (예정)
- "결정화 가설": dense 문장은 denoising 과정에서 더 늦게 "고체화"된다
- 측정: 각 토큰이 최종값으로 확정되는 step
- 예상 그림: denoising 과정의 heatmap (X=토큰위치, Y=step, 색=confidence)
- 사용 모델: MDLM (130M), BD3-LM

### Step 3: BD3-LM 스펙트럼 (예정)
- 같은 문장을 AR(block_size=1)에서 Diffusion(large block_size)까지 연속으로 처리
- 밀도 반응이 처리 방식에 따라 어떻게 달라지는가?
- "밀도는 모델 구조와 무관한 보편적 속성인가?"

---

## 블로그 구조 초안 (나중에 고칠 것)

```
제목 후보:
- "트랜스포머는 철학적 문장을 어떻게 처리하는가"
- "글의 밀도를 측정할 수 있을까?"
- "같은 20토큰, 다른 우주 — 텍스트 밀도 실험"

구조:
1. 도입: 하이젠베르크 문장 vs 일상 문장 대비
2. 문제 제기: 기존 NLP 측정법(perplexity, 문장 길이)의 한계
3. PER 아이디어: "풀어쓰면 몇 배?"
4. Step 0 결과: 6개 모델 실험
   - 한국어는 잘 작동한다
   - 모델 패밀리가 갈린다 (GPT+Claude vs Gemini)
   - 가장 dense한 문장은 무엇인가
5. Step 1 결과 (예정)
6. Step 2 Diffusion 결과 (예정)
7. 결론: 밀도는 보편적인가, 해석자에 따라 다른가?
```

---

## 기타 메모

- "사람도 밀도 높은 글을 이해하지 못할 수도 있음 — 이건 인생 또는 대화의 컨텍스트 때문?" → 이게 블로그의 마지막 질문으로 좋을 것 같다. 모델이 dense한 문장을 "이해" 못한다면, 그것도 사람이 컨텍스트 없이 이해 못하는 것과 같은 이유인가?
- GPT-5.4-thinking이 한국어 PER을 가장 잘 잡아낸다(ratio=3.82x) — thinking 과정이 암묵적 명제를 더 잘 펼치는 것과 관련있을 수 있음
- 나중에 Gemini가 왜 다른지 탐구하면 흥미로운 sub-story가 될 수 있음
