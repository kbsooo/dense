# Research Strategy: Text Density Experiment

## 1. Motivation & Core Insight

글에는 "밀도"가 있다. 같은 토큰 수를 가져도 어떤 문장은 훨씬 많은 내용을 담고 있다. 이건 단순히 어려운 어휘의 문제가 아니다 — 구조적으로 더 많은 명제가 압축되어 있거나, 해석에 더 많은 외부 맥락이 필요하거나, 다의성을 품고 있다.

문제는: **NLP 모델은 이 차이를 "느끼는가"?** 트랜스포머가 고밀도 문장을 처리할 때 내부에서 실제로 다른 일이 일어나는가?

---

## 2. What Exists: Literature Landscape

### 관련 연구들이 각각 한 조각씩만 다루고 있음

| 연구 | 다루는 것 | 빠진 것 |
|---|---|---|
| **Surprisal Theory** | 토큰별 정보량 → 인간 읽기 시간 | 모델 내부 상태 X |
| **UID Hypothesis** | 문장 내 정보 분포 균일성 | 밀도 자체가 변수가 아님 |
| **Entropy-Lens** | 층별 entropy 궤적 | 입력 밀도와 연결 안 됨 |
| **CPIDR (P-density)** | 명제 수 자동 측정 | 영어 전용, 모델 내부 X |
| **Diffusion crystallization** | 어려운 텍스트 → 느린 denoising | 밀도를 통제 변수로 안 씀 |

### 핵심 빈 자리 (Our Gap)
> **입력 텍스트의 명제 밀도 → 트랜스포머 내부 층별 반응** 을 직접 연결한 논문이 없다.

특히 Diffusion LM에서 밀도를 결정화 난이도로 관측한 연구는 전무하다.

---

## 3. The Critical Confound: Density vs. Surprisal

자연 텍스트에서 고밀도 문장은 보통 surprisal도 높고 구문도 복잡하다. 이를 통제하지 않으면 "밀도 때문인지, 단순히 어려운 문장이어서인지" 구분 불가.

### 해결: 2×2 실험 설계

```
                   High Surprisal        Low Surprisal
                ┌────────────────────┬────────────────────┐
  High Density  │ 철학적 아포리즘     │ 친숙한 격언·속담    │
                │ (Heisenberg, 니체) │ ("뿌린 대로 거둔다")│
                ├────────────────────┼────────────────────┤
  Low Density   │ 생소한 전문 용어   │ 일상적 서술         │
                │ ("양자 비국소성")  │ ("오늘 밥 먹고 감") │
                └────────────────────┴────────────────────┘
```

- **Density 주효과**: 행 간 차이 (열 통제 후)
- **Surprisal 주효과**: 열 간 차이 (행 통제 후)
- **교호작용**: 두 변수가 독립적으로 내부 신호에 기여하는지

---

## 4. Density Operationalization

"밀도"를 측정 가능한 스칼라로 만드는 세 가지 방법:

### A. Paraphrase Expansion Ratio (PER) — Primary
```
PER = (풀어쓴 버전의 토큰 수) / (원문 토큰 수)
```
- 고밀도 문장: 풀어쓰면 많이 늘어남 → 높은 PER
- 저밀도 문장: 이미 풀어져 있음 → PER ≈ 1
- **Novel contribution**: 기존 문헌에 없는 측정법
- **재현성 검증 방법**: 같은 문장에 다른 LLM 3개로 PER 측정 → 분산이 작아야 유효

### B. Propositional Density (P-density) — Secondary
- 문장 내 동사/형용사/전치사/접속사 수를 단어 수로 나눈 값
- CPIDR 도구 (영어), 의존 구문 분석기 기반 (한국어)
- PER과의 상관관계로 cross-validation

### C. Surprisal Variance — Control Variable
- 토큰별 surprisal의 표준편차
- UID 위반 정도를 나타냄
- 밀도와 분리해서 교란 변수로 통제

---

## 5. Three-Paradigm Comparison Strategy

같은 밀도 자극을 세 가지 처리 방식에 넣어서 비교:

### Encoder (BERT)
- 전체 시퀀스를 동시에 양방향으로 처리
- 밀도를 "한 번에" 소화
- 측정: attention entropy, effective rank, layer delta

### Decoder (GPT-2/LLaMA)
- 왼쪽→오른쪽 순차 처리
- 밀도를 "점진적으로" 풀어냄
- 측정: 토큰별 surprisal, layer-wise representation 변화

### Diffusion LM (MDLM/LLaDA)
- 전체 시퀀스를 노이즈에서 동시에 복원
- 밀도를 "결정화 난이도"로 관측
- 측정: crystallization step, convergence area, entropy trajectory

### BD3-LM Spectrum
- block_size=1 (AR) → block_size=large (Diffusion)
- 같은 모델에서 처리 방식을 연속으로 변화
- "처리 방식의 어느 지점에서 밀도 반응이 바뀌는가?"

---

## 6. Internal Signals: Why These Four

### Attention Entropy
```
H = -Σ_j a_j · log₂(a_j)
```
고밀도 문장의 모든 토큰이 의미적으로 중요하다면, 어느 한 토큰에 attention이 집중되지 않을 것이다 → entropy ↑.

Entropy-Lens (2025) 결과: entropy 프로파일로 task 유형을 94% 정확도로 구분. 밀도도 구분 가능한가?

### Hidden State Effective Rank
```python
U, S, V = torch.svd(h)  # h: (seq_len, hidden_dim)
S_norm = S / S.sum()
eff_rank = exp(-Σ S_norm · log(S_norm))
```
더 많은 정보를 담은 표현 → 더 많은 차원을 실제로 활용 → 높은 effective rank.

### Layer Delta (Representation Change)
```
delta_l = 1 - cosine_sim(h_l, h_{l-1})
```
중간 층에서 표현이 많이 변한다 = 그 층에서 정보 변환이 활발하다 = 더 많은 "처리"가 일어남.

고밀도 문장은 단순 표면 패턴 matching이 아닌, 심층 의미 처리가 필요하므로 더 많은 층에서 큰 변화가 일어날 것.

### Crystallization Step (Diffusion 전용)
각 토큰이 confidence > 임계값에 처음 도달하는 denoising step.

"답을 알기 전에 먼저 전체 문맥을 구성해야 하는" 고밀도 문장 → 토큰들이 서로 의존적 → 늦은 결정화.

---

## 7. PER as Novel Contribution: Validation Protocol

PER을 contribution으로 밀기 위해 검증해야 할 세 가지:

1. **일관성 (Consistency):** 같은 문장에 다른 LLM으로 PER 측정 → Spearman ρ > 0.8
2. **수렴 타당성 (Convergent Validity):** PER ↔ P-density 상관 > 0.6
3. **판별 타당성 (Discriminant Validity):** PER이 토큰 수, 문장 길이와 독립적 (partial correlation 통제 후)

---

## 8. Bilingual Strategy

### 왜 한국어가 흥미로운가
- **SOV 어순**: 동사가 문장 끝에 → 디코더가 문장 전체를 읽기 전까지는 행위를 모름
- **좌분기 관계절**: 명사를 수식하는 절이 명사 앞에 옴 → 구조적으로 더 깊은 embed
- **조사 체계**: 격 표시가 어휘적으로 명시적 → 의존 관계가 surface에 드러남

### 크로스링구얼 비교 가설
영어의 고밀도 처리는 attention distance (먼 토큰 참조)가 핵심이지만, 한국어의 고밀도 처리는 layer depth (더 깊은 층에서 동사-논항 통합)가 핵심일 것.

→ 밀도의 "내부 서명(internal signature)"이 언어에 따라 다른가?

---

## 9. Risk Register

| 위험 | 가능성 | 완화 방법 |
|---|---|---|
| PER이 LLM 종류에 따라 크게 달라짐 | 중간 | 여러 LLM으로 측정 후 평균 + 분산 보고 |
| 고밀도/저밀도 쌍의 토큰 수 통제 실패 | 중간 | ±2 토큰 이내로 엄격히 매칭 |
| 밀도와 surprisal 교란 분리 실패 | 높음 | 2×2 설계 + 부분상관 분석 |
| 한국어 명제 밀도 측정 도구 부재 | 높음 | LLM-assisted annotation + inter-annotator agreement |
| Diffusion LM이 영어 전용 | 중간 | MDLM은 OWT(영어) 학습, LLaDA-8B는 다국어 — 한국어는 LLaDA 사용 |

---

## 10. Timeline

```
Week 1-2:  Step 0 — PER 검증
            ├── 시드 데이터 20쌍 PER 측정 (Claude API 사용)
            ├── P-density와 상관관계 분석
            └── 데이터셋 2×2 매트릭스로 확장 (50쌍)

Week 3-4:  Step 1 — PoC
            ├── BERT + GPT-2 내부 신호 추출
            └── 4가지 신호의 고밀도/저밀도 차이 통계 검증

Week 5-6:  Step 2 — Diffusion LM
            ├── MDLM 설치 + denoising 궤적 추출
            └── 결정화 패턴 분석

Week 7-8:  Step 3 — Synthesis
            ├── BD3-LM 스펙트럼 실험
            ├── 3 패러다임 비교
            └── Effective Length 검증

Week 9+:   Writing
```
