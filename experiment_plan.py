#%% [markdown]
# # Text Density Across Three Paradigms
#
# ## Research Question
# 같은 토큰 수의 문장이라도 "밀도"가 다르면,
# 트랜스포머 내부에서 어떤 차이가 관측되는가?
# 그리고 이 차이는 모델의 처리 방식(Encoder/Decoder/Diffusion)에 따라 어떻게 달라지는가?
#
# ## Hypotheses
#
# **H1 (Context Retrieval):**
# 고밀도 문장은 내부적으로 더 많은 implicit context retrieval을 유발.
# → attention entropy ↑, layer delta ↑, effective rank ↑
#
# **H2 (Effective Length):**
# 고밀도 문장 N토큰의 내부 프로필 ≈ 저밀도 문장 kN토큰의 내부 프로필.
# → k = "밀도 배수" = PER과 상관?
#
# **H3 (Crystallization):**
# Diffusion LM에서 고밀도 문장은 denoising 결정화가 느림.
# → mean_crystallization ↑, convergence_area ↑
#
# **H4 (Paradigm Invariance):**
# 밀도 효과는 Encoder/Decoder/Diffusion 모두에서 관측되지만,
# 발현 양상이 다르다 (보편성 + 구조 의존성).
#
# ## Experimental Design
#
# ```
# Phase 0: PER Validation         ← 현재 단계
#   └── PER이 밀도의 유효한 proxy인지 검증
#   └── 시드 데이터 20쌍, 상관관계 분석
#
# Phase 1: Internal Signal PoC
#   └── BERT + GPT-2로 4가지 신호 추출
#   └── 고밀도 vs 저밀도 통계적 차이 검증
#   └── 50쌍 (한국어 25 + 영어 25)
#
# Phase 2: Diffusion LM
#   └── MDLM + BD3-LM으로 denoising 궤적 추출
#   └── 결정화 시점, entropy 수렴 비교
#   └── BD3-LM block_size 스펙트럼 실험
#
# Phase 3: Cross-Paradigm Synthesis
#   └── 3가지 패러다임의 밀도 반응 통합 비교
#   └── Effective Length 검증
#   └── 2x2 매트릭스 (density × surprisal) 교란 분리
#
# Phase 4 (optional): Causal Intervention
#   └── Activation patching
#   └── Attention ablation
# ```
#
# ## Models
#
# | Paradigm  | Small (PoC)            | Large (Full)            |
# |-----------|------------------------|-------------------------|
# | Encoder   | bert-base-multilingual | XLM-RoBERTa-large       |
# | Decoder   | gpt2                   | LLaMA-3-8B              |
# | Diffusion | MDLM (130M)           | LLaDA-8B / Dream-7B     |
# | Spectrum  | BD3-LM (130M)         | —                       |
#
# ## Languages
# Korean + English bilingual
# → 한국어 SOV + 좌분기 관계절 vs 영어 SVO + 우분기
# → 밀도가 언어 구조에 따라 다르게 발현되는가?
#
# ## Novel Contributions
# 1. **PER (Paraphrase Expansion Ratio)** — 새로운 밀도 측정법
# 2. **Diffusion crystallization** — 밀도를 denoising 난이도로 관측
# 3. **BD3-LM spectrum** — AR↔Diffusion 연속 스펙트럼에서의 밀도 반응
# 4. **Cross-paradigm comparison** — 같은 밀도를 3가지 처리 방식에서 관측
# 5. **Bilingual density** — 한국어/영어 밀도 발현 차이
