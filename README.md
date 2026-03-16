# Dense: Measuring Text Density Across Transformer Paradigms

> "같은 단어 수라도 훨씬 긴 내용을 표현할 수 있는 문장이 있다."

## What is "Text Density"?

Some sentences pack far more meaning into the same number of tokens than others.

```
# High density (14 tokens)
"혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다" — 야코프 부르크하르트

# Low density (~14 tokens)
"오늘 아침에 일어나서 밥을 먹고 학교에 갔다가 집에 돌아왔다"
```

The first sentence compresses at least 3 independent propositions, requires knowledge of political philosophy and revolution theory, and admits multiple interpretations. The second is a simple chronicle of events.

This project investigates: **when dense vs sparse text enters a transformer, what happens inside?**

---

## Research Question

> Do high-density sentences produce measurably different internal representations than low-density sentences of the same token length? And does this difference vary across transformer paradigms (Encoder / Decoder / Diffusion LM)?

---

## Novel Contributions

### 1. Paraphrase Expansion Ratio (PER)
A new density measurement: ask an LLM to "fully unpack" a sentence so a 10-year-old can understand it, then measure the token expansion ratio.

```
PER = expanded_tokens / original_tokens

"혁명이 숙적을 주인으로 만들지 않는다면 그것만으로도 행운이다"
  → 풀어쓴 버전 (60 tokens) / 원문 (14 tokens)
  → PER ≈ 4.3

"오늘 아침에 일어나서 밥을 먹고 학교에 갔다"
  → 풀어쓴 버전 (16 tokens) / 원문 (12 tokens)
  → PER ≈ 1.3
```

High PER = the sentence is semantically compressed. No existing NLP paper uses this as a density proxy.

### 2. Cross-Paradigm Density Observation
The same density stimulus is fed to three fundamentally different architectures, and their internal responses are compared:

| Paradigm | Model | How density might appear |
|---|---|---|
| **Encoder** | BERT / XLM-RoBERTa | Higher attention entropy, effective rank |
| **Decoder** | GPT-2 / LLaMA | Higher surprisal variance, layer deltas |
| **Diffusion** | MDLM / LLaDA | Slower crystallization, higher convergence area |

### 3. Diffusion Crystallization
In diffusion LMs, text is recovered from noise iteratively. Dense sentences require more denoising steps before tokens "crystallize" to their final values — offering a paradigm-unique density signal.

### 4. BD3-LM Spectrum
Using BD3-LMs (block_size controls the AR↔Diffusion interpolation), we observe how density response changes as the generation paradigm shifts continuously from autoregressive to full diffusion.

### 5. Bilingual Comparison
Korean (SOV, left-branching relative clauses) and English (SVO, right-branching) have structurally different density patterns. We compare how transformers handle density across these two typologically distinct languages.

---

## Hypotheses

**H1 — Context Retrieval:**
High-density sentences trigger more implicit context retrieval internally.
→ Attention entropy ↑, layer-wise delta ↑, effective rank ↑

**H2 — Effective Length:**
The internal processing profile of a dense N-token sentence resembles that of a sparse kN-token sentence.
→ k ≈ PER score (density multiplier)

**H3 — Crystallization (Diffusion):**
Dense sentences show later crystallization and larger convergence area in diffusion LMs.
→ mean_crystallization ↑, convergence_area ↑

**H4 — Paradigm Invariance:**
Density effects are observable across all three paradigms, but manifest differently.
→ Universal signal with architecture-dependent expression

---

## Internal Signals Measured

| Signal | Definition | Expected direction for high density |
|---|---|---|
| Attention Entropy | Shannon entropy of attention distribution | ↑ More distributed attention |
| Hidden State Norm | L2 norm of layer hidden states | ↑ More information encoded |
| Layer Delta | 1 − cosine_sim(h_l, h_{l-1}) | ↑ More change between layers |
| Effective Rank | SVD-based dimensionality of hidden states | ↑ Higher-dimensional representation |
| Mean Attention Distance | Attention-weighted token distance | ↑ Long-range dependencies |
| Crystallization Step (Diffusion) | Step at which token confidence > threshold | ↑ Later convergence |
| Convergence Area (Diffusion) | Area under entropy curve | ↑ More total uncertainty |

---

## Project Structure

```
dense/
├── data/
│   └── seed_sentences.json         # Seed sentence pairs (ko + en, high + low)
│
├── step0_per/
│   └── per_validation.py           # PER calculation & validation
│
├── step1_internal/
│   └── extract_signals.py          # Encoder + Decoder internal state extraction
│
├── step2_diffusion/
│   └── diffusion_density.py        # Diffusion LM denoising trajectory analysis
│
├── utils/                          # Shared utilities (TBD)
├── outputs/                        # Generated figures
│
├── experiment_plan.py              # Full experiment roadmap
└── docs/
    └── strategy.md                 # Research strategy & literature context
```

---

## Experimental Phases

| Phase | Goal | Status |
|---|---|---|
| **Step 0** | PER validation — confirm PER correlates with human density judgment | 🔄 In progress |
| **Step 1** | PoC — Encoder + Decoder internal signals on 50 sentence pairs | ⬜ Planned |
| **Step 2** | Diffusion LM crystallization analysis | ⬜ Planned |
| **Step 3** | Cross-paradigm synthesis + BD3-LM spectrum experiment | ⬜ Planned |
| **Step 4** | Causal intervention (activation patching, attention ablation) | ⬜ Optional |

---

## Models

| Paradigm | PoC | Full Scale |
|---|---|---|
| Encoder | `bert-base-multilingual-cased` | `xlm-roberta-large` |
| Decoder | `gpt2` | `meta-llama/Llama-3-8B` |
| Diffusion | `kuleshov-group/mdlm-owt` | `ML-GSAI/LLaDA-8B-Instruct` |
| Spectrum | `kuleshov-group/bd3lm-owt-block_size8` | — |

---

## Key References

- [Entropy-Lens: Information Signature of Transformer Computations (2025)](https://arxiv.org/abs/2502.16570)
- [Surprise! Uniform Information Density Isn't the Whole Story (2024)](https://arxiv.org/abs/2410.16062)
- [MDLM: Simplified and Improved Masked Diffusion (NeurIPS 2024)](https://arxiv.org/abs/2406.07524)
- [BD3-LMs: Broader and Deeper Discrete Diffusion (ICLR 2025)](https://github.com/kuleshov-group/bd3lms)
- [Diffusion LMs Know the Answer Before Decoding (2025)](https://arxiv.org/abs/2508.19982)
- [Comparing Human and LLM Sentence Processing Difficulties (2025)](https://arxiv.org/abs/2510.07141)
