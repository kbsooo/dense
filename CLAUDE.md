# Dense: Text Density Analysis Across Transformer Paradigms

## Project Goal
Investigate how transformer models internally process "dense" vs "sparse" text —
sentences with high vs low propositional content per token.

## Three Model Paradigms
- **Encoder** (BERT/RoBERTa): bidirectional, simultaneous density processing
- **Decoder** (GPT-2/LLaMA): left-to-right, incremental density unfolding
- **Diffusion** (MDLM/BD3-LM): noise-to-text, density as crystallization difficulty

## Languages
Korean + English bilingual comparison

## Key Metric: Paraphrase Expansion Ratio (PER)
PER = (tokens in expanded paraphrase) / (tokens in original)
Novel density measurement — higher PER = denser text

## Code Convention
- All scripts use `#%%` cell markers (Jupytext style)
- Python 3.10+, PyTorch 2.x
- Shape assertions on every tensor operation
