#%% [markdown]
# # Step 1 Extension: Surprisal / UID Analysis
#
# **Uniform Information Density (UID) Hypothesis** (Levy & Jaeger 2007):
# 인간은 단위 시간당 정보량을 균일하게 유지하도록 언어를 생성한다.
# → 고밀도 문장은 토큰당 정보량(surprisal)이 높지만 분산이 낮을 것.
#
# **이 분석의 목적:**
# Qwen3-0.6B의 Layer Delta가 High < Low 방향(decoder는 encoder와 반대)이었다.
# 가설: decoder는 고밀도 문장을 '더 예측하기 쉬운' 패턴으로 처리한다.
# → 격언/인용구 형태의 고밀도 문장은 학습 데이터에 많이 등장 → 낮은 surprisal?
# → 아니면 고밀도 = 높은 mean surprisal + 낮은 surprisal variance (UID)?
#
# **측정 지표:**
# - mean surprisal: 문장의 전반적 예측 난이도
# - surprisal std: UID 지표 (낮을수록 균일한 정보 배분)
# - surprisal CV (coeff. of variation): 정규화된 분산

#%%
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"
DECODER_MODEL = "Qwen/Qwen3-0.6B"

#%% [markdown]
# ## 1. Surprisal 계산

#%%
def compute_surprisal(text: str, model, tokenizer, device: str) -> dict:
    """
    Causal LM으로 토큰별 surprisal 계산.

    surprisal(t_i) = -log2 P(t_i | t_1, ..., t_{i-1})

    Insight: 이것이 Shannon 정보량 — 해당 토큰이 얼마나 '놀라운가'.
    UID는 이 값이 문장 전체에 걸쳐 균등하기를 예측한다.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # (1, seq_len)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: (1, seq_len, vocab_size)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # causal: position i predicts token i+1
    # so surprisal of token i = -log2 softmax(logits[i-1])[token_i]
    log_probs = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)

    # token i (i>=1)의 surprisal = -log2 P(token_i | context)
    # logits[i-1] → predicts token at position i
    token_ids = input_ids[0, 1:]         # target tokens (skip BOS): (seq_len-1,)
    pred_logprobs = log_probs[:-1, :]    # predictions: (seq_len-1, vocab)

    # gather: -log2 P(actual_token)
    surprisals = -pred_logprobs.gather(
        1, token_ids.unsqueeze(1)
    ).squeeze(1) / torch.log(torch.tensor(2.0))  # nats → bits

    surprisals_np = surprisals.cpu().float().numpy()

    tokens = tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())

    return {
        "tokens": tokens,
        "surprisals": surprisals_np,
        "mean": float(surprisals_np.mean()),
        "std": float(surprisals_np.std()),
        "cv": float(surprisals_np.std() / (surprisals_np.mean() + 1e-8)),
        "n_tokens": len(surprisals_np),
    }


#%% [markdown]
# ## 2. 전체 문장 분석

#%%
def analyze_surprisal_all(sentences: list[dict], device: str = DEVICE) -> list[dict]:
    """모든 문장에 대해 surprisal 계산 (Qwen3 1회 로드)"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {DECODER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL, trust_remote_code=True
        # SDPA OK here: we're NOT extracting attention weights, just logits
    ).to(device).eval()

    results = []
    for s in sentences:
        result = compute_surprisal(s["text"], model, tokenizer, device)
        result.update({
            "id": s["id"],
            "lang": s["lang"],
            "density": s["density"],
            "text": s["text"],
        })
        print(f"  [{s['id']}] mean={result['mean']:.2f} bits, std={result['std']:.2f}, cv={result['cv']:.3f}")
        results.append(result)

    return results


#%% [markdown]
# ## 3. 시각화

#%%
def plot_surprisal_comparison(results: list[dict]):
    """
    Fig 1: 언어별 × 밀도별 surprisal 3지표 box plot
    Fig 2: 대표 문장 surprisal curve (token-level trajectory)
    """
    langs = ["ko", "en"]
    lang_labels = {"ko": "Korean", "en": "English"}

    # ── Fig 1: Box plots (mean / std / cv) ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Surprisal Analysis — {DECODER_MODEL}\nUID Hypothesis: dense text should have lower surprisal variance",
        fontsize=11, fontweight="bold"
    )

    metrics = [
        ("mean", "Mean Surprisal (bits)", "Higher = harder to predict"),
        ("std",  "Surprisal Std (bits)",  "Lower = more uniform (UID)"),
        ("cv",   "Surprisal CV",          "Normalized variance (UID indicator)"),
    ]

    for row, lang in enumerate(langs):
        lang_results = [r for r in results if r["lang"] == lang]
        high = [r for r in lang_results if r["density"] == "high"]
        low  = [r for r in lang_results if r["density"] == "low"]

        for col, (metric, label, hint) in enumerate(metrics):
            ax = axes[row, col]
            h_vals = [r[metric] for r in high]
            l_vals = [r[metric] for r in low]

            bp = ax.boxplot([h_vals, l_vals], tick_labels=["High", "Low"],
                           patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor("#ffaaaa")
            bp["boxes"][1].set_facecolor("#aaaaff")

            # individual points
            ax.scatter([1]*len(h_vals), h_vals, color="red",  alpha=0.7, zorder=3, s=40)
            ax.scatter([2]*len(l_vals), l_vals, color="blue", alpha=0.7, zorder=3, s=40)

            # Mann-Whitney p
            if len(h_vals) >= 2 and len(l_vals) >= 2:
                _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"ns (p={p:.2f})"
                ax.text(0.98, 0.97, sig, transform=ax.transAxes,
                       ha="right", va="top", fontsize=10,
                       color="black" if "ns" in sig else "darkred")

            ax.set_title(f"{lang_labels[lang]} — {label}\n({hint})", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / "step1_surprisal_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")

    # ── Fig 2: Token-level surprisal curves (1 high + 1 low per lang) ──────
    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    fig.suptitle(
        "Token-level Surprisal Trajectories\n(UID predicts flatter curves for efficiently packed sentences)",
        fontsize=11, fontweight="bold"
    )

    for row, lang in enumerate(langs):
        lang_results = [r for r in results if r["lang"] == lang]
        high_ex = [r for r in lang_results if r["density"] == "high"][0]
        low_ex  = [r for r in lang_results if r["density"] == "low"][0]

        for col, (example, color, dlabel) in enumerate([
            (high_ex, "red",  "High density"),
            (low_ex,  "blue", "Low density"),
        ]):
            ax = axes[row, col]
            surp = example["surprisals"]
            toks = example["tokens"]
            x = np.arange(len(surp))

            ax.bar(x, surp, color=color, alpha=0.6)
            ax.axhline(surp.mean(), color=color, lw=1.5, ls="--",
                      label=f"mean={surp.mean():.1f}b")
            ax.fill_between(x,
                            surp.mean() - surp.std(),
                            surp.mean() + surp.std(),
                            alpha=0.1, color=color, label=f"±std={surp.std():.1f}b")

            # x-axis: token labels (every 2nd if long)
            step = max(1, len(toks) // 15)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(
                [t[:6] for t in toks[::step]],
                rotation=45, ha="right", fontsize=7
            )

            title_text = example["text"][:50] + "..."
            ax.set_title(
                f"{lang_labels[lang]} | {dlabel}\n\"{title_text}\"",
                fontsize=8
            )
            ax.set_ylabel("Surprisal (bits)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / "step1_surprisal_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


#%% [markdown]
# ## 4. 요약 통계

#%%
def print_surprisal_summary(results: list[dict]):
    print("\n" + "="*65)
    print(f"SURPRISAL SUMMARY — {DECODER_MODEL}")
    print("="*65)

    for lang in ["ko", "en"]:
        lang_r = [r for r in results if r["lang"] == lang]
        high = [r for r in lang_r if r["density"] == "high"]
        low  = [r for r in lang_r if r["density"] == "low"]

        print(f"\n[{lang.upper()}] n_high={len(high)}, n_low={len(low)}")
        for metric, label, _ in [
            ("mean", "Mean surprisal (bits)", ""),
            ("std",  "Surprisal std",         ""),
            ("cv",   "Surprisal CV",          ""),
        ]:
            h_vals = [r[metric] for r in high]
            l_vals = [r[metric] for r in low]
            h_mean = np.mean(h_vals)
            l_mean = np.mean(l_vals)
            direction = "High > Low" if h_mean > l_mean else "High < Low"
            _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {label:<25} {direction}  "
                  f"H={h_mean:.3f} L={l_mean:.3f}  p={p:.4f} {sig}")


#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    with open(DATA_DIR / "seed_sentences.json") as f:
        data = json.load(f)
    sentences = data["sentences"]
    print(f"Loaded {len(sentences)} sentences")

    results = analyze_surprisal_all(sentences, device=DEVICE)
    plot_surprisal_comparison(results)
    print_surprisal_summary(results)
