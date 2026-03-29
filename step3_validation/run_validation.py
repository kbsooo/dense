#%% [markdown]
# # Step 3: Validation Experiments
#
# 리뷰 지적사항 대응:
# 1. **표본 크기 확대**: general.csv (100문장) → n=25/group
# 2. **교락 변수 통제**: minimal_pair.csv (100쌍) → paired test
#
# minimal_pair는 "구문적 압축"(같은 명제, 다른 표현 길이)을 테스트.
# 원래 실험의 "명제적 밀도"(다른 명제 수)와 구분하여 해석.

#%%
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

REVIEW_DIR = Path(__file__).parent.parent / "review"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "mps"

#%% [markdown]
# ## 1. 신호 추출 함수 (기존 코드 재사용)

#%%
def extract_layer_delta(text, model, tokenizer, device):
    """Layer Delta 추출 (mean-pooled cosine distance between layers)"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[1:]  # skip embedding
    deltas = []
    for i in range(1, len(hs)):
        h_prev = hs[i-1].squeeze(0).float().mean(dim=0)
        h_curr = hs[i].squeeze(0).float().mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0))
        deltas.append((1 - cos.item()))
    return np.array(deltas)


def extract_surprisal(text, model, tokenizer, device):
    """토큰별 surprisal 추출 (causal LM)"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]  # (seq_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_ids = input_ids[0, 1:]
    pred_logprobs = log_probs[:-1, :]
    surprisals = -pred_logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1) / np.log(2)
    s = surprisals.cpu().numpy()
    if len(s) == 0:
        return {"mean": 0, "std": 0, "cv": 0}
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "cv": float(s.std() / (s.mean() + 1e-8)),
    }


def extract_convergence_area(text, model, tokenizer, device):
    """BERT iterative unmasking convergence area"""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    original_tokens = input_ids.squeeze(0).clone()

    special_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    special_mask[0] = True
    special_mask[-1] = True
    content_positions = (~special_mask).nonzero(as_tuple=True)[0]
    n_content = len(content_positions)
    if n_content == 0:
        return 0.0

    masked_ids = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    masked_ids[0, content_positions] = mask_token_id

    entropy_curve = []
    still_masked = set(content_positions.tolist())
    eps = 1e-10

    for step in range(n_content + 1):
        with torch.no_grad():
            outputs = model(masked_ids)
            logits = outputs.logits[0].float()
        probs = logits.softmax(dim=-1)
        ent = -(probs * torch.log2(probs + eps)).sum(dim=-1).cpu().numpy()
        mean_ent = ent[content_positions.cpu().numpy()].mean()
        entropy_curve.append(mean_ent)

        if not still_masked:
            break
        masked_list = list(still_masked)
        max_probs = probs.max(dim=-1).values.cpu().numpy()
        best_pos = max(masked_list, key=lambda p: max_probs[p])
        predicted = probs[best_pos].argmax().item()
        masked_ids[0, best_pos] = predicted
        still_masked.discard(best_pos)

    area = float(np.trapz(entropy_curve, dx=1.0 / max(len(entropy_curve) - 1, 1)))
    return area


#%% [markdown]
# ## 2. 모델 로더

#%%
def load_encoder(model_name, device):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, attn_implementation="eager"
    ).to(device).eval()
    return model, tokenizer


def load_decoder(model_name, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    trust = "qwen" in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust, attn_implementation="eager"
    ).to(device).eval()
    return model, tokenizer


def load_encoder_base(model_name, device):
    """For Layer Delta — use AutoModel, not AutoModelForMaskedLM"""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, attn_implementation="eager"
    ).to(device).eval()
    return model, tokenizer


#%% [markdown]
# ## 3. General.csv 실험

#%%
def run_general_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 1: general.csv (n=25/group, propositional density)")
    print("="*70)

    df = pd.read_csv(REVIEW_DIR / "general.csv")
    results = []

    # ── klue/bert-base: Layer Delta (KO) ────────────────────────────
    print("\n[klue/bert-base] Layer Delta (KO)...")
    model, tokenizer = load_encoder_base("klue/bert-base", DEVICE)
    ko = df[df["language"] == "ko"]
    for _, row in ko.iterrows():
        ld = extract_layer_delta(row["sentence"], model, tokenizer, DEVICE)
        results.append({
            "id": row["id"], "lang": "ko", "density": row["density_label"],
            "signal": "layer_delta", "value": ld.mean(), "model": "klue/bert-base"
        })
    del model

    # ── bert-base-uncased: Layer Delta (EN) ─────────────────────────
    print("[bert-base-uncased] Layer Delta (EN)...")
    model, tokenizer = load_encoder_base("bert-base-uncased", DEVICE)
    en = df[df["language"] == "en"]
    for _, row in en.iterrows():
        ld = extract_layer_delta(row["sentence"], model, tokenizer, DEVICE)
        results.append({
            "id": row["id"], "lang": "en", "density": row["density_label"],
            "signal": "layer_delta", "value": ld.mean(), "model": "bert-base-uncased"
        })
    del model

    # ── Qwen3: Surprisal (KO + EN) ─────────────────────────────────
    print("[Qwen3-0.6B] Surprisal (KO+EN)...")
    model, tokenizer = load_decoder("Qwen/Qwen3-0.6B", DEVICE)
    for _, row in df.iterrows():
        s = extract_surprisal(row["sentence"], model, tokenizer, DEVICE)
        for metric in ["mean", "cv"]:
            results.append({
                "id": row["id"], "lang": row["language"], "density": row["density_label"],
                "signal": f"surprisal_{metric}", "value": s[metric], "model": "Qwen3-0.6B"
            })
    del model

    # ── BERT: Convergence Area (KO + EN) ────────────────────────────
    for lang, mname in [("ko", "klue/bert-base"), ("en", "bert-base-uncased")]:
        print(f"[{mname}] Convergence Area ({lang.upper()})...")
        model, tokenizer = load_encoder(mname, DEVICE)
        subset = df[df["language"] == lang]
        for _, row in subset.iterrows():
            area = extract_convergence_area(row["sentence"], model, tokenizer, DEVICE)
            results.append({
                "id": row["id"], "lang": lang, "density": row["density_label"],
                "signal": "convergence_area", "value": area, "model": mname
            })
        del model

    return pd.DataFrame(results)


#%% [markdown]
# ## 4. Minimal Pair 실험

#%%
def run_minimal_pair_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 2: minimal_pair.csv (n=50/lang, syntactic compression)")
    print("="*70)

    df = pd.read_csv(REVIEW_DIR / "minimal_pair.csv")
    results = []

    # ── klue/bert-base: Layer Delta (KO pairs) ──────────────────────
    print("\n[klue/bert-base] Layer Delta (KO pairs)...")
    model, tokenizer = load_encoder_base("klue/bert-base", DEVICE)
    ko = df[df["language"] == "ko"]
    for _, row in ko.iterrows():
        for density, col in [("low", "low_density_sentence"), ("high", "high_density_sentence")]:
            ld = extract_layer_delta(row[col], model, tokenizer, DEVICE)
            results.append({
                "pair_id": row["pair_id"], "lang": "ko", "density": density,
                "signal": "layer_delta", "value": ld.mean(), "model": "klue/bert-base"
            })
    del model

    # ── bert-base-uncased: Layer Delta (EN pairs) ───────────────────
    print("[bert-base-uncased] Layer Delta (EN pairs)...")
    model, tokenizer = load_encoder_base("bert-base-uncased", DEVICE)
    en = df[df["language"] == "en"]
    for _, row in en.iterrows():
        for density, col in [("low", "low_density_sentence"), ("high", "high_density_sentence")]:
            ld = extract_layer_delta(row[col], model, tokenizer, DEVICE)
            results.append({
                "pair_id": row["pair_id"], "lang": "en", "density": density,
                "signal": "layer_delta", "value": ld.mean(), "model": "bert-base-uncased"
            })
    del model

    # ── Qwen3: Surprisal (KO + EN pairs) ───────────────────────────
    print("[Qwen3-0.6B] Surprisal (KO+EN pairs)...")
    model, tokenizer = load_decoder("Qwen/Qwen3-0.6B", DEVICE)
    for _, row in df.iterrows():
        lang = row["language"]
        for density, col in [("low", "low_density_sentence"), ("high", "high_density_sentence")]:
            s = extract_surprisal(row[col], model, tokenizer, DEVICE)
            for metric in ["mean", "cv"]:
                results.append({
                    "pair_id": row["pair_id"], "lang": lang, "density": density,
                    "signal": f"surprisal_{metric}", "value": s[metric], "model": "Qwen3-0.6B"
                })
    del model

    # ── BERT: Convergence Area (KO + EN pairs) ──────────────────────
    for lang, mname in [("ko", "klue/bert-base"), ("en", "bert-base-uncased")]:
        print(f"[{mname}] Convergence Area ({lang.upper()} pairs)...")
        model, tokenizer = load_encoder(mname, DEVICE)
        subset = df[df["language"] == lang]
        for _, row in subset.iterrows():
            for density, col in [("low", "low_density_sentence"), ("high", "high_density_sentence")]:
                area = extract_convergence_area(row[col], model, tokenizer, DEVICE)
                results.append({
                    "pair_id": row["pair_id"], "lang": lang, "density": density,
                    "signal": "convergence_area", "value": area, "model": mname
                })
        del model

    return pd.DataFrame(results)


#%% [markdown]
# ## 5. 통계 분석 및 시각화

#%%
def analyze_general(df):
    print("\n" + "="*70)
    print("GENERAL.CSV RESULTS (Mann-Whitney U, independent samples)")
    print("="*70)

    signals = df["signal"].unique()
    for lang in ["ko", "en"]:
        print(f"\n[{lang.upper()}]")
        subset = df[df["lang"] == lang]
        for sig in signals:
            s = subset[subset["signal"] == sig]
            high = s[s["density"] == "high"]["value"].values
            low = s[s["density"] == "low"]["value"].values
            if len(high) < 2 or len(low) < 2:
                continue
            _, p = stats.mannwhitneyu(high, low, alternative="two-sided")
            direction = "High > Low" if high.mean() > low.mean() else "High < Low"
            sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            model = s["model"].iloc[0]
            print(f"  {sig:<20} ({model:<20}) {direction}  "
                  f"H={high.mean():.4f} L={low.mean():.4f}  p={p:.4f} {sig_str}")


def analyze_minimal_pairs(df):
    print("\n" + "="*70)
    print("MINIMAL_PAIR.CSV RESULTS (Wilcoxon signed-rank, paired samples)")
    print("="*70)

    signals = df["signal"].unique()
    for lang in ["ko", "en"]:
        print(f"\n[{lang.upper()}]")
        subset = df[df["lang"] == lang]
        for sig in signals:
            s = subset[subset["signal"] == sig]
            pairs = s.groupby("pair_id")
            high_vals, low_vals = [], []
            for pid, group in pairs:
                h = group[group["density"] == "high"]["value"].values
                l = group[group["density"] == "low"]["value"].values
                if len(h) == 1 and len(l) == 1:
                    high_vals.append(h[0])
                    low_vals.append(l[0])
            high_vals = np.array(high_vals)
            low_vals = np.array(low_vals)
            if len(high_vals) < 5:
                continue
            _, p = stats.wilcoxon(high_vals, low_vals, alternative="two-sided")
            direction = "High > Low" if high_vals.mean() > low_vals.mean() else "High < Low"
            sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            model = s["model"].iloc[0]
            print(f"  {sig:<20} ({model:<20}) {direction}  "
                  f"H={high_vals.mean():.4f} L={low_vals.mean():.4f}  p={p:.4f} {sig_str}")


def plot_comparison(gen_df, mp_df):
    """General vs Minimal Pair 비교 시각화"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        "Propositional Density (general.csv) vs Syntactic Compression (minimal_pair.csv)",
        fontsize=13, fontweight="bold"
    )

    signals = ["layer_delta", "surprisal_mean", "surprisal_cv", "convergence_area"]
    titles = ["Layer Delta", "Surprisal Mean", "Surprisal CV", "Convergence Area"]

    for col, (sig, title) in enumerate(zip(signals, titles)):
        for row, (data, label, test) in enumerate([
            (gen_df, "General (propositional)", "Mann-Whitney"),
            (mp_df, "Minimal Pair (syntactic)", "Wilcoxon"),
        ]):
            ax = axes[row, col]
            for lang, color in [("ko", "tab:blue"), ("en", "tab:orange")]:
                subset = data[(data["signal"] == sig) & (data["lang"] == lang)]
                high = subset[subset["density"] == "high"]["value"].values
                low = subset[subset["density"] == "low"]["value"].values
                if len(high) < 2 or len(low) < 2:
                    continue

                if test == "Wilcoxon" and len(high) == len(low):
                    _, p = stats.wilcoxon(high, low, alternative="two-sided")
                else:
                    _, p = stats.mannwhitneyu(high, low, alternative="two-sided")
                sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

                x_offset = -0.15 if lang == "ko" else 0.15
                bp = ax.boxplot(
                    [high, low], positions=[0 + x_offset, 1 + x_offset],
                    widths=0.25, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.3),
                    medianprops=dict(color=color)
                )
                ax.text(0.5 + x_offset, ax.get_ylim()[1] * 0.95,
                       f"{lang.upper()} {sig_str}", fontsize=7, ha="center", color=color)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["High", "Low"], fontsize=8)
            ax.set_title(f"{title}\n({label})", fontsize=8)
            ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / "step3_validation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    gen_df = run_general_experiment()
    mp_df = run_minimal_pair_experiment()

    analyze_general(gen_df)
    analyze_minimal_pairs(mp_df)

    plot_comparison(gen_df, mp_df)

    # CSV 저장
    gen_df.to_csv(OUTPUT_DIR / "step3_general_results.csv", index=False)
    mp_df.to_csv(OUTPUT_DIR / "step3_minimal_pair_results.csv", index=False)
    print(f"\nSaved: step3_general_results.csv, step3_minimal_pair_results.csv")
