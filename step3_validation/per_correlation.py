#%% [markdown]
# # PER-Signal Correlation Analysis
#
# general.csv의 100문장에 대해:
# 1. Qwen3-0.6B로 PER을 자동 측정 (LLM paraphrase 대신 surprisal 기반 proxy)
# 2. PER(proxy) vs Layer Delta / Surprisal CV / Convergence Area 상관 분석
#
# 실제 LLM API로 PER을 재려면 6개 모델에 각각 프롬프트를 보내야 하므로,
# 여기서는 이미 측정된 내부 신호 간의 상관을 먼저 분석한다.
# → "Layer Delta가 높은 문장은 Surprisal도 높은가?" 등 신호 간 일관성 확인
#
# 추가로: Step 0에서 측정한 원본 20문장의 PER 값과 내부 신호의 상관도 계산

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import json

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
REVIEW_DIR = Path(__file__).parent.parent / "review"
DATA_DIR   = Path(__file__).parent.parent / "data"
PER_DIR    = Path(__file__).parent.parent / "step0_per" / "per_results"

#%% [markdown]
# ## 1. Step 3 신호 간 상관 (general.csv 100문장)

#%%
def analyze_signal_correlations():
    """general.csv 실험 결과에서 신호 간 Spearman 상관 행렬"""
    df = pd.read_csv(OUTPUT_DIR / "step3_general_results.csv")

    print("="*65)
    print("SIGNAL CROSS-CORRELATION (general.csv, n=100)")
    print("="*65)

    for lang in ["ko", "en"]:
        print(f"\n[{lang.upper()}]")
        sub = df[df["lang"] == lang]

        # pivot: 각 문장의 각 신호 값
        signals = {}
        for sig in ["layer_delta", "surprisal_mean", "surprisal_cv", "convergence_area"]:
            s = sub[sub["signal"] == sig][["id", "value"]].set_index("id")["value"]
            signals[sig] = s

        sig_df = pd.DataFrame(signals)
        if len(sig_df) < 5:
            continue

        # Spearman 상관 행렬
        corr_matrix = sig_df.corr(method="spearman")
        print(corr_matrix.round(3).to_string())

        # 주요 상관 p값
        pairs = [
            ("layer_delta", "surprisal_mean"),
            ("layer_delta", "surprisal_cv"),
            ("surprisal_mean", "convergence_area"),
            ("surprisal_cv", "convergence_area"),
        ]
        print("\n  Key correlations:")
        for a, b in pairs:
            if a in sig_df.columns and b in sig_df.columns:
                vals_a = sig_df[a].dropna()
                vals_b = sig_df[b].dropna()
                common = vals_a.index.intersection(vals_b.index)
                if len(common) > 5:
                    rho, p = stats.spearmanr(vals_a[common], vals_b[common])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"    {a:<20} vs {b:<20} rho={rho:.3f}  p={p:.4f} {sig}")

        # 시각화: scatter matrix
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Signal Cross-Correlations ({lang.upper()}, general.csv)", fontweight="bold")

        for ax, (a, b) in zip(axes.flat, pairs):
            if a in sig_df.columns and b in sig_df.columns:
                common = sig_df[[a, b]].dropna()
                colors = ["red" if i.endswith(("h01","h02","h03","h04","h05","h06","h07","h08","h09","h10",
                    "h11","h12","h13","h14","h15","h16","h17","h18","h19","h20","h21","h22","h23","h24","h25"))
                    else "blue" for i in common.index]
                ax.scatter(common[a], common[b], c=colors, alpha=0.6, s=30)
                rho, p = stats.spearmanr(common[a], common[b])
                ax.set_xlabel(a, fontsize=8)
                ax.set_ylabel(b, fontsize=8)
                ax.set_title(f"rho={rho:.3f}, p={p:.4f}", fontsize=9)
                ax.grid(alpha=0.2)

        plt.tight_layout()
        out = OUTPUT_DIR / f"step3_signal_corr_{lang}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {out.name}")


#%% [markdown]
# ## 2. 원본 20문장: PER vs 내부 신호 상관

#%%
def load_per_scores():
    """Step 0 PER 결과에서 GPT+Claude 앙상블 PER 계산

    PER 파일 형태: {KO_H01: "풀어쓴 텍스트", KO_H02: "...", ...}
    → 원본 토큰 수와 풀어쓴 토큰 수의 비율로 PER 계산
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    # 원본 문장 로드
    with open(DATA_DIR / "seed_sentences.json") as f:
        seeds = json.load(f)["sentences"]
    original_tokens = {}
    id_map = {}  # KO_H01 → ko_h01
    for s in seeds:
        n_tok = len(tokenizer.encode(s["text"]))
        # seed의 id는 ko_h01 형태, PER 파일은 KO_H01 형태
        per_id = s["id"].upper().replace("_", "_")
        original_tokens[per_id] = n_tok
        id_map[per_id] = s["id"]

    per_files = list(PER_DIR.glob("*.json"))
    valid_models = ["claude-opus-4.6", "claude-sonnet-4.6", "gpt-5.4-pro", "gpt-5.4-thinking"]

    all_per = {}
    for f in per_files:
        model_name = f.stem
        if model_name not in valid_models:
            continue
        with open(f) as fp:
            data = json.load(fp)

        for per_id, paraphrase_text in data.items():
            if not isinstance(paraphrase_text, str):
                continue
            if per_id not in original_tokens:
                continue
            n_para = len(tokenizer.encode(paraphrase_text))
            n_orig = original_tokens[per_id]
            per_val = n_para / n_orig if n_orig > 0 else 0

            seed_id = id_map[per_id]
            if seed_id not in all_per:
                all_per[seed_id] = []
            all_per[seed_id].append(per_val)

    # 앙상블: median PER across GPT+Claude models
    ensemble = {}
    for sid, vals in all_per.items():
        if len(vals) >= 2:
            ensemble[sid] = np.median(vals)

    print(f"  Loaded PER for {len(ensemble)} sentences "
          f"(median across {len(valid_models)} models)")
    return ensemble


def analyze_per_signal_correlation():
    """원본 20문장의 PER vs Step 1 내부 신호 상관"""
    # PER 로드
    per_scores = load_per_scores()
    if not per_scores:
        print("No PER scores found. Skipping.")
        return

    print("\n" + "="*65)
    print(f"PER vs INTERNAL SIGNALS (original 20 sentences, n_per={len(per_scores)})")
    print("="*65)

    # Step 1 신호 로드 (원본 20문장 — step3_general에는 없으므로 직접 계산이 필요)
    # 대신 step3_general_results.csv에서 원본 seed sentence ID와 매칭되는 것을 찾자
    # 원본 seed sentences의 ID: ko_h01~ko_h05, ko_l01~ko_l05, en_h01~en_h05, en_l01~en_l05
    # general.csv의 ID: G001~G100 (다른 문장)
    # → 매칭 불가. 원본 20문장에 대해 직접 신호를 추출해야 함.

    # Step 1 surprisal 결과를 직접 로드 (step1_surprisal.py의 결과를 재현)
    import sys, torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "step1_internal"))

    from pathlib import Path as P
    seed_path = DATA_DIR / "seed_sentences.json"
    with open(seed_path) as f:
        sentences = json.load(f)["sentences"]

    # Surprisal 추출
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DEVICE = "mps"
    model_name = "Qwen/Qwen3-0.6B"

    print(f"\nLoading {model_name} for surprisal...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(DEVICE).eval()

    # Layer Delta 추출
    from transformers import AutoModel
    enc_models = {
        "ko": ("klue/bert-base", None),
        "en": ("bert-base-uncased", None),
    }

    results = []
    for s in sentences:
        sid = s["id"]
        if sid not in per_scores:
            continue

        # Surprisal
        inputs = tokenizer(s["text"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = inputs["input_ids"][0, 1:]
        pred_lp = log_probs[:-1, :]
        surp = -pred_lp.gather(1, token_ids.unsqueeze(1)).squeeze(1) / np.log(2)
        surp_np = surp.cpu().numpy()

        results.append({
            "id": sid,
            "lang": s["lang"],
            "density": s["density"],
            "per": per_scores[sid],
            "surprisal_mean": float(surp_np.mean()),
            "surprisal_cv": float(surp_np.std() / (surp_np.mean() + 1e-8)),
        })

    del model

    # Layer Delta per language
    for lang in ["ko", "en"]:
        enc_name = "klue/bert-base" if lang == "ko" else "bert-base-uncased"
        print(f"Loading {enc_name} for Layer Delta...")
        enc_tok = AutoTokenizer.from_pretrained(enc_name)
        enc_model = AutoModel.from_pretrained(enc_name, attn_implementation="eager").to(DEVICE).eval()

        for r in results:
            if r["lang"] != lang:
                continue
            sent = [s for s in sentences if s["id"] == r["id"]][0]
            inputs = enc_tok(sent["text"], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = enc_model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[1:]
            deltas = []
            for i in range(1, len(hs)):
                h_prev = hs[i-1].squeeze(0).float().mean(dim=0)
                h_curr = hs[i].squeeze(0).float().mean(dim=0)
                cos = torch.nn.functional.cosine_similarity(h_prev.unsqueeze(0), h_curr.unsqueeze(0))
                deltas.append(1 - cos.item())
            r["layer_delta"] = np.mean(deltas)

        del enc_model

    rdf = pd.DataFrame(results)
    print(f"\nMatched {len(rdf)} sentences with PER scores")
    print(rdf[["id", "lang", "density", "per", "surprisal_mean", "surprisal_cv", "layer_delta"]].to_string())

    # 상관 분석
    print("\n--- PER vs Signal Correlations ---")
    for sig in ["layer_delta", "surprisal_mean", "surprisal_cv"]:
        valid = rdf[[sig, "per"]].dropna()
        if len(valid) < 5:
            continue
        rho, p = stats.spearmanr(valid["per"], valid[sig])
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  PER vs {sig:<20} rho={rho:.3f}  p={p:.4f} {sig_str}")

        # 언어별
        for lang in ["ko", "en"]:
            sub = rdf[rdf["lang"] == lang][[sig, "per"]].dropna()
            if len(sub) >= 4:
                r, pv = stats.spearmanr(sub["per"], sub[sig])
                s2 = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
                print(f"    [{lang.upper()}] rho={r:.3f}  p={pv:.4f} {s2}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("PER vs Transformer Internal Signals (original 20 sentences)", fontweight="bold")

    for ax, sig, label in zip(axes, ["layer_delta", "surprisal_mean", "surprisal_cv"],
                               ["Layer Delta", "Surprisal Mean", "Surprisal CV"]):
        for lang, color, marker in [("ko", "tab:blue", "o"), ("en", "tab:orange", "s")]:
            sub = rdf[rdf["lang"] == lang]
            ax.scatter(sub["per"], sub[sig], c=color, marker=marker, s=60, alpha=0.7,
                      label=f"{lang.upper()}", edgecolors="white", linewidth=0.5)

        # 전체 추세선
        valid = rdf[["per", sig]].dropna()
        if len(valid) > 3:
            z = np.polyfit(valid["per"], valid[sig], 1)
            x_line = np.linspace(valid["per"].min(), valid["per"].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4, lw=1)
            rho, p = stats.spearmanr(valid["per"], valid[sig])
            ax.text(0.05, 0.95, f"rho={rho:.3f}\np={p:.4f}",
                   transform=ax.transAxes, fontsize=9, va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("PER (Paraphrase Expansion Ratio)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "step3_per_signal_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    analyze_signal_correlations()
    analyze_per_signal_correlation()
