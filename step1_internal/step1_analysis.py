#%% [markdown]
# # Step 1: Transformer Internal Signals — Encoder vs Decoder
#
# 고밀도 vs 저밀도 문장을 4개 모델에 통과시켜 층별 내부 신호를 비교한다.
#
# **모델 구성 (이전 mBERT/GPT-2에서 개선):**
# - `klue/bert-base`     : 한국어 전용 encoder (KO 문장 전담)
# - `bert-base-uncased`  : 영어 전용 encoder   (EN 문장 전담)
# - `gpt2`               : 영어 causal decoder (EN baseline)
# - `Qwen/Qwen3-0.6B`    : 한/영 multilingual modern decoder
#
# **왜 이 조합인가?**
# 이전 null result의 주원인이 "언어-모델 불일치"였다.
# klue/bert-base는 한국어로 사전학습됨 → KO 고밀도 문장에 의미있는 신호 기대.
# Qwen3-0.6B는 최신 multilingual decoder → GPT-2(영어 전용)의 한계 극복.
#
# **문장 선택:** Step 0 PER 결과에서 GPT+Claude 앙상블 기준 상위 고밀도 / 하위 저밀도

#%%
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from extract_signals import (
    InternalSignals,
    extract_encoder_signals,
    extract_decoder_signals,
)

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"  # Apple Silicon; "cuda" or "cpu" otherwise

# ── 모델 설정 ──────────────────────────────────────────────────────────────
# 이전 실험의 언어-모델 불일치 문제를 해결하기 위해 언어별 전담 모델 분리
MODEL_CONFIGS = {
    # (model_name, model_type, lang_filter)
    # lang_filter: "ko" | "en" | None (all)
    "klue_bert":  ("klue/bert-base",      "encoder", "ko"),   # 한국어 전용 BERT
    "bert_en":    ("bert-base-uncased",   "encoder", "en"),   # 영어 전용 BERT
    "gpt2":       ("gpt2",                "decoder", "en"),   # 영어 causal baseline
    "qwen3":      ("Qwen/Qwen3-0.6B",     "decoder", None),  # multilingual modern
}

#%% [markdown]
# ## 문장 로드 — Step 0에서 신뢰도 높은 쌍만 사용

#%%
def load_sentences() -> tuple[list[dict], list[dict]]:
    """
    seed_sentences.json에서 고밀도/저밀도 문장 로드.
    Step 0 결과: KO High가 가장 일관되게 인식됨 (CV 0.27~0.28)
    """
    with open(DATA_DIR / "seed_sentences.json") as f:
        data = json.load(f)

    high, low = [], []
    for s in data["sentences"]:
        if s["density"] == "high":
            high.append(s)
        else:
            low.append(s)

    print(f"High-density: {len(high)} sentences")
    print(f"Low-density:  {len(low)} sentences")
    return high, low

#%% [markdown]
# ## 신호 추출 — 모델 1개 로드 후 전체 문장에 적용
#
# 매 문장마다 모델을 재로드하면 느리므로 모델을 먼저 로드하고 넘겨준다.

#%%
def extract_all_signals(
    sentences: list[dict],
    model_name: str,
    model_type: str,  # "encoder" or "decoder"
    device: str = DEVICE,
    lang_filter: str | None = None,  # "ko", "en", or None (all)
) -> list[InternalSignals]:
    """모든 문장에 대해 내부 신호 추출 (모델 1회 로드)

    lang_filter: 지정 시 해당 언어 문장만 처리 (언어-모델 매칭용)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    if lang_filter:
        sentences = [s for s in sentences if s["lang"] == lang_filter]
        print(f"  → {lang_filter.upper()} 문장만 처리 ({len(sentences)}개)")

    print(f"\nLoading {model_name} ({model_type})...")
    trust = "qwen" in model_name.lower() or "klue" in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "encoder":
        # attn_implementation="eager": transformers 5.x에서 SDPA가 기본값
        # output_attentions=True를 사용하려면 eager로 강제
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust, attn_implementation="eager"
        ).to(device).eval()
    else:
        # attn_implementation="eager": SDPA는 output_attentions=True 미지원
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust, attn_implementation="eager"
        ).to(device).eval()

    results = []
    for s in sentences:
        text = s["text"]
        print(f"  [{s['id']}] {text[:40]}...")

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        # ── Attention entropy & distance ──────────────────────────────
        from extract_signals import compute_attention_entropy, compute_attention_distance, compute_hidden_state_metrics

        attn_stack = torch.stack([a.squeeze(0) for a in outputs.attentions])
        # (layers, heads, S, S)
        entropy  = compute_attention_entropy(attn_stack)   # (layers, heads)
        attn_dist = compute_attention_distance(attn_stack) # (layers, heads)

        # ── Hidden state metrics ──────────────────────────────────────
        hs_metrics = compute_hidden_state_metrics(outputs.hidden_states[1:])

        sig = InternalSignals(
            text=text,
            density_label=s["density"],
            lang=s["lang"],
            model_name=model_name,
            model_type=model_type,
            attention_entropy=entropy.mean(dim=-1).cpu().numpy(),
            hidden_state_norm=hs_metrics["norm"],
            layer_delta=hs_metrics["delta"],
            effective_rank=hs_metrics["effective_rank"],
            mean_attention_distance=attn_dist.mean(dim=-1).cpu().numpy(),
        )
        results.append(sig)

    del model
    return results

#%% [markdown]
# ## Fig 1: 층별 신호 — 언어별 × 모델별
#
# 각 신호의 층별 평균 곡선 (High=red, Low=blue, shade=±std)

#%%
SIGNAL_META = [
    ("attention_entropy",       "Attention Entropy (bits)",  "분산된 attention → 높음"),
    ("hidden_state_norm",       "Hidden State L2 Norm",      "정보 인코딩량 → 높음"),
    ("layer_delta",             "Layer Delta (1−cos sim)",   "층간 표현 변화량 → 높음"),
    ("effective_rank",          "Effective Rank",            "활용 차원 수 → 높음"),
    ("mean_attention_distance", "Attention Distance",        "원거리 의존성 → 높음"),
]

def plot_layer_curves(
    signals: list[InternalSignals],
    model_label: str,
    lang_filter: str | None = None,  # "ko", "en", or None (all)
    save_name: str = "signals",
):
    """층별 신호 곡선 (5가지 신호 × high/low 비교)"""
    if lang_filter:
        signals = [s for s in signals if s.lang == lang_filter]

    high = [s for s in signals if s.density_label == "high"]
    low  = [s for s in signals if s.density_label == "low"]

    if not high or not low:
        print(f"No data for lang={lang_filter}")
        return

    n_layers = len(high[0].attention_entropy)
    layers = np.arange(n_layers)
    lang_label = {"ko": "Korean", "en": "English", None: "All"}[lang_filter]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"{model_label} | {lang_label} | Layer-wise Signals: High vs Low Density",
        fontsize=12, fontweight="bold"
    )

    for ax, (attr, title, hint) in zip(axes, SIGNAL_META):
        h_arr = np.stack([getattr(s, attr) for s in high if getattr(s, attr) is not None])
        l_arr = np.stack([getattr(s, attr) for s in low  if getattr(s, attr) is not None])

        h_mean, h_std = h_arr.mean(0), h_arr.std(0)
        l_mean, l_std = l_arr.mean(0), l_arr.std(0)

        ax.plot(layers, h_mean, "r-o", ms=3, lw=1.5, label="High")
        ax.fill_between(layers, h_mean - h_std, h_mean + h_std, alpha=0.15, color="red")
        ax.plot(layers, l_mean, "b-s", ms=3, lw=1.5, label="Low")
        ax.fill_between(layers, l_mean - l_std, l_mean + l_std, alpha=0.15, color="blue")

        ax.set_xlabel("Layer", fontsize=8)
        ax.set_title(f"{title}\n({hint})", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = OUTPUT_DIR / f"step1_{save_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")

#%% [markdown]
# ## Fig 2: 통계 검정 — 층별 t-test
#
# 각 층에서 High vs Low가 통계적으로 유의하게 다른가?
# → 어느 층에서 밀도 신호가 가장 강한지 알 수 있음

#%%
def plot_statistical_significance(
    signals: list[InternalSignals],
    model_label: str,
    save_name: str = "ttest",
):
    """층별 t-test: -log10(p) 히트맵"""
    high = [s for s in signals if s.density_label == "high"]
    low  = [s for s in signals if s.density_label == "low"]
    n_layers = len(high[0].attention_entropy)

    sig_matrix = np.zeros((len(SIGNAL_META), n_layers))
    direction  = np.zeros((len(SIGNAL_META), n_layers))  # +1: high>low, -1: high<low

    for i, (attr, _, _) in enumerate(SIGNAL_META):
        h_arr = np.stack([getattr(s, attr) for s in high])
        l_arr = np.stack([getattr(s, attr) for s in low])
        for l in range(n_layers):
            _, p = stats.ttest_ind(h_arr[:, l], l_arr[:, l])
            sig_matrix[i, l] = -np.log10(p + 1e-10)
            direction[i, l]  = np.sign(h_arr[:, l].mean() - l_arr[:, l].mean())

    # 방향 반영: High > Low → 양수 (red), High < Low → 음수 (blue)
    signed = sig_matrix * direction

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.5), 3.5))
    im = ax.imshow(signed, aspect="auto", cmap="RdBu_r",
                   vmin=-4, vmax=4)
    plt.colorbar(im, ax=ax, label="sign × −log₁₀(p)\nRed=High>Low, Blue=High<Low")

    ax.set_yticks(range(len(SIGNAL_META)))
    ax.set_yticklabels([m[1] for m in SIGNAL_META], fontsize=8)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers), fontsize=7)
    ax.set_xlabel("Layer")

    # p < 0.05 표시 (|signed| > 1.3)
    for i in range(len(SIGNAL_META)):
        for l in range(n_layers):
            if abs(signed[i, l]) > 1.3:
                ax.text(l, i, "*", ha="center", va="center", fontsize=10, color="black")

    ax.set_title(
        f"{model_label} | Statistical Significance per Layer\n"
        "* p < 0.05  (color intensity = -log₁₀p)",
        fontsize=10
    )
    plt.tight_layout()
    out = OUTPUT_DIR / f"step1_{save_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")

#%% [markdown]
# ## Fig 3: Encoder vs Decoder 직접 비교
#
# 같은 문장에 대해 BERT와 GPT-2의 신호가 어떻게 다른가?
# → 처리 방식(양방향 vs 단방향)이 밀도 반응에 영향을 주는가?

#%%
def plot_paradigm_comparison(
    enc_signals: list[InternalSignals],
    dec_signals: list[InternalSignals],
    attr: str = "attention_entropy",
    title: str = "Attention Entropy",
):
    """Encoder vs Decoder: 같은 신호를 양쪽에서 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    fig.suptitle(
        f"Paradigm Comparison: {title}\nEncoder (klue/bert-base) vs Decoder (Qwen3-0.6B)",
        fontsize=11, fontweight="bold"
    )

    for ax, (signals, label) in zip(axes, [
        (enc_signals, "klue/bert-base (Encoder — Bidirectional)"),
        (dec_signals, "Qwen3-0.6B (Decoder — Causal)"),
    ]):
        high = [s for s in signals if s.density_label == "high"]
        low  = [s for s in signals if s.density_label == "low"]

        h_arr = np.stack([getattr(s, attr) for s in high])
        l_arr = np.stack([getattr(s, attr) for s in low])
        layers = np.arange(h_arr.shape[1])

        ax.plot(layers, h_arr.mean(0), "r-o", ms=4, lw=2, label="High density")
        ax.fill_between(layers, h_arr.mean(0)-h_arr.std(0),
                         h_arr.mean(0)+h_arr.std(0), alpha=0.15, color="red")
        ax.plot(layers, l_arr.mean(0), "b-s", ms=4, lw=2, label="Low density")
        ax.fill_between(layers, l_arr.mean(0)-l_arr.std(0),
                         l_arr.mean(0)+l_arr.std(0), alpha=0.15, color="blue")

        # 두 그룹 차이 (shaded area)
        diff = h_arr.mean(0) - l_arr.mean(0)
        ax2 = ax.twinx()
        ax2.bar(layers, diff, alpha=0.15, color="green", label="Δ(High−Low)")
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        ax2.set_ylabel("Δ (High − Low)", color="green", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=7)

        ax.set_xlabel("Layer"); ax.set_title(label, fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

    plt.tight_layout()
    out = OUTPUT_DIR / f"step1_paradigm_{attr}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")

#%% [markdown]
# ## Summary Statistics

#%%
def print_summary(all_model_signals: dict[str, list[InternalSignals]]):
    print("\n" + "="*70)
    print("STEP 1 SUMMARY — 4-Model Experiment")
    print("="*70)

    for label, model_signals in all_model_signals.items():
        if not model_signals:
            continue
        high = [s for s in model_signals if s.density_label == "high"]
        low  = [s for s in model_signals if s.density_label == "low"]
        if len(high) < 2 or len(low) < 2:
            print(f"\n[{label}] insufficient samples (high={len(high)}, low={len(low)})")
            continue

        print(f"\n[{label}] n_high={len(high)}, n_low={len(low)}")
        for attr, title, _ in SIGNAL_META:
            h_vals = np.stack([getattr(s, attr) for s in high]).mean(axis=1)  # per-sentence mean
            l_vals = np.stack([getattr(s, attr) for s in low]).mean(axis=1)
            _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            direction = "High > Low" if h_vals.mean() > l_vals.mean() else "High < Low"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {title:<30} {direction}  p={p:.4f} {sig}")

#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    # 1. 문장 로드
    high_sents, low_sents = load_sentences()
    all_sents = high_sents + low_sents

    # 2. 모델별 신호 추출
    # ── 언어-모델 대응:
    #    klue/bert-base  → KO 전담 (한국어 사전학습)
    #    bert-base-uncased → EN 전담 (영어 사전학습)
    #    gpt2            → EN 전담 (영어 baseline)
    #    Qwen3-0.6B      → KO+EN (multilingual modern decoder)

    all_model_signals: dict[str, list[InternalSignals]] = {}

    for key, (model_name, model_type, lang_filter) in MODEL_CONFIGS.items():
        all_model_signals[key] = extract_all_signals(
            all_sents,
            model_name=model_name,
            model_type=model_type,
            device=DEVICE,
            lang_filter=lang_filter,
        )

    # 편의 변수
    klue_signals  = all_model_signals["klue_bert"]   # KO encoder
    berten_signals = all_model_signals["bert_en"]    # EN encoder
    gpt2_signals  = all_model_signals["gpt2"]        # EN decoder baseline
    qwen_signals  = all_model_signals["qwen3"]       # multilingual modern decoder

    # 3. 시각화
    # Fig 1: klue/bert-base — KO 전용
    plot_layer_curves(klue_signals, "klue/bert-base (KO)", lang_filter=None, save_name="klue_bert_ko")

    # Fig 2: bert-base-uncased — EN 전용
    plot_layer_curves(berten_signals, "bert-base-uncased (EN)", lang_filter=None, save_name="bert_en")

    # Fig 3: GPT-2 — EN
    plot_layer_curves(gpt2_signals, "GPT-2 (EN)", lang_filter=None, save_name="gpt2_en")

    # Fig 4: Qwen3-0.6B — KO / EN / All
    plot_layer_curves(qwen_signals, "Qwen3-0.6B", lang_filter=None, save_name="qwen3_all")
    plot_layer_curves(qwen_signals, "Qwen3-0.6B", lang_filter="ko", save_name="qwen3_ko")
    plot_layer_curves(qwen_signals, "Qwen3-0.6B", lang_filter="en", save_name="qwen3_en")

    # Fig 5: 통계 검정 히트맵 (모델별)
    for key, signals in all_model_signals.items():
        if len([s for s in signals if s.density_label == "high"]) >= 2:
            plot_statistical_significance(signals, key.upper(), save_name=f"{key}_ttest")

    # Fig 6: Encoder vs Decoder 패러다임 비교
    # klue/bert-base (KO) vs Qwen3-0.6B (KO)
    ko_qwen = [s for s in qwen_signals if s.lang == "ko"]
    for attr, title, _ in SIGNAL_META:
        plot_paradigm_comparison(klue_signals, ko_qwen, attr=attr, title=title)

    # 4. 요약
    print_summary(all_model_signals)
