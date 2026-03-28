#%% [markdown]
# # Step 1 Extension: Probing Classifiers
#
# **목적:** 트랜스포머 hidden state에 밀도 정보가 "선형 분리 가능한" 형태로 인코딩되어 있는가?
#
# **방법:** 각 레이어의 frozen hidden state에 logistic regression을 훈련시킨다.
# 높은 정확도 = 해당 레이어가 밀도 정보를 포함하고 있다.
#
# **왜 중요한가:**
# - Step 1 main에서 Layer Delta만 유의미했음 → 다른 신호가 놓친 게 있을 수 있음
# - Probing은 비선형 상호작용을 넘어서 "선형으로 읽어낼 수 있는" 정보만 측정
# - 특정 레이어에서 accuracy가 급등한다면 → 밀도 인코딩이 점진적이 아닌 "특정 층"에서 일어남
#
# **대상 모델:**
# - klue/bert-base (KO): Layer Delta p=0.032* → 가장 유망
# - Qwen3-0.6B (KO+EN): surprisal CV p=0.008** → decoder 관점

#%%
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"

#%% [markdown]
# ## 1. Hidden State 추출
#
# 모델을 로드하고 각 문장의 각 레이어에서 hidden state를 추출한다.
# Mean pooling → 문장 수준의 단일 벡터 (per layer)

#%%
def extract_hidden_states(
    sentences: list[dict],
    model_name: str,
    model_type: str,
    device: str = DEVICE,
    lang_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    각 레이어의 mean-pooled hidden state를 추출.

    Returns:
        features: (n_sentences, n_layers, hidden_dim)
        labels:   (n_sentences,)  — 1=high, 0=low
        sent_ids: list of sentence IDs
    """
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    if lang_filter:
        sentences = [s for s in sentences if s["lang"] == lang_filter]

    print(f"\nLoading {model_name} ({model_type})...")
    trust = "qwen" in model_name.lower() or "klue" in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "encoder":
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust, attn_implementation="eager"
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust, attn_implementation="eager"
        ).to(device).eval()

    all_features = []
    labels = []
    sent_ids = []

    for s in sentences:
        inputs = tokenizer(s["text"], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states[0] = embedding, [1:] = transformer layers
        # Mean pooling across tokens per layer → (n_layers, hidden_dim)
        hs = outputs.hidden_states[1:]  # skip embedding layer
        layer_features = []
        for h in hs:
            # h: (1, seq_len, hidden_dim)
            # Insight: mean pool은 토큰 순서 정보를 잃지만, 문장 수준 표현으로는 충분
            # CLS 토큰만 쓸 수도 있지만, decoder에는 CLS가 없으므로 mean pool로 통일
            pooled = h.squeeze(0).float().mean(dim=0).cpu().numpy()  # (hidden_dim,)
            layer_features.append(pooled)

        all_features.append(np.stack(layer_features))  # (n_layers, hidden_dim)
        labels.append(1 if s["density"] == "high" else 0)
        sent_ids.append(s["id"])

    del model

    features = np.stack(all_features)    # (n_sentences, n_layers, hidden_dim)
    labels = np.array(labels)            # (n_sentences,)
    assert features.shape[0] == len(labels)

    print(f"  Extracted: {features.shape} (sentences, layers, dim)")
    print(f"  Labels: {labels.sum()} high, {(1-labels).sum()} low")
    return features, labels, sent_ids


#%% [markdown]
# ## 2. Layer-wise Probing
#
# 각 레이어에서 독립적으로 logistic regression을 훈련.
# Leave-One-Out CV (n=10이라 k-fold보다 LOO가 적합)

#%%
def probe_per_layer(
    features: np.ndarray,  # (n, n_layers, dim)
    labels: np.ndarray,    # (n,)
) -> dict:
    """
    각 레이어에서 LOO-CV logistic regression으로 밀도 분류.

    Returns:
        dict with 'accuracy', 'auc' arrays per layer
    """
    n_sentences, n_layers, hidden_dim = features.shape

    accuracies = []
    aucs = []

    for layer in range(n_layers):
        X = features[:, layer, :]  # (n, dim)

        # Pipeline: StandardScaler + LogisticRegression
        # Insight: L2 regularization (C=1.0) — 과적합 방지 중요 (n=10, dim=768)
        # 고차원 + 저샘플 → 강한 정규화 필요
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")),
        ])

        # LOO-CV: n=10이면 10-fold와 동일
        loo = LeaveOneOut()
        scores = cross_val_score(pipe, X, labels, cv=loo, scoring="accuracy")
        acc = scores.mean()

        # AUC: LOO로 predicted probabilities 수집
        probs = np.zeros(n_sentences)
        for train_idx, test_idx in loo.split(X):
            pipe_clone = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")),
            ])
            pipe_clone.fit(X[train_idx], labels[train_idx])
            probs[test_idx] = pipe_clone.predict_proba(X[test_idx])[:, 1]

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5  # degenerate case

        accuracies.append(acc)
        aucs.append(auc)

    return {
        "accuracy": np.array(accuracies),
        "auc": np.array(aucs),
        "n_layers": n_layers,
        "n_samples": n_sentences,
    }


#%% [markdown]
# ## 3. 시각화

#%%
def plot_probing_results(
    results: dict[str, dict],
    save_name: str = "probing",
):
    """
    모델별 layer-wise probing accuracy + AUC 곡선.
    가로선 0.5 = chance level (random binary classification)
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Probing Classifiers: Can we linearly decode density from hidden states?",
        fontsize=12, fontweight="bold"
    )

    colors = {"klue_bert": "tab:blue", "bert_en": "tab:cyan",
              "gpt2": "tab:orange", "qwen3": "tab:red",
              "qwen3_ko": "tab:red", "qwen3_en": "tab:pink"}
    markers = {"klue_bert": "o", "bert_en": "s",
               "gpt2": "^", "qwen3": "D",
               "qwen3_ko": "D", "qwen3_en": "d"}

    for metric_ax, (metric, title) in zip(axes, [
        ("accuracy", "LOO-CV Accuracy"),
        ("auc", "LOO-CV AUC"),
    ]):
        for model_key, res in results.items():
            layers = np.arange(res["n_layers"])
            vals = res[metric]
            color = colors.get(model_key, "gray")
            marker = markers.get(model_key, "o")

            metric_ax.plot(layers, vals, f"-{marker}", ms=5, lw=1.5,
                          color=color, label=f"{model_key} (n={res['n_samples']})")

        # chance level
        metric_ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="Chance (0.5)")

        metric_ax.set_xlabel("Layer")
        metric_ax.set_ylabel(title)
        metric_ax.set_title(title)
        metric_ax.legend(fontsize=8, loc="lower right")
        metric_ax.grid(alpha=0.25)
        metric_ax.set_ylim(0.25, 1.05)

    plt.tight_layout()
    out = OUTPUT_DIR / f"step1_{save_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


def plot_probing_detail(
    results: dict[str, dict],
    save_name: str = "probing_detail",
):
    """
    각 모델을 개별 패널로: accuracy + AUC 동시 표시.
    peak layer 표시 (▼ 마커).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Layer-wise Probing: Where does density information emerge?",
        fontsize=12, fontweight="bold"
    )

    for ax, (model_key, res) in zip(axes, results.items()):
        layers = np.arange(res["n_layers"])
        acc = res["accuracy"]
        auc = res["auc"]

        ax.plot(layers, acc, "b-o", ms=4, lw=1.5, label="Accuracy")
        ax.plot(layers, auc, "r-s", ms=4, lw=1.5, label="AUC")
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)

        # peak annotation
        peak_acc_layer = int(np.argmax(acc))
        peak_auc_layer = int(np.argmax(auc))
        ax.annotate(f"peak acc={acc[peak_acc_layer]:.2f}\n(L{peak_acc_layer})",
                    xy=(peak_acc_layer, acc[peak_acc_layer]),
                    xytext=(0, 15), textcoords="offset points",
                    fontsize=7, ha="center", color="blue",
                    arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))
        if peak_auc_layer != peak_acc_layer:
            ax.annotate(f"peak AUC={auc[peak_auc_layer]:.2f}\n(L{peak_auc_layer})",
                        xy=(peak_auc_layer, auc[peak_auc_layer]),
                        xytext=(0, -20), textcoords="offset points",
                        fontsize=7, ha="center", color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        ax.set_xlabel("Layer")
        ax.set_title(f"{model_key}\n(n={res['n_samples']})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.25, 1.05)

    axes[0].set_ylabel("Score")
    plt.tight_layout()
    out = OUTPUT_DIR / f"step1_{save_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


#%% [markdown]
# ## 4. Summary

#%%
def print_probing_summary(results: dict[str, dict]):
    print("\n" + "="*65)
    print("PROBING CLASSIFIER SUMMARY")
    print("="*65)

    for model_key, res in results.items():
        acc = res["accuracy"]
        auc = res["auc"]
        peak_acc_layer = int(np.argmax(acc))
        peak_auc_layer = int(np.argmax(auc))

        print(f"\n[{model_key}] layers={res['n_layers']}, n={res['n_samples']}")
        print(f"  Peak accuracy: {acc[peak_acc_layer]:.2f} at layer {peak_acc_layer}")
        print(f"  Peak AUC:      {auc[peak_auc_layer]:.2f} at layer {peak_auc_layer}")
        print(f"  Chance level:  0.50")

        # above-chance layers
        above = np.where(acc > 0.6)[0]
        if len(above) > 0:
            print(f"  Layers > 60% acc: {above.tolist()}")
        else:
            print(f"  No layer exceeds 60% accuracy")


#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    with open(DATA_DIR / "seed_sentences.json") as f:
        data = json.load(f)
    sentences = data["sentences"]

    probing_results: dict[str, dict] = {}

    # ── klue/bert-base (KO only) ─────────────────────────────────────
    # Layer Delta p=0.032* → probing should confirm density encoding
    feat_klue, labels_klue, ids_klue = extract_hidden_states(
        sentences, "klue/bert-base", "encoder", DEVICE, lang_filter="ko"
    )
    probing_results["klue_bert"] = probe_per_layer(feat_klue, labels_klue)

    # ── Qwen3-0.6B (KO only) ────────────────────────────────────────
    # Surprisal CV p=0.008** for KO → decoder에서도 밀도 인코딩?
    feat_qwen_ko, labels_qwen_ko, ids_qwen_ko = extract_hidden_states(
        sentences, "Qwen/Qwen3-0.6B", "decoder", DEVICE, lang_filter="ko"
    )
    probing_results["qwen3_ko"] = probe_per_layer(feat_qwen_ko, labels_qwen_ko)

    # ── Qwen3-0.6B (EN only) ────────────────────────────────────────
    # Mean surprisal p=0.008** for EN → EN에서도 되는가?
    feat_qwen_en, labels_qwen_en, ids_qwen_en = extract_hidden_states(
        sentences, "Qwen/Qwen3-0.6B", "decoder", DEVICE, lang_filter="en"
    )
    probing_results["qwen3_en"] = probe_per_layer(feat_qwen_en, labels_qwen_en)

    # ── 시각화 ─────────────────────────────────────────────────────
    plot_probing_results(probing_results, save_name="probing")
    plot_probing_detail(probing_results, save_name="probing_detail")
    print_probing_summary(probing_results)

    # ── Control 1: Permutation test ──────────────────────────────
    # 라벨을 셔플해서 100회 반복 → 실제 accuracy가 chance보다 높은지 확인
    print("\n" + "="*65)
    print("CONTROL: Permutation Test (100 shuffles)")
    print("="*65)

    np.random.seed(42)
    for model_key, feat, labels in [
        ("klue_bert", feat_klue, labels_klue),
        ("qwen3_ko",  feat_qwen_ko, labels_qwen_ko),
    ]:
        real_peak = probing_results[model_key]["accuracy"].max()
        perm_peaks = []
        for _ in range(100):
            shuf = np.random.permutation(labels)
            perm = probe_per_layer(feat, shuf)
            perm_peaks.append(perm["accuracy"].max())
        perm_peaks = np.array(perm_peaks)
        p_val = (perm_peaks >= real_peak).mean()
        print(f"  [{model_key}] real peak={real_peak:.2f}, "
              f"perm peak={perm_peaks.mean():.2f}+/-{perm_peaks.std():.2f}, "
              f"p={p_val:.3f}")

    # ── Control 2: PCA dimensionality ────────────────────────────
    # PCA(k)만으로 분리 가능? → 1이면 표면 특성(어휘/문체) 지배적
    from sklearn.decomposition import PCA as _PCA

    print("\n" + "="*65)
    print("CONTROL: PCA Dimensionality (klue_bert, layer 6)")
    print("="*65)

    X_mid = feat_klue[:, 6, :]  # middle layer
    for k in [1, 2, 5, 10, 50]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", _PCA(n_components=k)),
            ("clf", LogisticRegression(C=0.1, max_iter=1000)),
        ])
        scores = cross_val_score(pipe, X_mid, labels_klue, cv=LeaveOneOut(), scoring="accuracy")
        print(f"  PCA({k:3d}): LOO acc = {scores.mean():.2f}")
    print("\n  Interpretation: PCA(1)=100% → surface features (vocab/style) dominate.")
    print("  Dense quotes vs sparse daily descriptions are trivially separable.")
    print("  This does NOT invalidate Layer Delta or Surprisal findings —")
    print("  those measure *how* the model processes text, not *whether* it can classify.")
