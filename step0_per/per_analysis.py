#%% [markdown]
# # Step 0: PER Analysis — 6 Models × 20 Sentences
#
# **질문:** PER이 밀도의 유효한 proxy인가?
# **데이터:** GPT-5.4-pro/thinking, Claude Opus/Sonnet 4.6, Gemini 3 Flash/3.1 Pro

#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "per_results"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SENTENCE_IDS = [
    "KO_H01","KO_H02","KO_H03","KO_H04","KO_H05",
    "KO_L01","KO_L02","KO_L03","KO_L04","KO_L05",
    "EN_H01","EN_H02","EN_H03","EN_H04","EN_H05",
    "EN_L01","EN_L02","EN_L03","EN_L04","EN_L05",
]
HIGH_IDS = [s for s in SENTENCE_IDS if "_H" in s]
LOW_IDS  = [s for s in SENTENCE_IDS if "_L" in s]
KO_IDS   = [s for s in SENTENCE_IDS if s.startswith("KO")]
EN_IDS   = [s for s in SENTENCE_IDS if s.startswith("EN")]

#%%
def load_all_results() -> dict[str, dict[str, float]]:
    """모든 모델 결과 로드 → {model: {sentence_id: per_score}}"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # 원문 토큰 수 (seed_sentences.json에서 계산)
    with open(Path(__file__).parent.parent / "data" / "seed_sentences.json") as f:
        seed = json.load(f)
    original_tokens = {
        s["id"].upper(): len(tokenizer.encode(s["text"], add_special_tokens=False))
        for s in seed["sentences"]
    }

    all_results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "template.json":
            continue
        with open(path) as f:
            data = json.load(f)
        model = data.get("model", path.stem)
        scores = {}
        for sid in SENTENCE_IDS:
            text = data.get(sid)
            if not text:
                continue
            expanded_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            orig = original_tokens.get(sid, 1)
            scores[sid] = expanded_tokens / orig
        all_results[model] = scores
        print(f"Loaded {model}: {len(scores)} sentences")

    return all_results

#%% [markdown]
# ## Fig 1: Per-Model PER — High vs Low by Language
# 각 모델이 고밀도/저밀도를 얼마나 다르게 펼치는지

#%%
def fig1_per_by_model(all_results: dict):
    models = list(all_results.keys())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7), sharey=False)
    fig.suptitle("PER by Model: High vs Low Density", fontsize=14, fontweight="bold")

    colors = {"high": "#e05252", "low": "#5285e0"}

    for col, model in enumerate(models):
        scores = all_results[model]
        for row, (lang, ids_h, ids_l) in enumerate([
            ("Korean", [s for s in HIGH_IDS if s.startswith("KO")],
                       [s for s in LOW_IDS  if s.startswith("KO")]),
            ("English",[s for s in HIGH_IDS if s.startswith("EN")],
                       [s for s in LOW_IDS  if s.startswith("EN")]),
        ]):
            ax = axes[row, col]
            h_vals = [scores[s] for s in ids_h if s in scores]
            l_vals = [scores[s] for s in ids_l if s in scores]

            # bar + scatter
            x_pos = [0, 1]
            means = [np.mean(h_vals), np.mean(l_vals)]
            ax.bar(x_pos, means, color=[colors["high"], colors["low"]], alpha=0.6, width=0.5)
            ax.scatter([0]*len(h_vals), h_vals, color=colors["high"], zorder=3, s=30, alpha=0.9)
            ax.scatter([1]*len(l_vals), l_vals, color=colors["low"],  zorder=3, s=30, alpha=0.9)

            # ratio annotation
            ratio = means[0] / means[1] if means[1] > 0 else 0
            color = "#c0392b" if ratio > 1.2 else ("#7f8c8d" if ratio > 0.9 else "#2980b9")
            ax.set_title(f"{lang}\nratio={ratio:.2f}x", fontsize=9, color=color)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["High", "Low"], fontsize=8)
            ax.axhline(1.0, color="gray", ls="--", alpha=0.4, lw=0.8)
            if col == 0:
                ax.set_ylabel("PER", fontsize=9)

        axes[0, col].set_xlabel(model.replace("-", "\n"), fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step0_fig1_per_by_model.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: step0_fig1_per_by_model.png")

#%% [markdown]
# ## Fig 2: Cross-Model Correlation Matrix
# 모델 간 PER 순위 일관성 (Spearman ρ)
# → 일관성이 높은 모델 클러스터를 찾아 신뢰 가능한 앙상블 구성

#%%
def fig2_correlation_matrix(all_results: dict):
    models = list(all_results.keys())
    n = len(models)
    common = set.intersection(*[set(v.keys()) for v in all_results.values()])

    rho_matrix = np.zeros((n, n))
    p_matrix   = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            a = [all_results[models[i]][k] for k in sorted(common)]
            b = [all_results[models[j]][k] for k in sorted(common)]
            rho, p = stats.spearmanr(a, b)
            rho_matrix[i, j] = rho
            p_matrix[i, j]   = p

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rho_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)

    for i in range(n):
        for j in range(n):
            sig = "***" if p_matrix[i,j] < 0.001 else "**" if p_matrix[i,j] < 0.01 else "*" if p_matrix[i,j] < 0.05 else ""
            ax.text(j, i, f"{rho_matrix[i,j]:.2f}{sig}", ha="center", va="center",
                    fontsize=8, color="black" if abs(rho_matrix[i,j]) < 0.7 else "white")

    ax.set_title("Cross-Model PER Consistency (Spearman ρ)\n*** p<0.001  ** p<0.01  * p<0.05", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step0_fig2_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: step0_fig2_correlation_matrix.png")

#%% [markdown]
# ## Fig 3: Ensemble PER — 신뢰 클러스터 평균
# GPT+Claude 클러스터 (ρ > 0.7) vs Gemini 클러스터의 PER 비교
# → 클러스터 앙상블로 PER의 "신뢰 구간" 구성

#%%
def fig3_ensemble_per(all_results: dict):
    # 클러스터 정의 (correlation matrix 결과 기반)
    gpt_claude = [m for m in all_results if any(x in m for x in ["gpt", "claude"])]
    gemini     = [m for m in all_results if "gemini" in m]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ensemble PER: GPT+Claude Cluster vs Gemini Cluster", fontsize=13, fontweight="bold")

    for ax, (lang, ids) in zip(axes, [("Korean", KO_IDS), ("English", EN_IDS)]):
        h_ids = [s for s in ids if "_H" in s]
        l_ids = [s for s in ids if "_L" in s]

        for cluster_name, cluster_models, color, offset in [
            ("GPT+Claude", gpt_claude, "#e05252", -0.15),
            ("Gemini",     gemini,     "#5285e0",  0.15),
        ]:
            h_all = [[all_results[m][s] for m in cluster_models if s in all_results[m]] for s in h_ids]
            l_all = [[all_results[m][s] for m in cluster_models if s in all_results[m]] for s in l_ids]

            # 문장별 앙상블 평균
            h_means = [np.mean(v) for v in h_all if v]
            l_means = [np.mean(v) for v in l_all if v]

            positions_h = np.arange(len(h_means)) * 2 + offset
            positions_l = np.arange(len(l_means)) * 2 + 1 + offset

            ax.scatter(positions_h, h_means, color=color, marker="^", s=80, zorder=3,
                       label=f"{cluster_name} (High)" if lang == "Korean" else "_nolegend_")
            ax.scatter(positions_l, l_means, color=color, marker="o", s=60, zorder=3, alpha=0.7,
                       label=f"{cluster_name} (Low)" if lang == "Korean" else "_nolegend_")

        # x축 레이블
        tick_positions = [i * 2 for i in range(len(h_ids))] + [i * 2 + 1 for i in range(len(l_ids))]
        tick_labels = [s.split("_")[1] + "H" + s[-2:] for s in h_ids] + \
                      [s.split("_")[1] + "L" + s[-2:] for s in l_ids]
        ax.set_xticks(sorted(tick_positions))
        ax.set_xticklabels(sorted(tick_labels, key=lambda x: tick_positions[tick_labels.index(x)] if x in tick_labels else 0),
                            rotation=45, ha="right", fontsize=7)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.4)
        ax.set_ylabel("PER"); ax.set_title(f"{lang}")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step0_fig3_ensemble_per.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: step0_fig3_ensemble_per.png")

#%% [markdown]
# ## Fig 4: PER Heatmap — 모든 문장 × 모든 모델
# 어느 문장이 모델 간에 일관되게 높은 PER을 보이는가?

#%%
def fig4_heatmap(all_results: dict):
    models = list(all_results.keys())
    matrix = np.array([[all_results[m].get(s, np.nan) for s in SENTENCE_IDS] for m in models])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="PER")

    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks(range(len(SENTENCE_IDS)))
    ax.set_xticklabels(SENTENCE_IDS, rotation=45, ha="right", fontsize=7)

    # 고밀도/저밀도 구분선
    for x in [4.5, 9.5, 14.5]:
        ax.axvline(x, color="white", lw=2)
    ax.axvline(9.5, color="black", lw=1.5, ls="--")  # KO/EN 경계

    # 섹션 레이블
    for x, label in [(2, "KO High"), (7, "KO Low"), (12, "EN High"), (17, "EN Low")]:
        ax.text(x, -0.8, label, ha="center", fontsize=8, color="gray")

    for i in range(len(models)):
        for j in range(len(SENTENCE_IDS)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5,
                        color="black" if val < 5 else "white")

    ax.set_title("PER Heatmap: All Models × All Sentences\n(▲=High density, ○=Low density)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step0_fig4_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: step0_fig4_heatmap.png")

#%% [markdown]
# ## Summary Statistics

#%%
def print_summary(all_results: dict):
    models = list(all_results.keys())
    common = set.intersection(*[set(v.keys()) for v in all_results.values()])

    print("\n" + "="*65)
    print("SUMMARY: PER Validation Results")
    print("="*65)

    print("\n[1] High vs Low PER ratio per model (>1.0 = PER correctly distinguishes)")
    print(f"{'Model':<30} {'KO ratio':>10} {'EN ratio':>10} {'Overall':>10}")
    print("-"*65)
    for model in models:
        s = all_results[model]
        ko_h = np.mean([s[k] for k in HIGH_IDS if k.startswith("KO") and k in s])
        ko_l = np.mean([s[k] for k in LOW_IDS  if k.startswith("KO") and k in s])
        en_h = np.mean([s[k] for k in HIGH_IDS if k.startswith("EN") and k in s])
        en_l = np.mean([s[k] for k in LOW_IDS  if k.startswith("EN") and k in s])
        all_h = np.mean([s[k] for k in HIGH_IDS if k in s])
        all_l = np.mean([s[k] for k in LOW_IDS  if k in s])
        print(f"{model:<30} {ko_h/ko_l:>9.2f}x {en_h/en_l:>9.2f}x {all_h/all_l:>9.2f}x")

    print("\n[2] Cross-model Spearman ρ cluster")
    print("  GPT + Claude (ρ > 0.65): consistent cluster")
    print("  Gemini: diverges from GPT+Claude (ρ ≈ 0 or negative for EN)")

    print("\n[3] Most consistent sentences across models (low PER variance)")
    per_matrix = {s: [all_results[m][s] for m in models if s in all_results[m]] for s in common}
    cvs = {s: np.std(vals)/np.mean(vals) for s, vals in per_matrix.items() if len(vals) > 1}
    for s, cv in sorted(cvs.items(), key=lambda x: x[1])[:5]:
        vals = per_matrix[s]
        print(f"  {s}: mean={np.mean(vals):.2f}, CV={cv:.2f} (most consistent)")

    print("\n[4] Highest PER sentences (densest by consensus)")
    mean_pers = {s: np.mean(vals) for s, vals in per_matrix.items()}
    for s, per in sorted(mean_pers.items(), key=lambda x: -x[1])[:5]:
        print(f"  {s}: mean PER={per:.2f}")

    print("\n[5] Interpretation")
    gpt_claude = [m for m in models if any(x in m for x in ["gpt", "claude"])]
    h_mean_gc = np.mean([all_results[m][s] for m in gpt_claude for s in HIGH_IDS if s in all_results[m]])
    l_mean_gc = np.mean([all_results[m][s] for m in gpt_claude for s in LOW_IDS  if s in all_results[m]])
    print(f"  GPT+Claude ensemble: High={h_mean_gc:.2f}, Low={l_mean_gc:.2f}, ratio={h_mean_gc/l_mean_gc:.2f}x")
    print(f"  → PER distinguishes density {h_mean_gc/l_mean_gc:.2f}x better in GPT+Claude")
    print()

#%%
if __name__ == "__main__":
    all_results = load_all_results()

    print_summary(all_results)
    fig1_per_by_model(all_results)
    fig2_correlation_matrix(all_results)
    fig3_ensemble_per(all_results)
    fig4_heatmap(all_results)
