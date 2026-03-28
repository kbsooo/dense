#%% [markdown]
# # Step 2: Diffusion-like Crystallization Analysis
#
# **핵심 가설 (H-Diffusion):**
# 고밀도 문장은 iterative denoising 과정에서 "결정화"에 더 많은 스텝이 필요하다.
#
# **방법: Masked LM as Discrete Diffusion**
# D3PM (Austin et al., 2021): absorbing-state discrete diffusion ≈ masked LM iterative decoding
# MDLM도 본질적으로 이 구조. 우리는 klue/bert-base를 discrete denoiser로 사용한다.
#
# Process:
#   1. 전체 토큰을 [MASK]로 치환 (= fully noised state)
#   2. BERT에 입력 → 각 위치의 logit/probability 관찰
#   3. 가장 confident한 위치부터 unmask (= denoising step)
#   4. 반복 → confidence/entropy trajectory 기록
#
# **측정 신호:**
# - confidence trajectory: 각 step에서 각 토큰의 max probability
# - entropy trajectory: 각 step에서의 vocabulary 분포 entropy
# - crystallization step: 토큰이 최종값으로 확정되는 시점
# - convergence area: entropy curve 아래 면적 (= 총 불확실성)
#
# **모델:**
# - klue/bert-base → 한국어 (Step 1에서 Layer Delta p=0.032* 나온 모델)
# - bert-base-uncased → 영어

#%%
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy import stats

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps"

#%% [markdown]
# ## 1. Iterative Unmasking Engine

#%%
@dataclass
class CrystallizationResult:
    """한 문장의 denoising 궤적"""
    text: str
    sent_id: str
    density: str
    lang: str
    model_name: str

    tokens: list[str]                       # original tokens
    n_steps: int                            # total unmasking steps
    confidence_traj: np.ndarray             # (n_steps, seq_len) — max prob per position
    entropy_traj: np.ndarray                # (n_steps, seq_len) — vocab entropy per position
    crystallization_step: np.ndarray        # (seq_len,) — step where token first matches final
    unmask_order: np.ndarray                # (seq_len,) — order in which positions were unmasked

    # summary scalars
    mean_crystallization: float             # mean normalized crystallization step
    convergence_area: float                 # area under mean entropy curve


def iterative_unmask(
    text: str,
    model,
    tokenizer,
    device: str = DEVICE,
    unmask_per_step: int = 1,
) -> dict:
    """
    BERT를 discrete diffusion denoiser로 사용한 iterative unmasking.

    Insight: 이것이 MDLM의 핵심 과정과 동치.
    absorbing state (MASK) → predicted token transition.
    가장 confident한 위치부터 unmask하는 것은
    MDLM의 "argmax sampling with confidence ordering"과 동일.
    """
    # 원본 토큰화
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)  # (1, seq_len) — includes [CLS], [SEP]
    original_tokens = input_ids.squeeze(0).clone()

    # [CLS], [SEP] 위치 제외 — 실제 content 토큰만 마스킹
    special_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    special_mask[0] = True   # [CLS]
    special_mask[-1] = True  # [SEP]
    content_positions = (~special_mask).nonzero(as_tuple=True)[0]
    n_content = len(content_positions)

    # 전체 content를 [MASK]로
    masked_ids = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    masked_ids[0, content_positions] = mask_token_id

    # 궤적 기록
    n_steps = (n_content + unmask_per_step - 1) // unmask_per_step
    seq_len = input_ids.shape[1]
    confidence_traj = np.zeros((n_steps + 1, seq_len))
    entropy_traj = np.zeros((n_steps + 1, seq_len))
    unmask_order = np.full(seq_len, -1)  # -1 = special token (not unmasked)

    still_masked = set(content_positions.tolist())
    step = 0

    while still_masked and step <= n_steps:
        with torch.no_grad():
            outputs = model(masked_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        probs = logits.float().softmax(dim=-1)  # (seq_len, V)

        # confidence & entropy for ALL positions at this step
        max_probs = probs.max(dim=-1).values.cpu().numpy()  # (seq_len,)
        eps = 1e-10
        ent = -(probs * torch.log2(probs + eps)).sum(dim=-1).cpu().numpy()

        confidence_traj[step] = max_probs
        entropy_traj[step] = ent

        # 아직 masked인 위치 중 가장 confident한 것부터 unmask
        masked_list = sorted(still_masked)
        if not masked_list:
            break

        masked_confidences = [(pos, max_probs[pos]) for pos in masked_list]
        masked_confidences.sort(key=lambda x: x[1], reverse=True)

        # unmask top-k
        to_unmask = masked_confidences[:unmask_per_step]
        for pos, conf in to_unmask:
            predicted_token = probs[pos].argmax().item()
            masked_ids[0, pos] = predicted_token
            still_masked.discard(pos)
            unmask_order[pos] = step

        step += 1

    # 마지막 step의 상태도 기록
    if step <= n_steps:
        with torch.no_grad():
            outputs = model(masked_ids)
            logits = outputs.logits[0]
        probs = logits.float().softmax(dim=-1)
        confidence_traj[step] = probs.max(dim=-1).values.cpu().numpy()
        entropy_traj[step] = -(probs * torch.log2(probs + eps)).sum(dim=-1).cpu().numpy()

    # trim to actual steps
    actual_steps = step + 1
    confidence_traj = confidence_traj[:actual_steps]
    entropy_traj = entropy_traj[:actual_steps]

    # 결정화 시점: 각 content 토큰이 최종값과 처음으로 일치하는 step
    final_tokens = masked_ids.squeeze(0).cpu()
    crystallization = np.full(seq_len, actual_steps - 1)
    for pos in content_positions.tolist():
        final_tok = final_tokens[pos].item()
        for s in range(actual_steps):
            # 해당 step에서의 predicted token 확인 (argmax of logits)
            # 간접 지표: unmask된 시점 = 결정화 시점
            if unmask_order[pos] >= 0 and s >= unmask_order[pos]:
                crystallization[pos] = unmask_order[pos]
                break

    # token strings
    token_strs = tokenizer.convert_ids_to_tokens(original_tokens.cpu().tolist())

    # convergence area (mean entropy curve 아래 면적)
    mean_ent_curve = entropy_traj[:, content_positions.cpu().numpy()].mean(axis=1)
    conv_area = float(np.trapz(mean_ent_curve, dx=1.0 / max(actual_steps - 1, 1)))

    # normalized mean crystallization (content tokens only)
    content_cryst = crystallization[content_positions.cpu().numpy()]
    mean_cryst = float(content_cryst.mean() / max(actual_steps - 1, 1))

    return {
        "tokens": token_strs,
        "n_steps": actual_steps,
        "confidence_traj": confidence_traj,
        "entropy_traj": entropy_traj,
        "crystallization_step": crystallization,
        "unmask_order": unmask_order,
        "mean_crystallization": mean_cryst,
        "convergence_area": conv_area,
        "content_positions": content_positions.cpu().numpy(),
    }


#%% [markdown]
# ## 2. 전체 문장에 대해 추출

#%%
def run_crystallization(
    sentences: list[dict],
    model_name: str,
    device: str = DEVICE,
    lang_filter: str | None = None,
) -> list[dict]:
    """모든 문장에 대해 crystallization 분석"""
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    if lang_filter:
        sentences = [s for s in sentences if s["lang"] == lang_filter]

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, attn_implementation="eager"
    ).to(device).eval()

    results = []
    for s in sentences:
        print(f"  [{s['id']}] {s['text'][:40]}...")
        res = iterative_unmask(s["text"], model, tokenizer, device)
        res["sent_id"] = s["id"]
        res["density"] = s["density"]
        res["lang"] = s["lang"]
        res["model_name"] = model_name
        results.append(res)

    del model
    return results


#%% [markdown]
# ## 3. 시각화

#%%
def plot_denoising_heatmaps(results: list[dict], model_label: str):
    """고밀도 vs 저밀도: denoising entropy heatmap 비교 (대표 1쌍)"""
    high = [r for r in results if r["density"] == "high"]
    low  = [r for r in results if r["density"] == "low"]

    if not high or not low:
        print("No data for heatmap")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(
        f"Denoising Trajectory — {model_label}\n"
        "Iterative unmasking: [MASK] → predicted tokens (by confidence order)",
        fontsize=11, fontweight="bold"
    )

    for col, (example, dlabel) in enumerate([
        (high[0], f"High density [{high[0]['sent_id']}]"),
        (low[0], f"Low density [{low[0]['sent_id']}]"),
    ]):
        cp = example["content_positions"]

        # Entropy heatmap (content tokens only)
        ent = example["entropy_traj"][:, cp]  # (steps, content_len)
        ax = axes[0, col]
        im = ax.imshow(ent, aspect="auto", cmap="hot_r", origin="lower")
        ax.set_title(f"{dlabel}\nEntropy (bits)")
        ax.set_xlabel("Content Token Position")
        ax.set_ylabel("Denoising Step")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Confidence heatmap
        conf = example["confidence_traj"][:, cp]
        ax = axes[1, col]
        im = ax.imshow(conf, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"{dlabel}\nConfidence (max prob)")
        ax.set_xlabel("Content Token Position")
        ax.set_ylabel("Denoising Step")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / f"step2_heatmap_{model_label.replace('/', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


def plot_convergence_comparison(results: list[dict], model_label: str):
    """고밀도 vs 저밀도: entropy 수렴 곡선 + 결정화 분포"""
    high = [r for r in results if r["density"] == "high"]
    low  = [r for r in results if r["density"] == "low"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Crystallization Analysis — {model_label}",
        fontsize=12, fontweight="bold"
    )

    # ── (1) Mean entropy convergence curve ──────────────────────────
    ax = axes[0]
    for group, color, label in [(high, "red", "High"), (low, "blue", "Low")]:
        # interpolate to common step axis (sentences have different n_steps)
        n_interp = 50
        interp_curves = []
        for r in group:
            cp = r["content_positions"]
            mean_ent = r["entropy_traj"][:, cp].mean(axis=1)  # (n_steps,)
            x_orig = np.linspace(0, 1, len(mean_ent))
            x_new = np.linspace(0, 1, n_interp)
            interp_curves.append(np.interp(x_new, x_orig, mean_ent))
        curves = np.stack(interp_curves)
        ax.plot(np.linspace(0, 1, n_interp), curves.mean(0),
                color=color, lw=2, label=f"{label} (n={len(group)})")
        ax.fill_between(np.linspace(0, 1, n_interp),
                        curves.mean(0) - curves.std(0),
                        curves.mean(0) + curves.std(0),
                        alpha=0.15, color=color)

    ax.set_xlabel("Denoising Progress (0=masked, 1=unmasked)")
    ax.set_ylabel("Mean Token Entropy (bits)")
    ax.set_title("Entropy Convergence")
    ax.legend()
    ax.grid(alpha=0.25)

    # ── (2) Mean crystallization ─────────────────────────────────────
    ax = axes[1]
    h_cryst = [r["mean_crystallization"] for r in high]
    l_cryst = [r["mean_crystallization"] for r in low]

    bp = ax.boxplot([h_cryst, l_cryst], tick_labels=["High", "Low"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#ffaaaa")
    bp["boxes"][1].set_facecolor("#aaaaff")
    ax.scatter([1]*len(h_cryst), h_cryst, color="red",  alpha=0.7, zorder=3, s=50)
    ax.scatter([2]*len(l_cryst), l_cryst, color="blue", alpha=0.7, zorder=3, s=50)

    if len(h_cryst) >= 2 and len(l_cryst) >= 2:
        _, p = stats.mannwhitneyu(h_cryst, l_cryst, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"ns (p={p:.3f})"
        ax.text(0.95, 0.95, sig, transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color="darkred" if "*" in sig else "gray")

    ax.set_ylabel("Normalized Crystallization Step")
    ax.set_title("Mean Crystallization\n(higher = later convergence)")
    ax.grid(alpha=0.25, axis="y")

    # ── (3) Convergence area ─────────────────────────────────────────
    ax = axes[2]
    h_area = [r["convergence_area"] for r in high]
    l_area = [r["convergence_area"] for r in low]

    bp = ax.boxplot([h_area, l_area], tick_labels=["High", "Low"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#ffaaaa")
    bp["boxes"][1].set_facecolor("#aaaaff")
    ax.scatter([1]*len(h_area), h_area, color="red",  alpha=0.7, zorder=3, s=50)
    ax.scatter([2]*len(l_area), l_area, color="blue", alpha=0.7, zorder=3, s=50)

    if len(h_area) >= 2 and len(l_area) >= 2:
        _, p = stats.mannwhitneyu(h_area, l_area, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"ns (p={p:.3f})"
        ax.text(0.95, 0.95, sig, transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color="darkred" if "*" in sig else "gray")

    ax.set_ylabel("Convergence Area (total uncertainty)")
    ax.set_title("Total Denoising Difficulty\n(higher = harder to crystallize)")
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / f"step2_convergence_{model_label.replace('/', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out.name}")


#%% [markdown]
# ## 4. Summary

#%%
def print_crystallization_summary(results_by_model: dict[str, list[dict]]):
    print("\n" + "="*65)
    print("STEP 2: CRYSTALLIZATION SUMMARY")
    print("="*65)

    for model_label, results in results_by_model.items():
        high = [r for r in results if r["density"] == "high"]
        low  = [r for r in results if r["density"] == "low"]

        if len(high) < 2 or len(low) < 2:
            continue

        print(f"\n[{model_label}] n_high={len(high)}, n_low={len(low)}")

        for metric, label in [
            ("mean_crystallization", "Mean crystallization"),
            ("convergence_area",    "Convergence area"),
        ]:
            h_vals = [r[metric] for r in high]
            l_vals = [r[metric] for r in low]
            _, p = stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            direction = "High > Low" if np.mean(h_vals) > np.mean(l_vals) else "High < Low"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {label:<25} {direction}  "
                  f"H={np.mean(h_vals):.4f} L={np.mean(l_vals):.4f}  p={p:.4f} {sig}")


#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    with open(DATA_DIR / "seed_sentences.json") as f:
        sentences = json.load(f)["sentences"]

    all_results: dict[str, list[dict]] = {}

    # ── klue/bert-base (KO) ─────────────────────────────────────────
    # Step 1에서 Layer Delta p=0.032*를 보인 모델 — diffusion 관점에서도 밀도를 느끼는가?
    ko_results = run_crystallization(
        sentences, "klue/bert-base", DEVICE, lang_filter="ko"
    )
    all_results["klue/bert-base (KO)"] = ko_results

    # ── bert-base-uncased (EN) ──────────────────────────────────────
    en_results = run_crystallization(
        sentences, "bert-base-uncased", DEVICE, lang_filter="en"
    )
    all_results["bert-base-uncased (EN)"] = en_results

    # ── 시각화 ───────────────────────────────────────────────────────
    for model_label, results in all_results.items():
        plot_denoising_heatmaps(results, model_label)
        plot_convergence_comparison(results, model_label)

    # ── 요약 ─────────────────────────────────────────────────────────
    print_crystallization_summary(all_results)
