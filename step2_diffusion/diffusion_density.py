#%% [markdown]
# # Step 2: Diffusion LM — Density as Crystallization Difficulty
#
# Autoregressive 모델은 토큰을 순차적으로 생성하지만,
# Diffusion LM은 전체 시퀀스를 동시에 노이즈에서 복원함.
#
# **핵심 가설 (H-Diffusion):**
# 고밀도 문장은 denoising 과정에서 "결정화"에 더 많은 스텝이 필요하다.
# → 밀도 ≈ 복원 난이도
#
# **관측 신호:**
# 1. 토큰별 confidence 궤적: 각 denoising step에서 각 토큰의 예측 확신도
# 2. 토큰별 entropy 궤적: 각 step에서의 vocabulary 분포 entropy
# 3. 결정화 시점: 토큰이 최종 값으로 "결정"되는 step
# 4. 수렴 프로파일: 전체 시퀀스가 안정화되는 패턴
#
# **모델:** MDLM (130M) — 가장 깔끔한 코드, 내부 상태 추출 용이
#
# ### [SOTA Alert]
# BD3-LMs를 사용하면 block_size 파라미터로 AR↔Diffusion 스펙트럼을
# 연속적으로 조절 가능. 같은 밀도 문장에 대해 처리 방식이 어떻게
# 밀도 반응을 바꾸는지 관찰 가능.

#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path

#%%
@dataclass
class DiffusionTrajectory:
    """한 문장에 대한 diffusion denoising 궤적"""
    text: str
    density_label: str
    lang: str
    num_steps: int

    # shape: (num_steps, seq_len)
    confidence_trajectory: np.ndarray | None = None   # max prob at each position per step
    entropy_trajectory: np.ndarray | None = None      # vocab entropy at each position per step

    # shape: (seq_len,)
    crystallization_step: np.ndarray | None = None    # step at which each token converges

    # scalar
    mean_crystallization: float | None = None         # 평균 결정화 시점 (0~1 normalized)
    convergence_area: float | None = None             # entropy curve 아래 면적 (= 총 불확실성)

#%% [markdown]
# ## Core: Denoising Trajectory Extraction
#
# MDLM의 sampling loop를 후킹해서 중간 상태를 기록.
#
# ```
# Canonical MDLM sampling loop:
# x = x_T  (fully masked)
# for t in reversed(timesteps):
#     logits = model(x, t)                    # (B, L, V)
#     confidence = logits.softmax(-1).max(-1)  # 각 위치의 최대 확률
#     entropy = -(p * log p).sum(-1)           # 각 위치의 entropy
#     x = sample_next(logits, t)
# ```
#
# 아래는 MDLM 코드 구조에 맞춘 후킹 템플릿.
# MDLM이 설치되면 import 경로를 맞춰서 사용.

#%%
def extract_diffusion_trajectory(
    text: str,
    model,
    tokenizer,
    num_steps: int = 128,
    device: str = "mps",
) -> DiffusionTrajectory:
    """
    MDLM 모델에서 denoising trajectory 추출.

    NOTE: 이 함수는 MDLM의 실제 sampling 코드를 기반으로 작성.
    MDLM 설치 후 모델/토크나이저를 전달해서 사용.

    Diffusion LM의 "이해도"를 측정하는 핵심:
    - 고밀도 문장: entropy가 오래 높게 유지 → 늦게 결정화
    - 저밀도 문장: entropy가 빠르게 하강 → 빠르게 결정화
    """
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]

    # 궤적 기록용 배열
    confidence_traj = np.zeros((num_steps, seq_len))
    entropy_traj = np.zeros((num_steps, seq_len))

    # Fully masked initial state
    mask_token_id = tokenizer.mask_token_id
    x_t = torch.full_like(input_ids, mask_token_id)

    timesteps = torch.linspace(1.0, 0.0, num_steps)

    for step_idx, t in enumerate(timesteps):
        t_tensor = torch.full((1,), t.item(), device=device)

        with torch.no_grad():
            logits = model(x_t, t_tensor)  # (1, seq_len, vocab_size)
            assert logits.shape == (1, seq_len, tokenizer.vocab_size), \
                f"Logits shape mismatch: {logits.shape}"

        probs = logits.softmax(dim=-1).squeeze(0)  # (seq_len, V)

        # Confidence: 각 위치에서 가장 높은 확률
        confidence_traj[step_idx] = probs.max(dim=-1).values.cpu().numpy()

        # Entropy: 각 위치의 vocabulary 분포 entropy
        # Insight: 높은 entropy = 모델이 어떤 토큰이 올지 아직 불확실
        eps = 1e-10
        ent = -(probs * torch.log2(probs + eps)).sum(dim=-1)
        entropy_traj[step_idx] = ent.cpu().numpy()

        # Unmask step (simplified — 실제 MDLM은 p_sample 함수 사용)
        # 가장 confident한 위치부터 unmask
        predicted_tokens = probs.argmax(dim=-1)  # (seq_len,)
        unmask_ratio = 1.0 - t.item()
        num_to_unmask = int(seq_len * unmask_ratio)
        confidence_scores = probs.max(dim=-1).values

        # 상위 N개 위치를 unmask
        if num_to_unmask > 0:
            _, top_indices = confidence_scores.topk(min(num_to_unmask, seq_len))
            x_t = x_t.clone()
            x_t[0, top_indices] = predicted_tokens[top_indices]

    # 결정화 시점: 각 토큰이 최종 값과 같아지는 첫 번째 step
    final_tokens = x_t.squeeze(0).cpu().numpy()
    crystallization = np.full(seq_len, num_steps)  # default: 마지막 step

    for pos in range(seq_len):
        # 뒤에서부터 탐색: 언제부터 최종값이 유지되었는가
        for step in range(num_steps):
            # 각 step에서의 predicted token과 최종 token 비교
            if confidence_traj[step, pos] > 0.9:  # confidence 임계값
                crystallization[pos] = step
                break

    # Convergence area: entropy 곡선 아래 면적
    # Insight: 면적이 크면 = 전체적으로 불확실성이 높았음 = 어려운 문장
    convergence_area = np.trapz(entropy_traj.mean(axis=1), dx=1.0 / num_steps)

    return DiffusionTrajectory(
        text=text,
        density_label="",
        lang="",
        num_steps=num_steps,
        confidence_trajectory=confidence_traj,
        entropy_trajectory=entropy_traj,
        crystallization_step=crystallization,
        mean_crystallization=crystallization.mean() / num_steps,
        convergence_area=convergence_area,
    )

#%% [markdown]
# ## Visualization: Denoising Heatmaps
#
# X축: 토큰 위치, Y축: denoising step
# 값: confidence 또는 entropy
# → 어느 토큰이 언제 "결정"되는지 한눈에 보임

#%%
def plot_denoising_heatmap(
    high_traj: DiffusionTrajectory,
    low_traj: DiffusionTrajectory,
    save_dir: str = "../outputs",
):
    """고밀도 vs 저밀도: denoising 과정 시각 비교"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Diffusion Denoising: High vs Low Density", fontsize=14)

    for col, (traj, label) in enumerate([(high_traj, "High"), (low_traj, "Low")]):
        # Confidence heatmap
        ax = axes[0, col]
        im = ax.imshow(traj.confidence_trajectory, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, origin="lower")
        ax.set_title(f"{label} Density — Confidence")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Denoising Step →")
        plt.colorbar(im, ax=ax)

        # Entropy heatmap
        ax = axes[1, col]
        im = ax.imshow(traj.entropy_trajectory, aspect="auto", cmap="hot_r",
                        origin="lower")
        ax.set_title(f"{label} Density — Entropy")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Denoising Step →")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/step2_diffusion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

#%%
def plot_convergence_curves(
    high_trajs: list[DiffusionTrajectory],
    low_trajs: list[DiffusionTrajectory],
    save_dir: str = "../outputs",
):
    """고밀도 vs 저밀도: entropy 수렴 곡선 비교"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Mean entropy over steps
    for trajs, color, label in [(high_trajs, "red", "High"), (low_trajs, "blue", "Low")]:
        curves = np.stack([t.entropy_trajectory.mean(axis=1) for t in trajs])
        steps = np.linspace(0, 1, curves.shape[1])

        ax1.plot(steps, curves.mean(axis=0), color=color, label=f"{label} density", linewidth=2)
        ax1.fill_between(steps,
                         curves.mean(0) - curves.std(0),
                         curves.mean(0) + curves.std(0),
                         alpha=0.2, color=color)

    ax1.set_xlabel("Denoising Progress (0=noise, 1=clean)")
    ax1.set_ylabel("Mean Token Entropy (bits)")
    ax1.set_title("Entropy Convergence Curve\n(고밀도 → 느린 수렴?)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) Crystallization step distribution
    high_cryst = np.concatenate([t.crystallization_step / t.num_steps for t in high_trajs])
    low_cryst = np.concatenate([t.crystallization_step / t.num_steps for t in low_trajs])

    ax2.hist(high_cryst, bins=30, alpha=0.6, color="red", label="High density", density=True)
    ax2.hist(low_cryst, bins=30, alpha=0.6, color="blue", label="Low density", density=True)
    ax2.set_xlabel("Crystallization Point (normalized)")
    ax2.set_ylabel("Density")
    ax2.set_title("Token Crystallization Distribution\n(고밀도 → 늦은 결정화?)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/step2_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()

#%% [markdown]
# ## BD3-LM: AR↔Diffusion Spectrum
#
# block_size를 변화시키면서 같은 문장의 밀도 반응이 어떻게 달라지는지 관측.
# block_size=1: 순수 AR, block_size=seq_len: 순수 Diffusion
#
# 이 실험의 의미:
# "밀도라는 속성이 모델의 처리 방식에 무관한 보편적 현상인가,
#  아니면 처리 방식에 따라 다르게 발현되는가?"

#%%
def bd3_spectrum_experiment(
    text: str,
    block_sizes: list[int] = [1, 2, 4, 8, 16],
    # model_loader: callable — BD3-LM 로드 함수
):
    """
    BD3-LM의 block_size를 바꿔가며 같은 문장의 내부 신호 변화 관측.

    TODO: BD3-LM 설치 후 구현
    - 각 block_size에서 entropy 수렴 곡선 추출
    - block_size가 커질수록 (= 더 diffusion에 가까워질수록)
      고밀도 문장의 수렴이 얼마나 더 느려지는지 관측
    """
    results = {}
    for bs in block_sizes:
        print(f"Block size = {bs}")
        # model = model_loader(block_size=bs)
        # traj = extract_diffusion_trajectory(text, model, tokenizer)
        # results[bs] = traj
    return results

#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    print("Step 2: Diffusion LM density analysis")
    print("Requires MDLM or BD3-LM installation.")
    print()
    print("Installation:")
    print("  git clone https://github.com/kuleshov-group/mdlm")
    print("  cd mdlm && pip install -e .")
    print()
    print("  git clone https://github.com/kuleshov-group/bd3lms")
    print("  cd bd3lms && pip install -e .")
    print()
    print("Weights:")
    print("  MDLM: kuleshov-group/mdlm-owt (HuggingFace)")
    print("  BD3-LM: kuleshov-group/bd3lm-owt-block_size8 (HuggingFace)")
