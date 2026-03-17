#%% [markdown]
# # Step 1: Transformer Internal State Extraction
#
# 고밀도 vs 저밀도 문장에 대한 트랜스포머 내부 신호 추출
#
# **측정 대상 (4가지):**
# 1. Attention Entropy — head별 attention 분포의 Shannon entropy
# 2. Hidden State Norm — 층별 hidden state의 L2 norm
# 3. Layer-wise Representation Delta — 연속 층 간 변화량
# 4. Effective Rank — hidden state의 SVD 기반 유효 차원 수
#
# **모델:** Encoder (BERT) + Decoder (GPT-2) 비교

#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

#%%
@dataclass
class InternalSignals:
    """한 문장에 대한 트랜스포머 내부 신호 측정 결과"""
    text: str
    density_label: str
    lang: str
    model_name: str
    model_type: str  # "encoder" or "decoder"

    # 층별 측정값 (shape: [num_layers])
    attention_entropy: Optional[np.ndarray] = None       # 각 층의 평균 attention entropy
    hidden_state_norm: Optional[np.ndarray] = None       # 각 층의 평균 hidden state L2 norm
    layer_delta: Optional[np.ndarray] = None             # 연속 층 간 cosine distance
    effective_rank: Optional[np.ndarray] = None          # 각 층의 effective rank

    # attention 거리 분포 (고밀도 → 먼 토큰에도 attention?)
    mean_attention_distance: Optional[np.ndarray] = None  # 각 층의 평균 attention 거리

#%% [markdown]
# ## Core: Attention Entropy
#
# H(attention) = -Σ_j a_j · log(a_j)
#
# 직관: attention이 몇 개 토큰에 집중 → 낮은 entropy (정보가 local)
#         attention이 분산 → 높은 entropy (모든 토큰이 의미 있음)
#
# 가설: 고밀도 문장 → 모든 토큰이 의미적으로 중요 → attention entropy 높음

#%%
def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Attention weights → Shannon entropy per head per layer.

    Args:
        attention_weights: (num_layers, num_heads, seq_len, seq_len)
    Returns:
        entropy: (num_layers, num_heads) — 각 head의 평균 entropy (query 위치에 대해 평균)
    """
    # attention_weights[l, h, i, j] = head h가 query i에서 key j에 얼마나 attend하는지
    # entropy는 j 방향 (key) 에 대해 계산
    eps = 1e-10
    # BF16 → float32: log2가 BF16에서 정밀도 손실 + numpy 변환 불가
    attn = attention_weights.float() + eps  # log(0) 방지

    # H = -Σ_j a_{ij} · log(a_{ij}), 각 query position i에 대해
    entropy = -(attn * torch.log2(attn)).sum(dim=-1)  # (layers, heads, seq_len)
    assert entropy.shape == attention_weights.shape[:3]

    # query 위치에 대해 평균
    return entropy.mean(dim=-1)  # (layers, heads)

#%% [markdown]
# ## Core: Hidden State Analysis

#%%
def compute_hidden_state_metrics(
    hidden_states: tuple[torch.Tensor, ...],
) -> dict[str, np.ndarray]:
    """
    층별 hidden state에서 norm, delta, effective rank 계산.

    Args:
        hidden_states: tuple of (1, seq_len, hidden_dim) — 각 층의 출력
    Returns:
        dict with 'norm', 'delta', 'effective_rank' — 각각 (num_layers,)
    """
    num_layers = len(hidden_states)
    norms = []
    deltas = []
    eff_ranks = []

    for l in range(num_layers):
        h = hidden_states[l].squeeze(0)  # (seq_len, hidden_dim)
        assert h.ndim == 2

        # L2 norm: 각 토큰의 norm 평균
        norms.append(h.norm(dim=-1).mean().item())

        # Layer delta: 이전 층과의 cosine distance
        if l > 0:
            h_prev = hidden_states[l - 1].squeeze(0)
            cos_sim = nn.functional.cosine_similarity(h, h_prev, dim=-1).mean()
            deltas.append(1.0 - cos_sim.item())  # distance = 1 - similarity
        else:
            deltas.append(0.0)

        # Effective rank: SVD → singular values의 entropy 기반
        # Insight: 높은 effective rank = 정보가 더 많은 차원에 분산 = 복잡한 표현
        # SVD: MPS 미구현 + BF16 미지원 → float32 CPU로 fallback
        S = torch.linalg.svdvals(h.float().cpu())
        S_normalized = S / S.sum()
        eff_rank = torch.exp(-(S_normalized * torch.log(S_normalized + 1e-10)).sum()).item()
        eff_ranks.append(eff_rank)

    return {
        "norm": np.array(norms),
        "delta": np.array(deltas),
        "effective_rank": np.array(eff_ranks),
    }

#%% [markdown]
# ## Core: Attention Distance
#
# 각 attention head에서 query와 key 사이의 평균 거리
# 고밀도 → 먼 토큰에도 attention → 평균 거리 큼?

#%%
def compute_attention_distance(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Attention-weighted mean distance between query and key positions.

    Args:
        attention_weights: (num_layers, num_heads, seq_len, seq_len)
    Returns:
        mean_distance: (num_layers, num_heads)
    """
    seq_len = attention_weights.shape[-1]
    # BF16 → float32 for numeric stability and numpy compatibility
    attn = attention_weights.float()
    # position distance matrix: |i - j| for all (i, j)
    positions = torch.arange(seq_len, device=attention_weights.device).float()
    dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq_len, seq_len)

    # attention-weighted distance per query position
    weighted_dist = (attn * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    # (layers, heads, seq_len) → average over query positions
    return weighted_dist.mean(dim=-1)  # (layers, heads)

#%% [markdown]
# ## Extraction: Encoder (BERT)

#%%
def extract_encoder_signals(
    text: str,
    model_name: str = "klue/bert-base",
    device: str = "mps",
) -> InternalSignals:
    """BERT 계열 encoder에서 내부 신호 추출"""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # attn_implementation="eager": SDPA/FlashAttention은 output_attentions=True 미지원
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device).eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    attn_stack = torch.stack([a.squeeze(0) for a in outputs.attentions])  # (layers, heads, S, S)
    entropy = compute_attention_entropy(attn_stack)  # (layers, heads)
    attn_dist = compute_attention_distance(attn_stack)

    # hidden_states[0] = embedding layer (attention 없음) → 제외하고 맞춤
    hs_metrics = compute_hidden_state_metrics(outputs.hidden_states[1:])

    return InternalSignals(
        text=text,
        density_label="",
        lang="",
        model_name=model_name,
        model_type="encoder",
        attention_entropy=entropy.mean(dim=-1).cpu().numpy(),  # head 평균
        hidden_state_norm=hs_metrics["norm"],
        layer_delta=hs_metrics["delta"],
        effective_rank=hs_metrics["effective_rank"],
        mean_attention_distance=attn_dist.mean(dim=-1).cpu().numpy(),
    )

#%% [markdown]
# ## Extraction: Decoder (GPT-2)

#%%
def extract_decoder_signals(
    text: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
) -> InternalSignals:
    """Decoder-only LM에서 내부 신호 추출 (GPT-2, Qwen3 등)"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # attn_implementation="eager": SDPA/FlashAttention은 output_attentions=True를 미지원
    # → eager attention으로 강제해야 attention weights를 반환받을 수 있음
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, attn_implementation="eager"
    ).to(device).eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Causal attention: 하삼각 마스크 적용된 attention
    attn_stack = torch.stack([a.squeeze(0) for a in outputs.attentions])
    entropy = compute_attention_entropy(attn_stack)
    attn_dist = compute_attention_distance(attn_stack)

    hs_metrics = compute_hidden_state_metrics(outputs.hidden_states[1:])

    return InternalSignals(
        text=text,
        density_label="",
        lang="",
        model_name=model_name,
        model_type="decoder",
        attention_entropy=entropy.mean(dim=-1).cpu().numpy(),
        hidden_state_norm=hs_metrics["norm"],
        layer_delta=hs_metrics["delta"],
        effective_rank=hs_metrics["effective_rank"],
        mean_attention_distance=attn_dist.mean(dim=-1).cpu().numpy(),
    )

#%% [markdown]
# ## Visualization: Layer-wise Signal Comparison

#%%
def plot_layer_comparison(
    high_signals: list[InternalSignals],
    low_signals: list[InternalSignals],
    model_type: str = "encoder",
    save_dir: str = "../outputs",
):
    """고밀도 vs 저밀도: 층별 4가지 신호 비교"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics = [
        ("attention_entropy", "Attention Entropy (bits)", "높은 entropy = 분산된 attention"),
        ("hidden_state_norm", "Hidden State L2 Norm", "높은 norm = 더 많은 정보 인코딩"),
        ("layer_delta", "Layer-wise Delta (1 - cosine sim)", "높은 delta = 층간 큰 변화"),
        ("effective_rank", "Effective Rank", "높은 rank = 더 많은 차원 활용"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Internal Signals: High vs Low Density ({model_type})", fontsize=14)

    for idx, (attr, title, insight) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        h_vals = np.stack([getattr(s, attr) for s in high_signals if getattr(s, attr) is not None])
        l_vals = np.stack([getattr(s, attr) for s in low_signals if getattr(s, attr) is not None])

        if h_vals.size == 0 or l_vals.size == 0:
            ax.set_title(f"{title} (no data)")
            continue

        layers = np.arange(h_vals.shape[1])

        # Mean ± std
        ax.plot(layers, h_vals.mean(axis=0), "r-o", label="High density", markersize=4)
        ax.fill_between(layers,
                        h_vals.mean(0) - h_vals.std(0),
                        h_vals.mean(0) + h_vals.std(0), alpha=0.2, color="red")

        ax.plot(layers, l_vals.mean(axis=0), "b-s", label="Low density", markersize=4)
        ax.fill_between(layers,
                        l_vals.mean(0) - l_vals.std(0),
                        l_vals.mean(0) + l_vals.std(0), alpha=0.2, color="blue")

        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\n({insight})")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/step1_{model_type}_signals.png", dpi=150, bbox_inches="tight")
    plt.show()

#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    # Quick test with one pair
    high_text = "He who fights with monsters should look to it that he himself does not become a monster."
    low_text = "She walked to the store on the corner and bought a bottle of water and a sandwich."

    print("Extracting encoder signals...")
    high_enc = extract_encoder_signals(high_text)
    low_enc = extract_encoder_signals(low_text)

    print(f"\nEncoder — Attention Entropy (layer mean):")
    print(f"  High: {high_enc.attention_entropy.mean():.4f}")
    print(f"  Low:  {low_enc.attention_entropy.mean():.4f}")

    print(f"\nEncoder — Effective Rank (layer mean):")
    print(f"  High: {high_enc.effective_rank.mean():.4f}")
    print(f"  Low:  {low_enc.effective_rank.mean():.4f}")

    plot_layer_comparison([high_enc], [low_enc], model_type="encoder")
