#%% [markdown]
# # Step 0: Paraphrase Expansion Ratio (PER) Validation
#
# **목표:** PER이 텍스트 밀도의 유효한 proxy인지 검증
#
# PER = (풀어쓴 문장의 토큰 수) / (원문 토큰 수)
# - 고밀도 문장 → PER 높음 (많이 풀어야 이해 가능)
# - 저밀도 문장 → PER ≈ 1 (이미 풀어져 있음)
#
# **추가 측정:** surprisal, dependency depth, token-level entropy
# → PER과의 상관관계로 밀도의 다차원 프로파일 구성

#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

#%%
@dataclass
class DensitySample:
    """하나의 문장에 대한 밀도 측정 결과"""
    id: str
    lang: str
    density_label: str  # "high" or "low" (human annotation)
    text: str
    source: str
    num_propositions_est: int

    # 측정값 (Step 0에서 채워짐)
    num_tokens: Optional[int] = None
    per_score: Optional[float] = None            # Paraphrase Expansion Ratio
    expanded_text: Optional[str] = None          # LLM이 풀어쓴 버전
    expanded_num_tokens: Optional[int] = None
    mean_surprisal: Optional[float] = None       # 토큰 평균 surprisal
    max_surprisal: Optional[float] = None
    surprisal_variance: Optional[float] = None   # UID 위반 정도

#%%
def load_seed_data(path: str = "../data/seed_sentences.json") -> list[DensitySample]:
    """시드 데이터 로드"""
    with open(Path(__file__).parent / path) as f:
        data = json.load(f)

    samples = []
    for s in data["sentences"]:
        samples.append(DensitySample(
            id=s["id"],
            lang=s["lang"],
            density_label=s["density"],
            text=s["text"],
            source=s["source"],
            num_propositions_est=s["num_propositions_est"],
        ))
    return samples

#%% [markdown]
# ## Part 1: Tokenization & Basic Stats
# 토큰 수 확인 — 고밀도/저밀도 쌍이 비슷한 토큰 수를 갖는지

#%%
from transformers import AutoTokenizer

def count_tokens(samples: list[DensitySample], model_name: str = "bert-base-multilingual-cased"):
    """각 문장의 토큰 수 계산 (multilingual tokenizer)"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for s in samples:
        tokens = tokenizer.encode(s.text, add_special_tokens=False)
        s.num_tokens = len(tokens)

    return samples

def print_token_stats(samples: list[DensitySample]):
    """고밀도 vs 저밀도 토큰 수 비교"""
    for lang in ["ko", "en"]:
        lang_samples = [s for s in samples if s.lang == lang]
        high = [s for s in lang_samples if s.density_label == "high"]
        low = [s for s in lang_samples if s.density_label == "low"]

        print(f"\n{'Korean' if lang == 'ko' else 'English'}:")
        print(f"  High density: mean={np.mean([s.num_tokens for s in high]):.1f} tokens")
        print(f"  Low density:  mean={np.mean([s.num_tokens for s in low]):.1f} tokens")

        for h, l in zip(high, low):
            diff = abs(h.num_tokens - l.num_tokens)
            print(f"    {h.id}({h.num_tokens}t) vs {l.id}({l.num_tokens}t) | diff={diff}")

#%% [markdown]
# ## Part 1.5: Load PER Results from Manual LLM Queries
#
# `per_results/` 폴더의 JSON 파일을 읽어서 expanded_text를 채움.
# 파일명 = 모델명 (e.g., claude-opus-4.json)

#%%
def load_per_results(
    samples: list[DensitySample],
    results_dir: str = "per_results",
) -> tuple[list[DensitySample], dict[str, dict]]:
    """
    per_results/*.json 파일들을 읽어서 expanded_text를 채움.
    여러 모델 결과가 있으면 모두 로드해서 per_scores를 모델별로 비교 가능하게 반환.

    Returns:
        samples: expanded_text가 채워진 샘플 (마지막 모델 기준)
        all_results: {model_name: {id: per_score}} — 모델별 PER 비교용
    """
    results_path = Path(__file__).parent / results_dir
    result_files = list(results_path.glob("*.json"))
    result_files = [f for f in result_files if f.name != "template.json"]

    if not result_files:
        print(f"No result files found in {results_path}")
        print("Run the prompts from per_prompt.md and save results there.")
        return samples, {}

    # id → sample 인덱스 매핑
    id_to_idx = {s.id.upper(): i for i, s in enumerate(samples)}

    all_results = {}  # {model_name: {id: per_score}}

    for result_file in result_files:
        model_name = result_file.stem
        with open(result_file) as f:
            data = json.load(f)

        model_key = data.get("model", model_name)
        print(f"\nLoaded: {model_key}")

        tokenizer_name = "bert-base-multilingual-cased"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        per_scores = {}
        for key, expanded_text in data.items():
            if key == "model" or not expanded_text:
                continue

            sentence_id = key.upper()
            if sentence_id not in id_to_idx:
                continue

            idx = id_to_idx[sentence_id]
            sample = samples[idx]
            sample.expanded_text = expanded_text
            sample.expanded_num_tokens = len(tokenizer.encode(expanded_text, add_special_tokens=False))

            assert sample.num_tokens and sample.num_tokens > 0
            sample.per_score = sample.expanded_num_tokens / sample.num_tokens
            per_scores[sentence_id] = sample.per_score

            print(f"  {sentence_id}: {sample.num_tokens}t → {sample.expanded_num_tokens}t | PER={sample.per_score:.2f}")

        all_results[model_key] = per_scores

    return samples, all_results


def compare_per_across_models(all_results: dict[str, dict]) -> None:
    """
    여러 모델의 PER 일관성 검증.
    PER의 재현성이 높아야 유효한 측정임.
    """
    if len(all_results) < 2:
        print("Need at least 2 model results for cross-model comparison.")
        return

    from scipy import stats

    models = list(all_results.keys())
    common_ids = set.intersection(*[set(v.keys()) for v in all_results.values()])

    print(f"\n=== Cross-Model PER Consistency ({len(common_ids)} sentences) ===")

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = [all_results[models[i]][k] for k in sorted(common_ids)]
            b = [all_results[models[j]][k] for k in sorted(common_ids)]
            rho, p = stats.spearmanr(a, b)
            print(f"  {models[i]} vs {models[j]}: Spearman ρ={rho:.3f}, p={p:.4f}")

    # 고밀도/저밀도 PER 차이가 모델 간에 일관적인가
    print("\n  Mean PER by density label:")
    for model, scores in all_results.items():
        high_per = [v for k, v in scores.items() if "_H" in k]
        low_per  = [v for k, v in scores.items() if "_L" in k]
        if high_per and low_per:
            print(f"  {model}: High={np.mean(high_per):.2f}, Low={np.mean(low_per):.2f}, ratio={np.mean(high_per)/np.mean(low_per):.2f}x")


#%% [markdown]
# ## Part 2: Surprisal Calculation
# 각 토큰의 surprisal (= -log P(token|context))
# → 고밀도 문장이 토큰당 surprisal이 높은지, 분산이 큰지

#%%
import torch

def compute_surprisal(
    samples: list[DensitySample],
    model_name: str = "skt/ko-gpt-trinity-1.2B-v0.5",
    device: str = "mps",
) -> list[DensitySample]:
    """
    Autoregressive LM으로 토큰별 surprisal 계산.

    Surprisal(t) = -log2 P(x_t | x_{<t})
    높은 surprisal = 모델이 예측하기 어려운 토큰 = 정보량 높음
    """
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    for s in samples:
        input_ids = tokenizer.encode(s.text, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids)
            # logits: (1, seq_len, vocab_size)
            logits = outputs.logits
            assert logits.shape[:2] == (1, seq_len), f"Shape mismatch: {logits.shape}"

        # Shift: logits[t]는 x_{t+1}을 예측 → surprisal[t]는 x_t가 x_{<t}로부터 얼마나 놀라운지
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq_len-1, V)
        target_ids = input_ids[:, 1:]  # (1, seq_len-1)

        # gather: 각 위치에서 실제 토큰의 log probability
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)
        surprisals = -token_log_probs.squeeze(0) / np.log(2)  # bits 단위

        assert surprisals.shape == (seq_len - 1,), f"Surprisal shape: {surprisals.shape}"

        s.mean_surprisal = surprisals.mean().item()
        s.max_surprisal = surprisals.max().item()
        s.surprisal_variance = surprisals.var().item()

    return samples

#%% [markdown]
# ## Part 3: PER Calculation
# LLM에게 "풀어쓰기" 요청 → 토큰 확장 비율 측정
#
# 이 부분은 API 호출 또는 로컬 LLM 필요.
# 여기서는 프롬프트 템플릿을 정의하고, 수동/API로 채울 수 있게 함.

#%%
PER_PROMPT_KO = """다음 문장의 모든 암묵적 의미, 전제, 함축을 명시적으로 풀어서 써주세요.
초등학생도 이해할 수 있도록, 생략된 맥락을 모두 보충해주세요.
원문의 뜻을 빠짐없이 전달하되, 최대한 쉽고 자세하게 써주세요.

원문: {text}

풀어쓴 버전:"""

PER_PROMPT_EN = """Rewrite the following sentence so that ALL implicit meanings, presuppositions,
and implications are made fully explicit. Write it so that a 10-year-old could understand.
Fill in every piece of missing context. Preserve the full meaning but make it as simple
and detailed as possible.

Original: {text}

Expanded version:"""

def get_per_prompt(sample: DensitySample) -> str:
    if sample.lang == "ko":
        return PER_PROMPT_KO.format(text=sample.text)
    return PER_PROMPT_EN.format(text=sample.text)

def calculate_per(
    samples: list[DensitySample],
    tokenizer_name: str = "bert-base-multilingual-cased",
) -> list[DensitySample]:
    """
    expanded_text가 채워진 샘플의 PER 계산.
    PER = expanded_tokens / original_tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for s in samples:
        if s.expanded_text is None:
            continue
        s.expanded_num_tokens = len(tokenizer.encode(s.expanded_text, add_special_tokens=False))

        assert s.num_tokens is not None and s.num_tokens > 0
        s.per_score = s.expanded_num_tokens / s.num_tokens

    return samples

#%% [markdown]
# ## Part 4: Visualization
# 밀도 라벨별 각 측정치 분포 비교

#%%
def plot_density_comparison(samples: list[DensitySample], save_dir: str = "../outputs"):
    """고밀도 vs 저밀도: surprisal, PER, proposition density 비교"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Text Density: High vs Low — Internal Signals", fontsize=14)

    for row, lang in enumerate(["ko", "en"]):
        lang_samples = [s for s in samples if s.lang == lang]
        high = [s for s in lang_samples if s.density_label == "high"]
        low = [s for s in lang_samples if s.density_label == "low"]

        lang_label = "Korean" if lang == "ko" else "English"

        # (1) Mean Surprisal
        ax = axes[row, 0]
        h_vals = [s.mean_surprisal for s in high if s.mean_surprisal is not None]
        l_vals = [s.mean_surprisal for s in low if s.mean_surprisal is not None]
        if h_vals and l_vals:
            ax.bar(["High", "Low"], [np.mean(h_vals), np.mean(l_vals)],
                   color=["#e74c3c", "#3498db"], alpha=0.8)
            ax.set_title(f"{lang_label}: Mean Surprisal (bits)")
            ax.set_ylabel("bits per token")

        # (2) Surprisal Variance (UID violation)
        ax = axes[row, 1]
        h_vals = [s.surprisal_variance for s in high if s.surprisal_variance is not None]
        l_vals = [s.surprisal_variance for s in low if s.surprisal_variance is not None]
        if h_vals and l_vals:
            ax.bar(["High", "Low"], [np.mean(h_vals), np.mean(l_vals)],
                   color=["#e74c3c", "#3498db"], alpha=0.8)
            ax.set_title(f"{lang_label}: Surprisal Variance")
            ax.set_ylabel("variance")

        # (3) PER
        ax = axes[row, 2]
        h_vals = [s.per_score for s in high if s.per_score is not None]
        l_vals = [s.per_score for s in low if s.per_score is not None]
        if h_vals and l_vals:
            ax.bar(["High", "Low"], [np.mean(h_vals), np.mean(l_vals)],
                   color=["#e74c3c", "#3498db"], alpha=0.8)
            ax.set_title(f"{lang_label}: PER (Paraphrase Expansion Ratio)")
            ax.set_ylabel("expansion ratio")
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/step0_density_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_dir}/step0_density_comparison.png")

#%% [markdown]
# ## Part 5: Correlation Analysis
# PER vs Surprisal vs Proposition Count 간 상관관계
# → PER이 다른 밀도 지표와 일관되게 움직이는지

#%%
def correlation_analysis(samples: list[DensitySample]):
    """PER, surprisal, proposition count 간 Spearman 상관계수"""
    from scipy import stats

    # PER이 채워진 샘플만
    valid = [s for s in samples if s.per_score is not None and s.mean_surprisal is not None]

    if len(valid) < 5:
        print("Not enough samples with PER scores. Fill expanded_text first.")
        return

    per_scores = [s.per_score for s in valid]
    surprisals = [s.mean_surprisal for s in valid]
    propositions = [s.num_propositions_est for s in valid]

    metrics = {
        "PER": per_scores,
        "Mean Surprisal": surprisals,
        "Proposition Count": propositions,
    }

    print("\n=== Spearman Correlations ===")
    keys = list(metrics.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho, p = stats.spearmanr(metrics[keys[i]], metrics[keys[j]])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {keys[i]} vs {keys[j]}: rho={rho:.3f}, p={p:.4f} {sig}")

#%% [markdown]
# ## Main: Run Step 0
#
# 실행 순서:
# 1. 시드 데이터 로드
# 2. 토큰 수 계산
# 3. Surprisal 계산 (LM 필요)
# 4. PER 계산 (LLM API 또는 수동)
# 5. 시각화 + 상관관계

#%%
if __name__ == "__main__":
    # 1. Load
    samples = load_seed_data()
    print(f"Loaded {len(samples)} sentences")

    # 2. Token counts
    samples = count_tokens(samples)
    print_token_stats(samples)

    # 3. Load PER results (per_results/*.json 파일이 있으면 자동으로 읽음)
    samples, all_results = load_per_results(samples)

    if all_results:
        # 여러 모델 결과가 있으면 일관성 검증
        compare_per_across_models(all_results)

        # 4. Surprisal (선택 — gpt2 또는 한국어 LM 필요)
        # samples = compute_surprisal(samples, model_name="gpt2", device="mps")

        # 5. 시각화 + 상관관계
        plot_density_comparison(samples)
        correlation_analysis(samples)
    else:
        print("\n--- Next Step ---")
        print("1. per_prompt.md 의 프롬프트를 LLM에 붙여넣기")
        print("2. 결과 JSON을 per_results/<model-name>.json 으로 저장")
        print("3. 이 스크립트 재실행")

#%% [markdown]
# ## Next Steps
# - [ ] LLM API로 PER 자동 계산 (anthropic API 또는 local LLM)
# - [ ] 시드 데이터 확장 (2x2 매트릭스: density x surprisal)
# - [ ] Inter-annotator agreement: PER의 재현성 검증 (같은 문장, 다른 LLM → PER 차이?)
