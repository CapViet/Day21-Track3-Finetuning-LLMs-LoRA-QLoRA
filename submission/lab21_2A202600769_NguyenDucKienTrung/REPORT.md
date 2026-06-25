# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Đức Kiên Trung — 2A202600769
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight ZIP)

## 1. Setup

| Item | Value |
|------|-------|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5-3B, 4-bit NF4 QLoRA via Unsloth) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train / 20 eval, seed 42) |
| **max_seq_length** | p95 of token lengths, rounded up to a power of 2 (auto from the notebook) |
| **GPU** | Google Colab **Tesla T4** (16 GB) — peak usage 6.6–8.0 GB across ranks |
| **Quantization** | 4-bit NF4 + double quantization, Unsloth custom kernels |
| **Optimizer / schedule** | `adamw_8bit`, 3 epochs, cosine LR, lr = 2e-4, warmup ratio = 0.10, effective batch = 8 (1 × grad-accum 8) |
| **LoRA targets** | `q_proj`, `v_proj` (lab spec), dropout = 0, gradient checkpointing on |
| **Stack** | Unsloth + TRL `SFTTrainer` (the provided `Lab21_LoRA_Finetuning_T4.ipynb`) |
| **Training cost** | ~11.2 min total for all 3 ranks ≈ **\$0** on free Colab (~\$0.07 @ \$0.35/hr) |

---

## 2. Rank Experiment Results

Same dataset, same hyperparameters — only `(r, alpha)` changes.

| Rank | α | Trainable Params | % of 3.09B | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|----|------------------|-----------|-----------|-----------|-----------|------------|
| **8**  | 16  | 1,843,200  | 0.060 % | 3.69 min | 7.22 GB | 1.5577 | **4.75** |
| **16** | 32  | 3,686,400  | 0.120 % | 3.88 min | 6.62 GB | 1.5161 | **4.55** |
| **64** | 128 | 14,745,600 | 0.478 % | 3.63 min | 8.00 GB | 1.4768 | **4.38** |

*(Source: `results/rank_experiment_summary.csv`. Perplexity = exp(eval_loss) on the
held-out 20-sample eval set.)*

> **Note on base model**: this Colab run evaluated the three fine-tuned ranks but did **not**
> compute a base (un-adapted) perplexity, so the table reports the 3 measured ranks. The
> relative rank trade-off — which is what this experiment isolates — is fully captured.

**Key observations:**
- Perplexity decreases **monotonically** with rank: 4.75 → 4.55 → 4.38.
- **Diminishing returns are sharp.** r=8→r=16 doubles the parameters (+1.84 M) for −0.19
  perplexity ≈ **0.105 ppl per million** added params. r=16→r=64 adds **6× more** params
  (+11.06 M) for only −0.18 perplexity ≈ **0.016 ppl per million** — a **~6.6× drop** in
  marginal efficiency.
- **VRAM is essentially flat / noise-dominated** (6.6–8.0 GB) — even r=64's 14.7 M LoRA
  params are tiny next to the 4-bit 3B base + activations, so peak VRAM is driven by
  allocator fragmentation, not rank. All ranks fit comfortably in a 16 GB T4.
- **Train time is independent of rank** (~3.6–3.9 min each) — the rank only changes small
  matmuls, not the dominant forward/backward cost.

---

## 3. Loss Curve Analysis

See `results/loss_curve.png` (r=16 training loss, 66 steps = 3 epochs).

- The curve descends from ~1.61 (step 5) to ~1.39 (step 65) — a clear downward trend with
  the expected mini-batch **noise** (effective batch = 8 on only 180 samples, so each step
  is a high-variance gradient estimate; the bumps at steps 15 and 35 are noise, not
  divergence).
- Final **training** loss (~1.39) sits **below** the final **eval** loss (1.5161 for r=16),
  a small ~0.12 gap — the model fits train slightly harder than eval, but the gap is modest.
- **No harmful overfitting at 3 epochs**: eval perplexity keeps *improving* with rank, and
  the eval loss (4.55 ppl) is healthy. Mid-training eval was disabled (`eval_strategy="no"`)
  to save T4 VRAM, so the curve shows training loss only — consistent with the notebook's
  T4 design.

---

## 4. Qualitative Comparison (Base vs Fine-tuned r=16)

Full table in `results/qualitative_comparison.csv` (5 examples). Honest mix — including a
clear regression — not cherry-picked.

### Example 1 — "Giải thích khái niệm machine learning cho người mới bắt đầu."
- **Base**: correct, reasonable definition.
- **r=16**: also correct, slightly more structured ("một bộ môn công nghệ máy tính dựa trên việc học tập…").
- **Verdict: ~equal / slight improvement** ✅ — knowledge already present in base; SFT mainly tidies phrasing.

### Example 2 — "Viết đoạn code Python tính số Fibonacci thứ n."
- **Base**: recursive, awkward edge cases (returns a string for n≤0).
- **r=16**: cleaner **iterative** version with `raise ValueError(...)` input validation.
- **Verdict: improved** ✅ — better code style and error handling.

### Example 3 — "Liệt kê 5 nguyên tắc thiết kế UI/UX."
- **Base**: verbose, repeats "thân thiện" several times, loses the list structure.
- **r=16**: crisp numbered list (1. Chuyển đổi … 2. Thích ứng … 3. Đơn giản …).
- **Verdict: improved** ✅ — format adherence is the headline SFT win.

### Example 4 — "Tóm tắt sự khác biệt giữa LoRA và QLoRA."
- **Base**: correctly expands **LoRA = "Low-Rank Adaptation"**.
- **r=16**: **hallucinates** LoRA = "Layer-wise Adaptive Regularization Optimization" — **factually wrong**.
- **Verdict: degraded** ❌ — a real regression: fine-tuning on style data made the model
  *more confident but less accurate* on a factual acronym. Good evidence that **SFT tunes
  form, not knowledge** — and can even erode facts the base knew.

### Example 5 — "Phân biệt prompt engineering, RAG, và fine-tuning."
- **Base**: accurate, slightly rambling.
- **r=16**: accurate, marginally more organized.
- **Verdict: ~equal** ➖.

**Summary**: fine-tuning improved **format and code style** (Examples 2, 3) but left
knowledge roughly unchanged (1, 5) and even **regressed one factual answer** (Example 4).
This is exactly the lecture's golden rule in action: *fine-tune for style/format; use RAG
for knowledge.*

---

## 5. Conclusion về Rank Trade-off

For this dataset and base model, **r=16 offers the best ROI**, while **r=64 gives the best
absolute perplexity**. The decisive metric is *marginal* perplexity per trainable parameter.
Going r=8→r=16 doubles the adapter (+1.84 M params) and buys −0.19 perplexity, about 0.105
ppl per million added parameters. Going r=16→r=64 adds **six times more** parameters
(+11.06 M) for only −0.18 perplexity — about 0.016 ppl per million, a **~6.6× collapse** in
marginal efficiency. So **diminishing returns set in clearly after r=16**: r=64 is still the
best in absolute terms, but each extra parameter buys far less, and it carries 4× the
adapter size and higher overfitting risk for a task this small (180 training samples). VRAM
and training time were effectively rank-independent (6.6–8.0 GB, ~3.6–3.9 min), so the
trade-off here is purely *capacity vs. marginal benefit*, not a memory or speed cost.
**Production recommendation: deploy r=16** — it captures the bulk of the achievable gain at a
quarter of r=64's parameters. I would only reach for r=64 with a larger, cleaner dataset
whose extra signal could actually fill that capacity without memorizing.

## 6. What I Learned

- **Rank is a capacity dial with a sharp knee, not a linear quality slider.** Most of the
  gain lands by r=16; pushing to r=64 quadrupled the adapter for a marginal perplexity
  improvement (~6.6× worse efficiency per parameter). The engineering question is *where the
  knee is*, not "how high can I go."
- **On a 3B model, rank is decoupled from VRAM and speed.** I expected higher rank to cost
  more memory/time; in practice the 4-bit base + activations dominate, so peak VRAM
  (6.6–8.0 GB) and step time were basically rank-independent. The real cost of high rank is
  *overfitting capacity*, not hardware.
- **SFT tunes form, not facts — and can even erode facts.** The biggest wins were
  format/code-style (numbered lists, iterative Fibonacci with validation), but Example 4
  showed the fine-tuned model *hallucinating* a wrong expansion of "LoRA" that the base got
  right. Concrete proof of the lecture rule: **fine-tune for style, use RAG for knowledge.**
