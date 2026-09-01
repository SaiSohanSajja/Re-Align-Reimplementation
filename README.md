# Re-Align: Simplified Preference Generation Pipeline

A lightweight reimplementation of the **Preference Generation** pipeline from *"Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization"* (Xing et al., EMNLP 2025, TAMU TACO Group), demonstrating the paper's core retrieval-based hallucination injection mechanism on a small sample set.

## Overview

VLM alignment via Direct Preference Optimization (DPO) typically relies on preference pairs — a "chosen" (correct) response and a "rejected" (incorrect) response for a given image. Prior methods construct rejected responses through brute-force techniques (random word swaps, corrupted images), producing negatives that are too easy to distinguish from language patterns alone, without requiring the model to actually ground its answer in the image.

Re-Align's key insight: instead of generating arbitrary wrong answers, **retrieve a visually similar but different image**, and use it to construct a *plausible, natural* rejected response — one that reads fluently but describes the wrong image. This forces genuine visual grounding rather than language-only shortcuts.

This repo implements that **Preference Generation** stage end-to-end on a small sample set of images.

## Pipeline

1. **Chosen response generation** — caption each image using BLIP (`Salesforce/blip-image-captioning-base`), producing $y_w$
2. **Strategic masking** — mask out attribute/color words and noun-like tokens from the chosen caption, producing $y_m$
3. **Image retrieval** — embed all images using CLIP (`openai/clip-vit-base-patch32`) and retrieve the most visually similar *other* image via cosine similarity
4. **Inducing hallucination** — generate a candidate completion conditioned on the retrieved (wrong) image, producing the reconstructed response $y_c$
5. **Rejection validation** — embed both $y_w$ and $y_c$ using SentenceTransformer (`all-mpnet-base-v2`) and check cosine similarity against the paper's 0.95 threshold; if below threshold, $y_c$ is accepted as the valid rejected response $y_l$

## Results

On a sample set of 6 images (2 cars, 2 dogs, 2 rooms), retrieval correctly paired each image with its category-mate (similarity 0.71–0.75), and all 6 generated preference pairs passed the rejection validity check with similarity scores well below the 0.95 threshold (0.27–0.52), confirming the reconstructed responses were meaningfully different from the correct ones.

| Image | Chosen Response | Rejected Response | Similarity |
|-------|----------------|--------------------|-----------| 
| car_red | a blue chevrolet camaro on a dirt road | a black porsche driving down a highway | 0.4416 |
| dog_1 | a puppy with a stick in its mouth | a white dog laying in the snow | 0.2732 |
| room_1 | a hotel room with a bed and television | a living room with a large window and a couch | 0.5153 |

## Key Simplifications (vs. the original paper)

This is a small-scale, compute-constrained reimplementation focused on demonstrating the *mechanism*, not reproducing benchmark results. Honest deviations from the original method:

- **Masking**: the paper uses GPT-4o mini with a dedicated prompt to intelligently mask objects, attributes, and logical relationships. This implementation uses a simple heuristic (predefined attribute word list + noun-following-article pattern) due to API cost constraints.
- **Masked completion**: the paper conditions the VLM directly on the masked sentence + retrieved image to fill in only the blanked segments, preserving sentence structure. BLIP does not support conditioning on partial/masked text, so this implementation generates a fresh caption for the retrieved image instead — approximating the same outcome (a plausible response describing the wrong image) without preserving exact sentence structure.
- **Scale**: 6 sample images across 3 categories vs. the paper's 11k–16k images sampled from LLaVA-Instruct-150K.
- **No fine-tuning**: this repo implements Preference Generation (Section 3.1) only. The paper's rDPO fine-tuning stage (Section 3.2 — combining $\mathcal{L}_{DPO}$ and $\mathcal{L}_{vDPO}$ to actually update VLM weights) requires GPU resources (the paper uses 4x NVIDIA A6000ada GPUs) and is not implemented here.

## Reference

Xing, S., Wang, Y., Li, P., Bai, R., Wang, Y., Qian, C., Yao, H., & Tu, Z. (2025). *Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization.* EMNLP 2025. [arXiv:2502.13146](https://arxiv.org/abs/2502.13146) | [Official Repo](https://github.com/taco-group/Re-Align)
