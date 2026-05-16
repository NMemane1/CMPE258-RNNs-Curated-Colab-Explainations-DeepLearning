# Vision Transformers & the Frontier of Computer Vision — Code Walkthrough

This document explains the notebook `final_vision_transformers_tutorial.ipynb`
cell by cell, and serves as the script for the accompanying video.

**Notebook goal:** build a Vision Transformer (ViT) from scratch in PyTorch —
attention, patch embedding, encoder blocks, the full model — and then survey the
modern vision frontier: CLIP, DINOv2, SAM, Swin, and ConvNeXt.

The notebook runs on PyTorch (CPU works; a GPU is faster). The from-scratch parts
build and forward-pass models but don't run a long training job — the focus is
architecture understanding.

---

## Setup

### Cell 1 — Imports
PyTorch, torchvision (for pretrained models later), NumPy, Matplotlib, PIL. Sets
the device (`cuda` if available) and `torch.manual_seed(42)` for reproducibility.

## Chapter 1 — The attention mechanism

### Cell 3 — `scaled_dot_product_attention`
The heart of every Transformer. Given Q, K, V it computes
`scores = Q·Kᵀ / √d_k`, optionally applies a mask (`masked_fill` with −1e9 so those
positions vanish after softmax), softmaxes into attention weights, and returns
`weights·V`. The √d_k scaling keeps the dot products from growing with dimension
and saturating the softmax. The demo runs random Q/K/V and confirms each attention
row sums to 1.

### Cell 4 — `MultiHeadAttention` (nn.Module)
The proper PyTorch version. Linear projections `W_q`, `W_k`, `W_v`, `W_o`; the
input is reshaped into `num_heads` parallel heads each of dimension
`d_k = d_model / num_heads`; each head runs scaled dot-product attention; outputs
are concatenated and projected by `W_o`. Multiple heads let the model attend to
different relationships at once. The demo runs self-attention (`Q=K=V=x`) on a
`(2, 16, 64)` tensor with 8 heads.

### Cell 5 — `visualize_attention`
Plots the attention matrix for each of the 8 heads as a grid of heatmaps — making
visible that different heads focus on different positions/patterns.

## Chapter 2 — Vision Transformer (ViT)

### Cell 7 — `PatchEmbedding`
ViT's key idea: "an image is worth 16×16 words." An image is cut into
non-overlapping patches, and each patch becomes a token. The efficient trick used
here is a single `nn.Conv2d` with `kernel_size = stride = patch_size` — that one
convolution simultaneously extracts and linearly projects every patch. For a
224×224 image with 16×16 patches you get 196 patch tokens of dimension 768.

### Cell 8 — `visualize_patches`
Builds a synthetic gradient image, overlays the patch grid, and shows the first 16
patches linearized — a visual of what "patchification" actually does.

### Cell 9 — `TransformerEncoderBlock`
One encoder block in the **pre-norm** variant used by ViT:
`x → LayerNorm → MultiHeadAttention → + x` (residual), then
`x → LayerNorm → MLP → + x` (residual). The MLP is
`Linear(d_model → 4·d_model) → GELU → Linear(back)`. Residual connections and
LayerNorm are what make deep Transformers trainable.

### Cell 10 — `VisionTransformer` (the full model)
Assembles everything:
1. `PatchEmbedding` → 196 patch tokens.
2. Prepend a learnable **`[CLS]` token** → 197 tokens. Its final state is the
   image summary used for classification.
3. Add learnable **position embeddings** (the model must be told patch order).
4. A stack of 12 `TransformerEncoderBlock`s.
5. Final LayerNorm, take the `[CLS]` token, pass it through a linear head to class
   logits.

The demo builds ViT-Base (~86M parameters) and forwards a `(2, 3, 224, 224)` batch.

### Cell 11 — ViT variants comparison
Instantiates ViT-Ti / S / B / L / H and prints hidden dim, layers, heads, and
parameter count for each. The cell notes ViT needs large datasets (ImageNet-21k)
to train from scratch — otherwise use pretrained weights.

## Chapter 3 — CLIP

### Cell 13 — CLIP zero-shot classification
CLIP is trained to align images and text in a shared embedding space. The cell
*tries* to load `ViT-B/32`; if the `clip` package isn't installed it falls back to
printing conceptual code. The workflow: encode the image and a set of text prompts
(`"a photo of a {class}"`), L2-normalize both, take the cosine similarity, softmax
→ class probabilities. The power: classification with **no training**, on any set
of labels you can phrase in words.

## Chapter 4 — DINOv2

### Cell 15 — Using DINOv2
DINOv2 is a **self-supervised** model — it learns strong visual features from 142M
images with *no labels*. The cell loads `dinov2_vits14` from `torch.hub` and shows
how to get both the global `[CLS]` feature and the per-patch tokens (useful for
dense tasks like segmentation). Wrapped in try/except with a fallback example.

### Cell 16 — DINOv2 variants
Prints the S/B/L/G variants with parameter counts (22M → 1.1B) and lists DINOv2's
properties: self-supervised, universal features, emergent object boundaries.

## Chapter 5 — SAM (Segment Anything)

### Cell 18 — SAM architecture
A printed explanation of SAM's three parts: a heavy **image encoder** (ViT, run
once per image), a flexible **prompt encoder** (points, boxes, masks, text), and a
lightweight **mask decoder** (run per prompt, outputs 3 candidate masks + scores).
Lists SAM variants including SAM 2 for video.

### Cell 19 — SAM usage example
Printed code showing the real API: `sam_model_registry` → `SamPredictor` →
`set_image()` → `predict()` with point or box prompts. Highlights: zero-shot,
interactive refinement, multiple mask outputs.

## Chapter 6 — Hybrid architectures

### Cell 21 — ConvNeXt
A CNN modernized with Transformer design choices (patchify stem, inverted
bottleneck, 7×7 depthwise convs, fewer activations/norms). Loads the pretrained
`convnext_tiny` from torchvision, prints its parameter count and ImageNet accuracy
(~82.5%), and forwards a test image. The lesson: a well-designed ConvNet still
matches Transformers.

### Cell 22 — Swin Transformer
A hierarchical ViT that does attention inside **local windows** (O(n) instead of
O(n²)) and uses **shifted windows** to connect them. Loads pretrained `swin_t`,
prints parameters and accuracy, forwards a test image.

## Chapter 7 — Practical applications

### Cell 24 — Model selection guide
Printed decision tables: which model to pick for classification, detection,
segmentation, and feature extraction, given dataset size and constraints.

### Cell 25 — Pretrained model comparison
A table comparing ResNet, EfficientNet, ViT, Swin, ConvNeXt, DINOv2 on type,
parameters, resolution, and ImageNet top-1 accuracy.

### Cell 26 — Quick-start snippets
Copy-paste code for the five common workflows (ViT fine-tuning, DINOv2 feature
extraction, CLIP zero-shot, SAM segmentation, ConvNeXt fine-tuning).

### Cell 28 — Cheat sheet
A printed reference: the attention formula, the ViT architecture diagram, the
encoder block, key models, and how to load each from torchvision / torch.hub.

---

## Key takeaways
- **Attention** (`softmax(QKᵀ/√d_k)·V`) is the universal building block.
- **ViT** turns an image into patch tokens + a `[CLS]` token + position embeddings,
  then applies a standard Transformer encoder.
- **CLIP** aligns vision and language → zero-shot classification.
- **DINOv2** learns universal features with no labels (self-supervised).
- **SAM** is promptable, zero-shot segmentation.
- **Swin / ConvNeXt** show hybrid and modernized-CNN designs that rival pure ViTs.
