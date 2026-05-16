# CMPE 258 — Curated Deep Learning Colabs with Explanations

A curated portfolio of four deep-learning tutorial notebooks, executed end to end,
archived **with their outputs**, and explained code block by code block. Each
notebook has a written walkthrough and an accompanying video explanation.

The four notebooks cover the modern deep-learning landscape for sequences, language,
vision, and graphs:

| # | Topic | Notebook | Open in Colab | Walkthrough | Video |
|---|-------|----------|---------------|-------------|-------|
| 1 | RNN / LSTM / GRU / WaveNet — Sequence Models Zero to Hero | [notebook](01_RNN_LSTM_GRU_WaveNet/final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NMemane1/CMPE258-RNNs-Curated-Colab-Explainations-DeepLearning/blob/main/01_RNN_LSTM_GRU_WaveNet/final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb) | [EXPLANATION.md](01_RNN_LSTM_GRU_WaveNet/EXPLANATION.md) | _add link_ |
| 2 | 10 Years of Deep Learning in NLP — Words to ChatGPT | [notebook](02_NLP_Deep_Learning_10_Years/final_nlp_deep_learning_10_years_tutorial.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NMemane1/CMPE258-RNNs-Curated-Colab-Explainations-DeepLearning/blob/main/02_NLP_Deep_Learning_10_Years/final_nlp_deep_learning_10_years_tutorial.ipynb) | [EXPLANATION.md](02_NLP_Deep_Learning_10_Years/EXPLANATION.md) | _add link_ |
| 3 | Vision Transformers — ViT, CLIP, DINOv2, SAM | [notebook](03_Vision_Transformers/final_vision_transformers_tutorial.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NMemane1/CMPE258-RNNs-Curated-Colab-Explainations-DeepLearning/blob/main/03_Vision_Transformers/final_vision_transformers_tutorial.ipynb) | [EXPLANATION.md](03_Vision_Transformers/EXPLANATION.md) | _add link_ |
| 4 | Graph Neural Networks — Fundamentals & GCNs | [notebook](04_Graph_Neural_Networks/final_gnn_fundamentals_tutorial.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NMemane1/CMPE258-RNNs-Curated-Colab-Explainations-DeepLearning/blob/main/04_Graph_Neural_Networks/final_gnn_fundamentals_tutorial.ipynb) | [EXPLANATION.md](04_Graph_Neural_Networks/EXPLANATION.md) | _add link_ |

> The "Open in Colab" badges open each notebook **directly from this GitHub repo** —
> no Drive copy required. In Colab you can then use *File → Save a copy in Drive* if
> you want your own editable copy.

---

## Video Walkthroughs

One video per notebook, walking through the code **block by block** and explaining
both the code and its output.

| Notebook | YouTube Video |
|----------|---------------|
| 1 — RNN / LSTM / GRU / WaveNet | _YouTube link to be added_ |
| 2 — 10 Years of NLP | _YouTube link to be added_ |
| 3 — Vision Transformers | _YouTube link to be added_ |
| 4 — Graph Neural Networks | _YouTube link to be added_ |

---

## Repository Structure

```
.
├── README.md
├── 01_RNN_LSTM_GRU_WaveNet/
│   ├── final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb   (executed, with outputs)
│   └── EXPLANATION.md                                  (code-block walkthrough)
├── 02_NLP_Deep_Learning_10_Years/
│   ├── final_nlp_deep_learning_10_years_tutorial.ipynb
│   └── EXPLANATION.md
├── 03_Vision_Transformers/
│   ├── final_vision_transformers_tutorial.ipynb
│   └── EXPLANATION.md
└── 04_Graph_Neural_Networks/
    ├── final_gnn_fundamentals_tutorial.ipynb
    └── EXPLANATION.md
```

---

## Notebook Summaries

### 1 — RNN / LSTM / GRU / WaveNet: Sequence Models Zero to Hero
Builds four sequence models in PyTorch — a vanilla RNN, an LSTM, a GRU, and a
WaveNet (dilated causal convolutions) — and trains them all on the same
character-level language-modeling task. Ends with a head-to-head comparison of
loss, training speed, parameter count, and generated text. Executed on a Colab
**Tesla T4 GPU**.

### 2 — 10 Years of Deep Learning in NLP
Walks the history of NLP from 2013 to ChatGPT by implementing the core ideas from
scratch in NumPy: tokenization, word embeddings (with the King − Man + Woman
analogy), RNN/LSTM cells, the attention mechanism, self-attention, multi-head
attention, positional encodings, and the causal mask behind GPT.

### 3 — Vision Transformers
Builds a Vision Transformer from scratch in PyTorch — scaled dot-product attention,
multi-head attention, patch embedding, the encoder block, and the full ViT — then
surveys the modern vision frontier: CLIP, DINOv2, SAM, Swin Transformer, and
ConvNeXt, with a practical model-selection guide.

### 4 — Graph Neural Networks: Fundamentals
Builds GNNs from the ground up: graph representations, why ordinary ML fails on
graphs (permutation invariance), the message-passing paradigm, and a Graph
Convolutional Network implemented twice — once in pure NumPy with hand-written
backprop, once in PyTorch — then trained on Zachary's Karate Club network.

---

## Execution Notes

All four notebooks are archived **with their cell outputs preserved** — they were
executed end to end in Google Colab (the sequence-models notebook on a Tesla T4
GPU). You can re-run any of them by clicking its "Open in Colab" badge above; for
the PyTorch notebooks, select a GPU runtime via *Runtime → Change runtime type*
for best speed.

---

## Course

CMPE 258 — Deep Learning. This repository is maintained as a learning portfolio:
curated tutorial notebooks, executed and explained, organized by topic.
