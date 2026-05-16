# 10 Years of Deep Learning in NLP — Code Walkthrough

This document explains the notebook `final_nlp_deep_learning_10_years_tutorial.ipynb`
cell by cell, and serves as the script for the accompanying video.

**Notebook goal:** tell the story of NLP from 2013 to ChatGPT — tokenization, word
embeddings, RNNs/LSTMs, attention, the Transformer, and modern LLMs — by
*implementing the core ideas from scratch in NumPy*. There is no large training
run here; every concept is built small enough to see exactly how it works, and
each one ships with a visualization.

The notebook is mostly NumPy + Matplotlib, so it runs anywhere (CPU is fine).

---

## Setup

### Cell 4 — Imports
NumPy, Matplotlib, Seaborn, and typing helpers. Sets a Seaborn plot style and
`np.random.seed(42)` for reproducibility. The `pip install` line is commented out
because Colab already has these packages.

## Chapter 1 — The basics of language modeling

### Cell 7 — `visualize_tokenization`
Takes a sentence and breaks it three ways: **character-level** (every character is
a token), **word-level** (regex split on words/punctuation), and a simplified
**subword** scheme (splits common suffixes like `-ing`, `-ed`, `-s` into `##`
pieces, mimicking BPE/WordPiece). It then draws all three side by side as labeled
boxes. The takeaway: the same sentence yields very different token counts depending
on the strategy — the core tokenization trade-off.

### Cell 9 — `SimpleTokenizer`
A from-scratch word-level tokenizer, the way early NLP systems worked. It holds
`word_to_id` / `id_to_word` dictionaries and four special tokens (`[PAD]`,
`[UNK]`, `[START]`, `[END]`). `fit()` builds the vocabulary from a list of texts;
`encode()` turns a sentence into IDs (unknown words map to `[UNK]`); `decode()`
reverses it. The demo fits on five sentences and shows the full encode/decode of
`"I love neural networks"`.

## Chapter 1 (cont.) — Word embeddings

### Cell 12 — `WordEmbeddings`
Embeddings give words *meaning* as vectors instead of arbitrary IDs. This class
stores a vector per word and computes **cosine similarity** between two words
(dot product divided by the product of norms — measures the angle, range −1 to 1).
The demo hand-crafts 4-dimensional embeddings whose dimensions are interpretable:
`[royalty, gender_male, age, power]`. So `king`, `queen`, `man`, `woman`, `cat`,
`dog` etc. get vectors that *should* place similar words close together.

### Cell 13 — `visualize_word_similarities`
Builds an N×N cosine-similarity matrix for a chosen word list and draws it as a
red-yellow-green heatmap. Royalty words light up green with each other, animals
cluster separately.

### Cell 15 — `word_analogy` — the famous King − Man + Woman test
Implements analogy arithmetic: `embedding(A) − embedding(B) + embedding(C)`, then
finds the existing word whose embedding is closest to that result. The demo runs
`king − man + woman` (expecting `queen`) and `prince − boy + girl` (expecting
`princess`), printing each step in an ASCII box.

### Cell 17 — `visualize_embeddings_2d`
Projects the 4-D embeddings down to 2-D with **PCA** and scatter-plots them, colored
by group (royalty / people / animals). Arrows connect gendered pairs to show the
analogy direction is roughly consistent — the geometric meaning of "word vectors."

## Chapter 2 — Sequential models

### Cell 20 — `SimpleRNN`
A vanilla RNN cell built from raw NumPy. Weights `W_i`, `W_h`, `W_o` and biases.
The core recurrence is `h_new = tanh(W_i·x + W_h·h + b_h)` applied token by token,
carrying the hidden state forward. `visualize_processing` then shows two things:
a heatmap of how the hidden state evolves word by word, and a bar chart of the
hidden-state magnitude growing as information accumulates.

### Cell 23 — `SimpleLSTM`
A from-scratch LSTM showing the gate mechanics explicitly. It keeps a hidden state
`h` and a cell state `c`, and at each step computes:
- **forget gate** `f` — what to drop from the cell state
- **input gate** `i` and **candidate** `c_tilde` — what new info to add
- cell update `c = f*c + i*c_tilde`
- **output gate** `o` — what to expose: `h = o * tanh(c)`

`visualize_gates` plots the average activation of each gate per word, plus the cell
state magnitude — you can literally watch the gates open and close.

### Cell 25 — `visualize_seq2seq`
A diagram of the encoder–decoder architecture for translation (French → English).
The encoder LSTM compresses the whole French sentence into one **context vector**;
the decoder LSTM generates English from it. The cell ends by naming the key
weakness: that single context vector is a **bottleneck** for long sentences —
which motivated attention.

### Cell 27 — Attention from scratch
Defines `softmax` and an `attention(query, keys, values)` function: score the
query against every key (dot product), softmax the scores into weights, return the
weighted sum of values. `visualize_attention_mechanism` simulates French encoder
states and English decoder queries, computes the attention matrix, and draws it as
a heatmap — showing each English word "looking at" the right French word.

## Chapter 3 — The Transformer

### Cell 30 — `SelfAttention`
Self-attention from scratch. From the input `X` it creates **Q, K, V** via three
weight matrices, computes `scores = Q·Kᵀ`, **scales by √d_k** (keeps values from
exploding), optionally adds a mask, softmaxes into weights, and returns
`weights·V`. Every word attends to every other word. The visualizer shows the
attention matrix for `"The cat sat on mat"`.

### Cell 32 — `MultiHeadAttention`
Runs several `SelfAttention` heads in parallel and combines them. Each head can
specialize (syntax, semantics, position, long-range). `visualize_heads` displays
all four heads' attention matrices side by side so you can see they learn
different patterns.

### Cell 34 — Positional encodings
Transformers have no built-in notion of order, so `get_positional_encoding`
generates the classic **sinusoidal** encodings: even dimensions use `sin`, odd use
`cos`, each dimension at a different frequency. `visualize_positional_encodings`
draws four views — the full heatmap, individual position patterns, the sine waves
per dimension, and a position-similarity matrix. Without these, "dog bites man"
and "man bites dog" would be indistinguishable to the model.

## Chapter 4 — Large Language Models

### Cell 37 — `visualize_causal_mask`
Shows the causal (look-ahead) mask used by GPT-style decoders: a lower-triangular
matrix where a token may attend to itself and the past but **not** the future.
This is what lets a decoder be trained on all positions at once without "cheating."

### Cells 38–41 (markdown) — BERT, XLNet, ChatGPT
These are narrative cells (no code): BERT as the bidirectional encoder-only model,
other architectures like XLNet, hallucination and alignment, and the path from
InstructGPT to ChatGPT/GPT-4 (RLHF — making models helpful and safe).

### Cell 42 (markdown) — Summary timeline
Recaps the decade: word2vec (2013) → seq2seq (2014) → attention (2015) →
Transformer (2017) → BERT/GPT (2018) → scaling → ChatGPT.

---

## Key takeaways
- **Tokenization** turns text into units; subword tokenization balances vocabulary
  size against sequence length.
- **Embeddings** turn tokens into vectors where geometry encodes meaning (analogy
  arithmetic works).
- **RNN/LSTM** process sequences step by step; gates fix vanishing gradients.
- **Attention** removes the seq2seq context-vector bottleneck by letting the
  decoder look at all encoder states.
- **Self-attention + multi-head + positional encodings** = the Transformer.
- **LLMs** are large Transformers; a causal mask makes them autoregressive, and
  RLHF aligns them with human intent.
