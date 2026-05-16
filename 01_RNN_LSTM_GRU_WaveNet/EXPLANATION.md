# Sequence Models: Zero to Hero — Code Walkthrough

This document explains the notebook `final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb`
cell by cell. It also doubles as the script for the accompanying video walkthrough.

**Notebook goal:** build four different sequence models — a vanilla RNN, an LSTM,
a GRU, and a WaveNet — train all of them on the *same* character-level language
modeling task, and compare them head to head on loss, speed, and text quality.

The notebook was executed end to end on a Colab **Tesla T4 GPU** (PyTorch 2.10,
CUDA enabled), so every cell already contains its real output.

---

## Part 0 — Setup

### Cell 3 — Imports, seeds, and device
We import PyTorch (`torch`, `nn`, `functional`, `Dataset`/`DataLoader`), NumPy,
and Matplotlib. Two things matter here:
- **Reproducibility:** `SEED = 42` is applied to `torch`, `numpy`, and CUDA so every
  run produces the same numbers.
- **Device selection:** `device = 'cuda' if available else 'cpu'`. The output
  confirms it ran on a Tesla T4 GPU.

### Cell 4 — Plotting defaults
Sets Matplotlib `rcParams` (figure size, grid, fonts) once so every later plot is
consistent, and defines a `COLORS` dictionary so each model keeps the same color
across all charts (RNN = red, LSTM = blue, GRU = green, WaveNet = purple).

## Part 1 — The data pipeline

### Cell 9 — The text corpus
Instead of downloading a dataset, the notebook hard-codes a ~3,000-character story
about a mountain village. Keeping the corpus inline makes the notebook fully
reproducible and fast. The output prints the length and the first 200 characters.

### Cell 10 — Character vocabulary
This is **character-level** modeling, so the "vocabulary" is every unique character
in the corpus. We sort the unique characters (deterministic order) and build two
lookup dictionaries: `char_to_idx` (encoding) and `idx_to_char` (decoding). The
output shows the vocab size and the example mappings like `'a' -> 5`.

### Cell 11 — Encode the whole corpus
Every character is converted to its integer index, producing one long 1-D
`torch.long` tensor. The cell also decodes the first 50 values back to text to
prove the round-trip works.

### Cell 13 — `CharDataset` and hyperparameters
`CharDataset` is a standard PyTorch `Dataset`. For each index `i` it returns a
sliding window:
- `x = data[i : i+seq_length]` — the input characters
- `y = data[i+1 : i+seq_length+1]` — the same window shifted right by one

So the target at every position is "the next character." Predicting at *every*
position (not just the last one) makes training far more efficient. The
hyperparameters are defined here and **shared by all four models** for a fair
comparison: `SEQ_LENGTH=64`, `BATCH_SIZE=64`, `EMBED_DIM=64`, `HIDDEN_DIM=128`,
`NUM_EPOCHS=20`, `LR=0.003`.

### Cell 14 — Train/validation split
The corpus is split **sequentially** 90/10 (not randomly), because we want to keep
the text's structure intact and evaluate on a contiguous block the model never saw.
Two `DataLoader`s handle batching and shuffling.

### Cell 15 — Inspect a batch
Pulls one batch and prints the input and target as decoded strings. You can
literally see the target is the input shifted by one character — a good sanity
check that the pipeline is correct.

## Part 1 (cont.) — Shared training infrastructure

### Cell 17 — `train_model`
The single training loop used by **every** model. Per epoch it: runs the forward
pass, computes `CrossEntropyLoss` (logits reshaped to `(batch*seq, vocab)`), does
backprop, applies **gradient clipping** (`clip_grad_norm_` at 1.0 — essential for
RNNs to stop exploding gradients), and steps the Adam optimizer. It also runs a
validation pass and records train loss, val loss, and epoch time into a `history`
dict.

### Cell 18 — `generate_text`
Generates text one character at a time: feed the seed, take the logits for the
**last** position, divide by `temperature`, softmax into probabilities, and sample
the next character with `torch.multinomial`. Lower temperature = more
deterministic/repetitive; higher = more random. It also initializes the
`all_results` dictionary that collects every model's output.

## Part 2 — Vanilla RNN

### Cell 24 — `CharRNN`
Three layers: `nn.Embedding` (char index → dense vector), `nn.RNN` (the recurrent
core, `batch_first=True`), and `nn.Linear` (hidden state → vocabulary logits). The
vanilla RNN updates its hidden state with a `tanh` at every step, which means
early information is gradually overwritten — the root cause of the
vanishing-gradient problem. ~33K parameters.

### Cell 25 — Train the RNN
Calls `train_model`. Watch the printed loss drop each epoch.

### Cell 26 — Generate from the RNN
Uses the shared seed `"The village "`. The RNN produces text that has roughly
word-like shapes but limited coherence.

### Cell 27 — RNN training curves
Two plots: loss curves (train vs. val) and time per epoch.

## Part 3 — LSTM

### Cell 32 — `CharLSTM`
Identical structure to `CharRNN` but `nn.RNN` → `nn.LSTM`. The LSTM keeps **two**
internal states: the hidden state `h` and the **cell state** `c` (the "memory
highway"). It uses three gates (forget, input, output) to control what to keep,
add, and expose. Because of the four gate weight matrices it has ~4× the
parameters of the vanilla RNN (~107K). The cell prints the exact ratio.

### Cell 33 — Train the LSTM
Same loop. Despite more parameters, gates help gradients flow, so optimization is
smooth.

### Cell 34 — Generate from the LSTM

### Cell 35 — RNN vs LSTM comparison
Side-by-side train and validation loss curves.

## Part 4 — GRU

### Cell 40 — `CharGRU`
Same pattern with `nn.GRU`. The GRU merges the cell and hidden state into one and
uses only **two** gates (update, reset), so it sits between RNN and LSTM in size
(~83K parameters). The cell prints a parameter-count table for all three.

### Cell 41–42 — Train and generate from the GRU

### Cell 43 — Three-way comparison
RNN vs LSTM vs GRU training and validation curves on one figure.

## Part 5 — Deep RNNs and practical tricks

### Cell 48 — `DeepCharLSTM`
A 2-layer stacked LSTM with `dropout=0.2` between layers, plus a dropout before the
output projection. Stacking lets layer 1 learn low-level patterns (letter pairs)
and layer 2 learn higher-level structure; dropout regularizes. ~240K parameters.

### Cell 49–50 — Train and generate from the deep LSTM

### Cell 51 — Effect of depth
Plots validation loss for the 1-layer vs 2-layer LSTM.

### Cell 53 — Temperature experiment
Picks whichever model had the best validation loss and generates the same prompt
at temperatures 0.2, 0.8, and 1.5 — showing how one parameter shifts output from
conservative/repetitive to creative/chaotic.

## Part 6 — WaveNet

### Cell 58 — `CausalConv1dBlock`
WaveNet's building block. A **causal** convolution: it left-pads the input by
`(kernel_size-1)*dilation` so the output at time `t` never sees the future.
**Dilation** spaces the kernel out, so stacking blocks grows the receptive field
exponentially. The block adds LayerNorm, a GELU activation, and a residual
connection so gradients flow cleanly.

### Cell 59 — `CharWaveNet`
Embedding → linear projection → a stack of 7 `CausalConv1dBlock`s with dilations
`[1, 2, 4, 8, 16, 32, 64]` → output projection. The total receptive field is 128
characters. Unlike RNNs, **all positions are computed in parallel** during
training. ~248K parameters.

### Cell 60–61 — Train and generate
Note the trade-off the cell calls out: WaveNet trains fast (parallel) but is
*slower* at generation, because every new character requires re-running the full
convolution stack, whereas an RNN only needs one step.

## Part 7 — Grand comparison

### Cell 64 — Comprehensive comparison plot
A 2×2 figure: training loss, validation loss, average time per epoch, and
parameter count for all four models.

### Cell 65 — Summary table
The key results table. From the executed run:

| Model | Params | Best Train | Best Val | Avg Time |
|-------|--------|-----------|----------|----------|
| RNN | 32,938 | 0.1400 | 2.1834 | 0.22s |
| LSTM | 107,434 | 0.1132 | 2.1986 | 0.18s |
| GRU | 82,602 | 0.1033 | **2.1429** | 0.16s |
| Deep LSTM | 239,530 | 0.1446 | 2.1661 | 0.24s |
| WaveNet | 248,490 | 0.1008 | 3.0741 | 0.39s |

**Honest reading of the results:** train losses near 0.1 with validation losses
above 2.0 means every model **overfits heavily** — unavoidable with a ~3K-character
corpus. The point of the notebook is not a state-of-the-art language model; it is
to *compare architectures* under identical conditions. The GRU edges out the best
validation loss; WaveNet overfits the most here because its large receptive field
memorizes this tiny corpus quickly.

### Cell 66 — Generated text comparison
Prints the first 200 generated characters from each model with the same seed, so
you can eyeball the qualitative difference.

---

## Key takeaways
- **RNN:** simplest recurrence; struggles with long-range dependencies.
- **LSTM:** cell state + 3 gates solve vanishing gradients; ~4× the parameters.
- **GRU:** 2 gates, fewer parameters, performance comparable to LSTM.
- **WaveNet:** dilated causal convolutions; parallel training, sequential
  generation, exponentially growing receptive field.
- Gradient clipping, dropout, and stacking layers are the practical tricks that
  make recurrent models trainable.
