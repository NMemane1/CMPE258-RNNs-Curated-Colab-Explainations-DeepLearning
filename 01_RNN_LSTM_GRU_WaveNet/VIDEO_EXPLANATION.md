# Video Explanation — RNN / LSTM / GRU / WaveNet

A recorded video walkthrough of the notebook
[`final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb`](final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb),
explaining the code and outputs cell by cell.

## Watch the video

**▶ [RNN / LSTM / GRU / WaveNet — Code Walkthrough](https://docs.google.com/videos/d/1B5XGMrUgGmmw8hXGVxwL1L7h1M6DACpeQYDy7Hiq1lY/edit?usp=sharing)**

> If the link asks for permission, the video is shared from Google Drive — make
> sure link sharing is set to "Anyone with the link can view."

## What the video covers

The walkthrough builds and compares four sequence models on the same
character-level language-modeling task:

| Part | Topic |
|------|-------|
| Setup & data | Imports, the text corpus, character vocabulary, the sliding-window dataset |
| Training tools | The shared training loop (with gradient clipping) and the text generator |
| Vanilla RNN | Simplest recurrence; ~33K parameters |
| LSTM | Cell state + 3 gates; ~107K parameters |
| GRU | 2 gates, lighter than LSTM; ~83K parameters |
| Deep LSTM & temperature | Stacked layers, dropout, and the temperature experiment |
| WaveNet | Dilated causal convolutions; parallel training |
| Grand comparison | Loss, speed, parameter count, and generated-text comparison |

## Related files

- **Notebook (executed, with outputs):** [`final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb`](final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb)
- **Written walkthrough (cell-by-cell):** [`EXPLANATION.md`](EXPLANATION.md)
- **Open in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NMemane1/CMPE258-RNNs-Curated-Colab-Explainations-DeepLearning/blob/main/01_RNN_LSTM_GRU_WaveNet/final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb)
- **Back to repository home:** [README](../README.md)
