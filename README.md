# Attention Is All You Need — Transformer From Scratch

A from-scratch implementation of the Transformer architecture introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), built with PyTorch and
trained for English-to-Dutch neural machine translation.

The goal of this project is not just to use a Transformer, but to understand and
implement its core components from the ground up.

## Project Overview

The Transformer was introduced in 2017 by Vaswani et al. and replaced recurrent
architectures with an attention-based encoder-decoder architecture.

In this project, I implement the main Transformer components manually, including:

* Token embeddings
* Positional encoding
* Multi-head self-attention
* Scaled dot-product attention
* Feed-forward neural networks
* Layer normalization
* Residual connections
* Transformer encoder
* Transformer decoder
* Encoder-decoder cross-attention
* Padding masks
* Causal masks
* Autoregressive greedy decoding
* Training and validation pipeline

The model is trained on the OPUS Books English–Dutch dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Train](#2-train)
  - [3. Resume Training](#3-resume-training)
  - [4. Inference / Translation](#4-inference--translation)
- [Architecture Summary](#architecture-summary)
- [Known Limitations & Output Quality](#known-limitations--output-quality)
- [Resources & Acknowledgements](#resources--acknowledgements)
- [Contribution](#contribution)
- [License](#license)

## Technologies Used

| Technology | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | Core deep learning framework — model, training loop, autograd |
| [HuggingFace `datasets`](https://huggingface.co/docs/datasets) | Loads the `Helsinki-NLP/opus_books` (en–nl) dataset |
| [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) | Trains/loads WordLevel tokenizers for source & target languages |
| [`torchmetrics`](https://lightning.ai/docs/torchmetrics/) | Character Error Rate, Word Error Rate, and BLEU during validation |
| [TensorBoard](https://www.tensorflow.org/tensorboard) (`torch.utils.tensorboard`) | Logging training loss and validation metrics |
| [`tqdm`](https://github.com/tqdm/tqdm) | Training progress bars |
| [`nltk`](https://www.nltk.org/) | Imported for tokenization utilities |
| Python 3 | Language runtime |

GPU (CUDA) or Apple Silicon (MPS) is used automatically if available, otherwise the
model falls back to CPU.

## Project Structure

```
.
├── Paper/                    # Reference copy of the "Attention Is All You Need" paper
├── config.py                  # Hyperparameters, paths, and checkpoint helpers
├── Dataset_Translate.py        # PyTorch Dataset: tokenizes + pads source/target pairs, builds masks
├── Transformer_model.py        # Core Transformer architecture (encoder, decoder, attention, etc.)
├── Train_Model.py               # Data pipeline, training loop, validation, checkpointing
└── README.md
```

### File Descriptions

| File | Purpose |
|------|---------|
| `Paper/` | Reference copy of the paper used while implementing the architecture. |
| `config.py` | Returns a `get_config()` dict with all hyperparameters and paths; `get_weights_path()` / `latest_weight_path()` locate checkpoint files. |
| `Dataset_Translate.py` | `TranslateDataset` wraps a HuggingFace dataset split, tokenizes each sentence pair, adds `[SOS]`/`[EOS]`/`[PAD]` tokens up to `seq_len`, and builds the encoder padding mask and decoder padding+causal mask (`causual_mask`). |
| `Transformer_model.py` | Defines `Embed`, `PositionalEncoding`, `MultiHeadAttentions`, `FeedForwardNetwork`, `LayerNormalization`, `EncoderLayer`/`Encoder`, `DecoderLayer`/`Decoder`, `Generation` (output projection + log-softmax), and `EncoderDecoder`. `build_model()` assembles the full model. |
| `Train_Model.py` | Downloads/filters the `opus_books` (en–nl) dataset, trains WordLevel tokenizers per language, builds `DataLoader`s, trains the model epoch by epoch, runs validation with greedy decoding + CER/WER/BLEU metrics, and saves checkpoints per epoch. |

## Configuration

All settings live in `config.py`'s `get_config()`:

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `3` | Training batch size |
| `num_epochs` | `5` | Number of training epochs |
| `lr` | `1e-4` | Learning rate (Adam) |
| `seq_len` | `512` | Max sequence length (source & target) |
| `d_model` | `512` | Model embedding dimension |
| `datasource` | `'opus_books'` | HuggingFace dataset config name |
| `lang_src` / `lang_tgt` | `'en'` / `'nl'` | Source / target languages |
| `model_folder` | `'weights'` | Checkpoint folder (prefixed with `datasource`) |
| `model_basename` | `'tmodel_'` | Checkpoint filename prefix |
| `preload` | `None` | Set to `'latest'` or an epoch string to resume training |
| `tokenizer_file` | `'tokenizer_{0}.json'` | Per-language tokenizer file path template |
| `experiment_name` | `'runs'` | TensorBoard log directory |

Checkpoints are written to `{datasource}_{model_folder}/{model_basename}{epoch}.pt`
(e.g. `opus_books_weights/tmodel_00.pt`).

> **Note:** `get_model()` calls `build_model(src_vocab, trg_vocab, config['seq_len'], config['d_model'])`,
> but `build_model`'s signature is `(src_vocab, trg_vocab, src_seq, trg_seq, d_model=512, ...)`.
> This means `config['d_model']` is currently passed in as the *target sequence length*, and
> `d_model` falls back to its default (`512`). With the default config this is harmless since
> `seq_len == d_model == 512`, but if you change one of these values independently, update this
> call in `get_model()` accordingly.

## How to Run

### 1. Install Dependencies

```bash
pip install torch datasets tokenizers torchmetrics tqdm nltk tensorboard
```

### 2. Train

```bash
python Train_Model.py
```

On first run this will:
1. Download `Helsinki-NLP/opus_books` (`en-nl`) via `datasets`.
2. Filter out sentence pairs that don't fit within `seq_len`.
3. Train (or load) WordLevel tokenizers for English and Dutch, saved as
   `tokenizer_en.json` / `tokenizer_nl.json`.
4. Split the filtered data 90/10 into train/validation.
5. Build the Transformer via `build_model()` and train it with Adam +
   label-smoothed cross-entropy (`label_smoothing=0.1`, ignoring `[PAD]`).
6. Log training loss to TensorBoard (`runs/`) every step.
7. After each epoch, run greedy-decode validation on 2 examples, log
   CER / WER / BLEU, and save a checkpoint.

To view training curves:

```bash
tensorboard --logdir runs
```

### 3. Resume Training

Set `"preload": "latest"` (or a specific epoch string, e.g. `"03"`) in `config.py`,
then rerun:

```bash
python Train_Model.py
```

### 4. Inference / Translation

`greedy_decode()` in `Train_Model.py` shows the pattern for translating a single
sentence: encode the source, then autoregressively decode with the target
tokenizer until `[EOS]` or `max_len` is reached. You can reuse this function with
a saved checkpoint to translate new sentences.

## Architecture Summary

Following the original paper, with `N=6` layers, `d_model=512`, `8` attention heads,
and `d_ff=2048` by default (see `build_model()` defaults):

- **Embeddings**: Token embeddings scaled by `√d_model`, plus fixed sinusoidal
  positional encodings added before the encoder/decoder stacks.
- **Encoder**: `N` stacked layers, each with multi-head self-attention followed by
  a position-wise feed-forward network, each wrapped in a residual "Add & Norm"
  (`LayerConnection`).
- **Decoder**: `N` stacked layers, each with causally-masked multi-head
  self-attention, encoder–decoder cross-attention, and a feed-forward network.
- **Attention**: Scaled dot-product attention (`QKᵀ/√d_k`, masked, softmax) computed
  in parallel across heads and recombined via an output projection.
- **Output**: A final linear layer projects decoder outputs to vocabulary size,
  followed by `log_softmax` (`Generation` module), matched against `CrossEntropyLoss`.

## Known Limitations & Output Quality

This implementation was built primarily as a learning exercise to understand the
Transformer architecture, not as a production-grade translation model. A few things
to keep in mind:

- **Training scale is intentionally small.** The default config uses `batch_size: 3`
  and `num_epochs: 5`, which is far below what's typically used to get strong
  translation quality (the original paper trained for 100k+ steps on much larger
  batches across multiple GPUs/TPUs).
- **Hardware constraints.** This was trained on limited hardware, which is why the
  batch size and epoch count are kept low. On stronger hardware (a modern GPU with
  more VRAM), you can significantly increase `batch_size` for more stable gradients
  and faster epochs.
- **The current checkpoints/outputs in this repo are not the best achievable result.**
  Translations from the included weights (if any) should be treated as a proof of
  concept, not a benchmark of the architecture's real potential.
- **How to improve results:**
  - Increase `num_epochs` (e.g. 20–50+) in `config.py`.
  - Increase `batch_size` if you have more GPU memory available.
  - Train on a full/less-aggressively-filtered dataset, or a larger parallel corpus.
  - Consider learning rate warmup/scheduling (as in the original paper) instead of
    a fixed `lr`.
  - Use beam search instead of greedy decoding for better inference quality.
  - Train longer before relying on BLEU/WER/CER numbers — a handful of epochs on a
    small dataset will understate what the architecture is capable of.

## Resources & Acknowledgements

These resources were invaluable in helping me understand the Transformer
architecture well enough to implement it from scratch:

- [Coding a Transformer from scratch (video)](https://youtu.be/bCz4OMemCcA?si=BuRUZuCK__Xkh79t) —
  a walkthrough of implementing the Transformer architecture step by step.
- [The Transformer: Attention Is All You Need — Glass Box Medicine](https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/) —
  a clear, illustrated breakdown of the paper's core ideas.
- [The Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding) —
  a line-by-line PyTorch walkthrough of the paper, especially useful for the
  positional encoding and attention implementation details.

Huge thanks to the creators of these resources for making a fairly dense paper
approachable.

## Contribution

Contributions are welcome! If you'd like to help improve this project:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them with a clear message.
4. Push to your fork and open a pull request describing what you changed and why.
Bug reports and suggestions are just as welcome as code — feel free to open an issue.
you can also reach out via email at rseyednozadi@gmail.com.

## License

This project is licensed under the [MIT License](LICENSE) — see the `LICENSE` file for details. The original paper referenced in this repository remains the intellectual property of its respective authors and is included here for educational/reference purposes only.
