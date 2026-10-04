# Mini GPT from scratch (PyTorch)

A small GPT-style language model (decoder-only transformer) written in PyTorch to understand how the architecture works. It is trained on 100 short English sentences, so it is a learning project and not a useful language model.

Built while following an online course on building a mini GPT with PyTorch: [Build a Mini GPT From Scratch with PyTorch | Step-by-Step LLM Tutorial / [link](https://youtu.be/5BQZEQUL9-k?si=dn-LrDdeBMceOoQf)].

## What it does

The script runs the whole pipeline in one file:

1. **Tokenizer:** a SentencePiece BPE tokenizer (vocabulary of 320 tokens) is trained on `corpus.txt`.
2. **Embeddings:** each token id is mapped to a 32-dimensional vector (token embedding) and a learned position embedding is added.
3. **Transformer blocks:** 2 blocks, each with layer norm, masked multi-head self-attention (2 heads), layer norm and a feed-forward network, connected by residual connections (pre-norm).
4. **Language modeling head:** a final layer norm and a linear layer produce one score per token; softmax turns them into probabilities for the next token.
5. **Training:** next-token prediction with cross-entropy loss and the AdamW optimizer.
6. **Generation:** tokens are sampled one at a time with temperature and top-k.

## Configuration

| Setting | Value |
|---|---|
| Vocabulary size | 320 (SentencePiece BPE) |
| Context window (`block_size`) | 16 tokens |
| Embedding size (`embedding_dim`) | 32 |
| Attention heads (`n_heads`) | 2 |
| Transformer blocks (`n_layers`) | 2 |
| Learning rate | 1e-3 (constant) |
| Training steps (`epochs`) | 5000, batch size 16 |

## Project structure

```
.
├── L3/
│   ├── tinyGPT_tokenizer.py  # tokenizer, model, training, generation
│   ├── transformer_block.py  # attention head, multi-head attention, feed forward, block
│   └── corpus.txt            # 100 training sentences, one per line
├── requirements.txt
└── README.md
```

## Setup and run

Requires Python 3.10 or newer. The model is small and runs on the CPU, so no GPU is needed.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
# Windows (PowerShell):  venv\Scripts\Activate.ps1
# Linux / macOS:         source venv/bin/activate

pip install -r requirements.txt
python L3/tinyGPT_tokenizer.py
```

The script prints the training loss every 300 steps, then a few generated texts, and finally how many of 200 generated samples are copied word for word from the corpus. It also writes `tokenizer.model` and `tokenizer.vocab` into the `L3` folder.

## Results

- After 5000 steps the training loss is about 0.2 to 0.25 (across several runs). The printed loss comes from a single random batch, so it fluctuates.
- In one run, 119 of 200 samples (about 60 %) generated from the prompt `the` with temperature 1.0 and top-k 40 were copied word for word from the corpus. The number changes between runs because weights and sampling are random.

Example outputs:

```
the museum in Madrid opens at nine he drives a red
```
Copied: two consecutive training sentences.

```
the bus to the airport in Atlanta is very busy our dog l
```
Not in the corpus: it joins "the bus to the airport ..." and "the airport in Atlanta is very busy".

```
the coffee in Persailles isckister stctws h tra a lovel
```
Made-up word "Persailles" (pieces of "Paris" and "Versailles"), followed by noise. This is a sample with temperature 1.2 and top-k 40.

## Observations and limitations

- With only 100 sentences the model mostly memorizes the corpus, including the order of the sentences. Lower temperature and smaller top-k make this more visible.
- There is no end-of-sentence token, so generation simply continues into the next sentence.
- Subword tokens can combine into words that do not exist.
- The loss is measured on the training data only. There is no validation set, so overfitting is not measured.

## Ideas for next steps

- Train on a larger dataset (for example TinyStories or Tiny Shakespeare).
- Add an end-of-sentence token and stop generation when it appears.
- Add a validation split and a learning rate schedule.
- Move model and data to the GPU.
- Compare with an encoder-only variant (no causal mask).
