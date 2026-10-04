import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Always work in the folder containing this file.


from transformer_block import Block #code related to transformer


print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# first version (kept for reference): word-level tokenizer on 10 sentences, every word between spaces is one token
# corpus = [
#         "hello friends how are you",
#         "the tea is very hot",
#         "my name is Steven",
#         "the roads of Delhi are busy",
#         "it is raining in Mumbai",
#         "the train is late again",
#         "i love eating samosas and drinking tea",
#         "holi is my favorite festival",
#         "diwali brings lights and sweets",
#         "india won the cricket match"
# ]

# # <END> is used to indicate the boundary of a sentence (How much is there in one sentence)
# corpus = [s + " <END>" for s in corpus]
# text = " ".join(corpus) # join all the sentences in the corpus into a single string with spaces in between
# print(text)

# # divide the text into numbers
# words = list(set(text.split()))
# print(words)

# # how many unique words are there in the text
# vocab_size = len(words)
# print("Vocab size:", vocab_size)

# # convert words to index
# word2idx = {w: i for i, w in enumerate(words)}
# print("word2idx : ", word2idx)

# # convert index to words
# idx2word = {i: w for w, i in word2idx.items()}
# print("idx2word : ", idx2word)

# # convert the text into a tensor of indices
# data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
# print(data)
# print(len(data)) #62


import sentencepiece as spm

# load and read corpus (plain text, one sentence per line)
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Train a SentencePiece Tokenizer (it is retrained on every run and the files below are overwritten)

spm.SentencePieceTrainer.Train(
    input="corpus.txt",             # path to the input text file
    model_prefix="tokenizer",       # prefix for the output model files
    vocab_size=320,                  # size of the vocabulary (incl. the special tokens <unk>, <s>, </s>); BPE keeps merging frequent pairs until this size is reached
    model_type="bpe",               # type of the model (bpe, unigram, char, word)
)
# After training, the following files will be generated:
    # tokenizer.model (trained model)
    # tokenizer.vocab (tokens/ vocabulary file)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")          # load the trained tokenizer model

ids = sp.encode(text, out_type=int) # encode the text into a list of token ids
data = torch.tensor(ids, dtype=torch.long) # convert the list of token ids into a tensor: one long stream of token ids, shape (len(data),)
print (data)

vocab_size = sp.get_piece_size()     # get the size of the vocabulary
print("Vocab size:", vocab_size)



# get batch function to give data to model in batches
block_size = 16             # (Context Window) how much data that LLM can see, before generating the next token or the response (here, before generating a new token, it sees the previous 16 tokens - tokens, not words: one sentence is about 16 tokens with this tokenizer)
embedding_dim = 32          # (d_model) For each token there will be a 1d vector, a list for each token, and that list will have 32 floating point values. Initially these will be random values, but as the model trains, these values will be updated to represent the meaning of the token in the context of the training data. This is called embedding. The vector keeps this length in every layer of the model.
n_heads = 2                 # how many self attention head we use in the transformer block. Each head will learn to focus on different parts of the input sequence, allowing the model to capture different aspects of the input data. Each head works with head_size = embedding_dim // n_heads = 16 values (embedding_dim must be divisible by n_heads).
n_layers = 2                # how many transformer blocks we use in the model. Each block will consist of a self attention layer and a feed forward layer, and the output of one block will be the input to the next block. By stacking multiple blocks, the model can learn more complex representations of the input data.
lr = 1e-3                   # learning rate, which is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. A smaller learning rate means the model will learn more slowly, but it can lead to better convergence and stability. A larger learning rate means the model will learn faster, but it can lead to overshooting the optimal solution and instability. It stays constant here, which is one reason why the printed loss keeps jumping around a bit.
epochs = 5000               # number of training STEPS (not real epochs): each step processes one batch of 16 random sequences. A real epoch would be one pass over the whole dataset. 5000 steps x 16 sequences x 16 tokens = 1.28M tokens = about 780 passes over the ~1650 tokens of the corpus. The number of steps is a hyperparameter that can be tuned to achieve better performance on the training data.

# get_batch function to give data to model in batches
def get_batch(batch_size=16):                                  # provide 16 examples to LLM in one go (16 sequences)
    ix = torch.randint(len(data) - block_size, (batch_size,))   # len(data) (about 1650) - 16 (block_size) = about 1630 possible start indexes, starting from 0 (the last one still leaves room for y, which is shifted by 1). We randomly select 16 (batch_size) of them. These indexes will be used to select the sequences of tokens from the data tensor.
    # [12,23,45,...,4] 16 random starting indexes, 16 examples, 16 sequences of tokens to model in one batch.
    # 12(token12, t13, t14,...,t27) 16 tokens (block_size=16), 1 sequence of tokens, 1 example
    x = torch.stack([data[i:i+block_size] for i in ix])         # input: block_size tokens starting at each random index i → shape (batch_size, block_size) = (16, 16)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])     # target: same window shifted by 1 position → y[t] is the correct next token for x[t], same shape (16, 16)
    # x  [token_12, token_13, token_14, ..., token_27]
    # y  [token_13, token_14, token_15, ..., token_28] (Token_28 is the correct next token for the token_27)

    return x, y

class TinyGPT(nn.Module):   # Name of Class (nn.Module = base class of every PyTorch model/layer)
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)      # (320 vocab_size, 32 embedding_dim) For all 320 tokens we have a 1d vector of 32 floating point values (lookup table: row i = embedding of token id i)

        self.position_embedding = nn.Embedding(block_size, embedding_dim)   # (16 block_size, 32 embedding_dim(for each of the number, the position will be provided)) position embedding added for the order of the tokens in the sequence, as the model needs to know the position of each token in the input sequence to make accurate predictions. The position embedding is a learnable parameter (like in GPT-2; the original paper used fixed sin/cos values instead) that is added to the token embedding to provide information about the position of each token in the sequence.
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) # n_layers (2) transformer blocks one after the other; each block = layer norm + masked multi-head attention + layer norm + feed forward, with residual connections (pre-norm)

        self.ln_f = nn.LayerNorm(embedding_dim) # final layer norm after the last block (pre-norm models need this extra norm at the end)
        self.head = nn.Linear(embedding_dim, vocab_size) # language modeling head: maps each 32-dim vector to 320 scores, one per token (no weight tying here: it is a separate matrix from token_embedding)

    def forward(self, idx, targets=None):
        B,T = idx.shape # 16, 16 (batch_size, block_size) during training; during generation B is 1 and T is at most 16
        tok_emb = self.token_embedding(idx) # convert numbers to vectors, (16, 16, 32) (B, T, C) with C = embedding_dim

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (16, 32) (T, C): the positions 0..T-1, the same for every sequence of the batch (it gets broadcast over B when added below)
        x = tok_emb + pos_emb   # x = input to transformer, which has token embedding and position embedding added together, (16, 16, 32) (B, T, C)
        x = self.blocks(x) # input going through transformer blocks, (16, 16, 32) (B, T, C) - the shape stays the same in every block (residual stream)
        x = self.ln_f(x) 
        logits = self.head(x) # raw prediction (scores) of a model for all the 320 tokens in the vocabulary, (16, 16, 320) (B, T, vocab_size)
        loss = None # calculate loss 
        if targets is not None:
            B, T, C = logits.shape # values, model predicted, 16, 16, 320 - careful: here C = vocab_size (320), not embedding_dim
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) # flatten to (256, 320) scores vs (256,) correct next tokens: compares the prediction with the correct next token at every position (cross_entropy applies the softmax internally)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):    # generating the next token based on the previous tokens, idx is the input sequence of token ids (shape (1, T) here), max_new_tokens is the number of new tokens to generate
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # get the last block_size tokens from the input sequence (the model can only look at block_size tokens), (1, T) with T <= 16 (B, T)
            logits, _ = self(idx_cond) # get the model's predictions for the next token, (1, T, 320) (B, T, C) - self(...) calls forward()
            logits = logits[:, -1, :] # focus only on the last time step, (1, 320) (B, C)
            logits = logits / temperature # temperature < 1 makes the distribution sharper (safer choices), > 1 makes it flatter (more random)
            v, _ = torch.topk(logits, top_k) # the top_k highest scores
            logits[logits < v[:, [-1]]] = float('-inf') # forbid every token that is not among the top_k candidates (softmax turns -inf into probability 0)
            probs = F.softmax(logits, dim=-1) # convert logits to probabilities, (1, 320) (B, C)
            next_idx = torch.multinomial(probs, 1) # sample the next token from the probability distribution, (1, 1) (B, 1)    # e.g. "coffee"
            idx = torch.cat((idx, next_idx), dim=1) # append the sampled token to the input sequence, (1, T+1) (B, T+1)          # e.g. "the coffee"
        return idx



model = TinyGPT() # note: model and data stay on the CPU (no .to("cuda")), which is fine for a model this small
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW: a variant of the Adam optimizer (adaptive step size per parameter)

for step in range(epochs):
    xb, yb = get_batch() # one random batch
    logits, loss = model(xb, yb) # forward pass: prediction + loss
    optimizer.zero_grad() # reset the gradients of the previous step
    loss.backward() # backward pass (backpropagation): compute the gradients
    optimizer.step() # update the weights
    if step % 300 == 0:
        print(f"Step {step}: loss {loss.item():.4f}") # loss of ONE random batch, so it fluctuates from print to print



# context = torch.tensor([[word2idx["hello"]]], dtype=torch.long)  # starting word for generation
# out = model.generate(context, max_new_tokens=15)  # generate 15 new tokens based on the context

# print(" ".join(idx2word[int(i)] for i in out[0])) 


import sentencepiece as spm # (already imported above, this second import is not needed)
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# corpus as one long text, to check if an output is copied word for word
corpus_text = open("corpus.txt", encoding="utf-8").read().replace("\n", " ")

def check(label, out):
    s = sp.decode(out[0].tolist())
    print(f"{label}: {s}")
    print("   same as in corpus:", s in corpus_text)

context = torch.tensor([sp.encode("the")], dtype=torch.long) # start text "the" encoded as token ids, shape (1, number of tokens). It should appear in the corpus: "the" starts about half of the sentences, a capital "The" would be split into pieces the model never saw at a sentence start


out = model.generate(context, max_new_tokens=20)
check("standard", out) # same settings as "middle" (the defaults temperature=0.8, top_k=10)

# careful: few candidates, favourite wins most of the time
out = model.generate(context, max_new_tokens=20, temperature=0.6, top_k=5)
check("careful", out)

# wild: more candidates, more randomness
out = model.generate(context, max_new_tokens=20, temperature=1.2, top_k=40)
check("wild", out)

out = model.generate(context, max_new_tokens=20, temperature=0.8, top_k=10)
check("middle", out)


# count how many of n_samples generated texts are copied word for word from the corpus
n_samples = 200
n_copy = 0
for _ in range(n_samples):
    out = model.generate(context, max_new_tokens=20, temperature=1.0, top_k=40)
    n_copy += sp.decode(out[0].tolist()) in corpus_text
print(f"{n_copy}/{n_samples} samples are copied word for word from the corpus")