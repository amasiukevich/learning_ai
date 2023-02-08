import torch
import torch.nn as nn
import torch.nn.functional as F



# hyperparameters

# ==============================
batch_size = 32
block_size = 8

max_iters = 3000
eval_interval = 300

learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ===============================


# reproducibility
torch.manual_seed(1337)


# data loading
with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# all uique variables
chars = sorted(list(set(text)))
vocab_size = len(chars)


# mappings (encode/decode)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take a string and encode it
decode = lambda vec: "".join([itos[pos] for pos in vec]) # take a list of integers, output a string


# train-validation split
data = torch.tensor(encode(text), dtype=torch.long)

train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]


### Making minibatches of sequences

def get_batch(split):

    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # getting random indexes from the dataset
    x = torch.stack([data[i: (i + block_size)] for i in ix]) # getting the batch_size random sequences of size block_size - to be processed for contexts
    y = torch.stack([data[(i + 1): (i + block_size + 1)] for i in ix]) # getting the targets for that sequences
    
    x, y = x.to(device), y.to(device)
    return x, y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) - tensors of integers (batch, time, channel)
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # reshaping
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets) # Neg-Log-Likelihood loss
        
        return logits, loss

    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

# TODO: Where is the training???
model = BigramLanguageModel(vocab_size)
model = model.to(device)

# creating PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# training loop

for iter in range(max_iters):
    xb, yb = get_batch('train')

    # evaluate the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(context, max_new_tokens=300)[0].tolist()

decoded_sequence = decode(generated)
print(decoded_sequence)