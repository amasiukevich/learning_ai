import torch
import torch.nn as nn
import torch.nn.functional as F



# hyperparameters

# ==============================
batch_size = 64
block_size = 256

max_iters = 5001
eval_interval = 10

learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # 6 heads x 64
n_head = 6
n_layer = 6
dropout = 0.2
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


class Head(nn.Module):

    """ one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False) 

        # Not the parameter of the module, registered as buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):

        B, T, C = x.shape

        key_val = self.key(x)
        query_val = self.query(x)

        # compute attention scores ("affinities")

        wei = query_val @ key_val.transpose(-2, -1) * C**-0.5 #  (B, T, head_size) @ (B, head_size, T) => (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values

        value_result = self.value(x) # (B, T, C)
        out = wei @ value_result

        return out


class MultiHeadAttention(nn.Module):

    """ multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # comes from the Original Transformer Paper

    def forward(self, x): 
        
        out = torch.cat([head(x) for head in self.heads], dim=-1) + self.proj(x)
        out = self.proj(out)
        return out



class FeedForward(nn.Module):

    """ a simple linear layer followed by a non-linearity """


    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # growing layer dim as in the Paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd) # projection layer
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    """ Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LayerNorm1d:

    def __init__(self, dim, eps=1e-5):
        
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    

    def __call__(self, x):

        x_mean = x.mean(axis=1, keepdim=True)
        x_var = x.var(axis=1, keepdim=True)

        xhat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # n_blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        
        B, T = idx.shape


        # idx and targets are both (B, T) - tensors of integers (batch, time, channel)
        tok_emb = self.token_embedding_table(idx) # (B, T, C_emb)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)

        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

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
            # cropping the context to the block_size (position embedding otherwise will fail)
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

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
model = BigramLanguageModel()
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

