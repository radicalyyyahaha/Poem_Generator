import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
import pickle

import numpy as np
import random
import os


block_size = 1024
n_embd = 768
n_head = 12
n_layers = 12
dropout = 0.1

max_iters = 800001

batch_size = 8
learning_rate = 3e-4
eval_iters = 200
eval_interval = 100
save_iterval = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# RMSNorm form meta-LlaMA
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int = n_embd, eps: float = 1e-5):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def process_file(file_path, tokenizer, output_file):
    with open(file_path, 'r', encoding='utf-8') as f, open(output_file, 'ab') as out_f:
        for line in f:
            encoded_data = tokenizer.encode(line)
            np.array(encoded_data, dtype = np.int64).tofile(out_f)
            #pickle.dump(encoded_data, out_f)

def preprocessing(data_dir, bin_file, enc_model):

    tokenizer = tiktoken.encoding_for_model(enc_model)

    # inistial bin file
    if os.path.exists(bin_file):
        os.remove(bin_file)

    # for every txt fil in dir
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing {filename}")
            process_file(file_path, tokenizer, bin_file)

    with open('tokenizer_mapping.pkl', 'wb') as f:
        vocab_size = tokenizer.n_vocab
        print(f"Vocab size: {vocab_size}")
        pickle.dump({'vocab_size': vocab_size}, f)

    return vocab_size

def split_large_file(output_file, part_size_gb, unit):
    
    if unit == "GB":
        part_size_bytes = part_size_gb * 1024 * 1024 * 1024
    else:
        part_size_bytes = part_size_gb * 1024 * 1024
    
    # where the part files stored
    output_dir = "split_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Spliting bin file to small size")
    with open(output_file, 'rb') as file:
        i = 0
        while True:
            # read size
            chunk = file.read(part_size_bytes)
            if not chunk:
                break
            
            # part_X.bin in split_files dir
            chunk_file = os.path.join(output_dir, f"part_{i}.bin")
            with open(chunk_file, 'wb') as chunk_f:
                chunk_f.write(chunk)
            
            print(f"Part {i} saved as {chunk_file}")
            i += 1
    
    print(f"File split into {i} parts, each approximately {part_size_gb} GB.")
    return output_dir


def load_data_from_file(output_dir, split_ratio=0.8):
    # gathering all part file
    chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("part_")])
    

    total_chunks = len(chunk_files)
    split_index = int(split_ratio * total_chunks)
    print(f"Split index: {split_index}")

    # split
    train_data_files = chunk_files[:split_index]
    val_data_files = chunk_files[split_index:]
    
    print(f"train_data_files: {train_data_files}")
    print(f"val_data_files: {val_data_files}")
    
    # only file name
    return train_data_files, val_data_files


def get_batch_for_val(data):
    """
    read once for val
    """
    batches_x = []
    batches_y = []

    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1]) for i in ix])
        
        batches_x.append(x)
        batches_y.append(y)

    return batches_x, batches_y


def get_batch(split, train_data_files, val_data_files):
    # random choose
    if split == 'train':
        chosen_file = random.choice(train_data_files)
    else:
        chosen_file = random.choice(val_data_files)

    with open(chosen_file, 'rb') as file:
        data = np.fromfile(file, dtype=np.int64)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1]) for i in ix])

    x = x.to(torch.long)
    y = y.to(torch.long)
    x, y = x.to(device), y.to(device)

    return x, y


# not used 
def get_sinusoidal_position_encoding(seq_len, d_model):
    """
    generate sinusoidal_position_encoding

    Args:
    - seq_len: block_size
    - d_model: n_embd

    Returns:
    - pos_encoding: get the shape of (seq_len, d_model) 
    """
    # initial
    pos_encoding = np.zeros((seq_len, d_model))
    
    #  [0, 1, 2, ..., seq_len-1]
    position = np.arange(0, seq_len)[:, np.newaxis]
    
    # use different frequency 
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # even:sin, odd:cos
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    # numpy ---> tensor
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32).to(device)
    
    return pos_encoding


def apply_rope(q, k, seq_len, dim):
    # q, k: shape (B, T, head_size)
    # RoPE is applied to query and key embeddings.
    # `dim` is the embedding dimension for a single head.
    
    # Create sinusoidal position embeddings (cosine and sine components)
    pos = torch.arange(seq_len, dtype=torch.float32, device=q.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=q.device) * -(torch.log(torch.tensor(10000.0)) / dim))

    # Create the cosine and sine parts of the embeddings
    sin_emb = torch.sin(pos * div_term).unsqueeze(0).expand_as(q[..., ::2])
    cos_emb = torch.cos(pos * div_term).unsqueeze(0).expand_as(q[..., ::2])
    
    # Apply rotary position embeddings to q and k (for each dimension)
    q_rope = torch.zeros_like(q)
    k_rope = torch.zeros_like(k)

    q_rope[..., ::2] = q[..., ::2] * cos_emb - q[..., 1::2] * sin_emb
    q_rope[..., 1::2] = q[..., ::2] * sin_emb + q[..., 1::2] * cos_emb

    k_rope[..., ::2] = k[..., ::2] * cos_emb - k[..., 1::2] * sin_emb
    k_rope[..., 1::2] = k[..., ::2] * sin_emb + k[..., 1::2] * cos_emb
    
    return q_rope, k_rope

class Head(nn.Module):
    # one head self-attention

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # apply RoPE
        q, k = apply_rope(q, k, T, self.head_size)

        # scale
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = RMSNorm()
        self.ln2 = RMSNorm()

    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class decoder(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_f = RMSNorm()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        token_accumulator = []
        print(decode(idx[0].tolist()), end='', flush=True)
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            #policies
            idx_next = torch.multinomial(probs, num_samples=1) # sampling
            #idx_next = torch.argmax(probs, dim=-1, keepdim=True) # greedy search
            #idx_next = top_p_sampling(probs, p=0.9) # top-p sampling
            
            idx = torch.cat((idx, idx_next), dim=1)

            token_accumulator.append(idx_next[0].item())
            generated_token = decode(token_accumulator)
            
            # if you use char encoding for hanzi, you will never meet this problem
            if generated_token == "�":
                continue
            print(generated_token, end='', flush=True)
            token_accumulator.clear()
            

        return idx

def decode(l):
    enc = tiktoken.encoding_for_model("gpt-2")
    return ''.join(enc.decode(l))

def top_p_sampling(probs, p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()  
    sorted_indices_to_keep[..., 0] = True  
    
    probs_to_keep = sorted_probs * sorted_indices_to_keep
    probs_to_keep /= probs_to_keep.sum(dim=-1, keepdim=True) 
    
    idx_next = torch.multinomial(probs_to_keep, num_samples=1)
    
    return sorted_indices.gather(dim=-1, index=idx_next)




@torch.no_grad()
def eval_loss(train_data_files, val_data_files):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        if split == 'train':
            chosen_file = random.choice(train_data_files)
        else:
            chosen_file = random.choice(val_data_files)

        with open(chosen_file, 'rb') as file:
            data = np.fromfile(file, dtype=np.int64)

        batches_x, batches_y = get_batch_for_val(data)

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batches_x[k].to(device), batches_y[k].to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out
    

def train():
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = eval_loss(train_data_files, val_data_files)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")

        if iter % save_iterval == 0:
                torch.save(model.state_dict(), f'./ckpt/model_{iter}.pt')

        xb, yb = get_batch('train', train_data_files, val_data_files)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    

if __name__ == '__main__':

    print(device)

    data_dir = './data'
    bin_file = 'encoded_data.bin'

    enc_model = "gpt-2" #gpt-3.5-turbo/gpt-4o (https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)

    split_ratio = 0.8 #ratio of train data size

    part_size=30 # size of part file
    unit = "GB" # or MB

    '''
    Preprocessing
    1. encode file to a big .bin
    2. apart .bin into small parts(around 30GB) in order to read in memory
    3. split train and val files

    if you find your file small enough, just read it and figure out len, then use len to split train and val
    '''
    vocab_size = preprocessing(data_dir, bin_file, enc_model)
    output_dir = split_large_file(bin_file, part_size, unit)
    train_data_files, val_data_files = load_data_from_file(output_dir)

    # initial model
    model = decoder(vocab_size).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-6)

    train()
