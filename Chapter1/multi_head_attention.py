import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time

class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute the sinusoidal frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.arange(0, max_seq_len).float()
        freqs = torch.einsum("i,j->ij", position, inv_freq)  # (max_seq_len, head_dim/2)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        Apply Rotary Positional Embeddings to the input tensor.
        """
        B, H, S, D = x.shape  # Batch, Num_heads, Seq_len ,Head_dim
        assert D == self.head_dim, "Head dimension mismatch"
        if S + offset > self.max_seq_len:
            raise ValueError("Sequence length exceeds maximum length for RoPE")
        
        cos = self.cos_cache[offset : offset + S][None, None, :, :] # (1, 1, S, D/2)
        sin = self.sin_cache[offset : offset + S][None, None, :, :] # (1, 1, S, D/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_max_seq_len: int = 2048,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output Projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # RoPE only for Q, K
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(self.head_dim, base=rope_base, max_seq_len=rope_max_seq_len)
        
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        offset: int = 0,
    ):
        """
        Forward pass for multi-head attention.
        Args:
            query: Tensor of shape (batch_size, seq_len, d_model)
            key: Tensor of shape (batch_size, seq_len, d_model)
            value: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        B, S, _ = query.shape

        # Linear projections
        Q = self.q_proj(query)  # (batch_size, seq_len, d_model)
        K = self.k_proj(key)    # (batch_size, seq_len, d_model)
        V = self.v_proj(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)  
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # if using RoPE, apply to Q and K
        if self.use_rope:
            Q = self.rope(Q, offset=offset)
            K = self.rope(K, offset=offset)   
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        if causal:
            mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)  # [S,S]
            scores = scores.masked_fill(mask, float("-inf"))
        attn_score = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attn_score = self.attn_dropout(attn_score)
        attn_output = torch.matmul(attn_score, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)  # (batch_size, seq_len, d_model)

        return output         
        
        
class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for feed-forward network.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        offset : int = 0,
    ):
        """
        Forward pass for Transformer block.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-LN attention
        attn_out = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), causal=True, offset=offset)
        x = x + self.dropout(attn_out)

        # Pre-LN FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x
class MiniGPT(nn.Module):
    """
    a Mini GPT model with multi-head attention and feed-forward layers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.1,
        rope_max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Transformer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, S) int64 token ids
        return logits: (B, S, V)
        """
        x = self.tok_emb(idx)             # (B,S,C)
        x = self.drop(x)

        # offset is 0 since we always process from the start in this simple example
        for blk in self.blocks:
            x = blk(x, offset=0)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


@torch.no_grad()
def estimate_metrics(model, device, vocab_size, seq_len, batches=50, batch_size=32):
    """
    Estimate loss, perplexity, and accuracy on random data.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for _ in range(batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = x[:, 1:].contiguous()
        x_in = x[:, :-1].contiguous()

        logits = model(x_in)  # (B, S-1, V)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()
        total_loss += loss.item()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(20.0, avg_loss))  # overflow safeguard
    acc = total_correct / total_tokens
    return avg_loss, ppl, acc


@torch.no_grad()
def greedy_generate(model, device, prompt, max_new_tokens=32):
    """
    prompt: (1, S) token ids
    """
    model.eval()
    idx = prompt.clone().to(device)

    for _ in range(max_new_tokens):
        logits = model(idx)               # (1, S, V)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


def main():
    # --------- hyperparameters ----------
    vocab_size = 128        # toy vocab
    seq_len = 64            # sequence length
    batch_size = 64
    steps = 2000            # training steps
    lr = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0
    log_every = 100
    eval_every = 500

    # model strcuture
    d_model = 256
    num_heads = 8
    d_ff = 1024
    n_layers = 4
    dropout = 0.1

    # --------- device ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    # --------- model and optimizer ----------
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # warmup + cosine lr decay
    def lr_schedule(step):
        warmup = 200
        if step < warmup:
            return (step + 1) / warmup
        # cosine decay to 10%
        t = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

    # --------- Training Loop ----------
    model.train()
    t0 = time.time()
    ema_loss = None

    for step in range(1, steps + 1):
        # generate random data
        # x: (B, S) ; y: (B, S-1) next token
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = x[:, 1:].contiguous()
        x_in = x[:, :-1].contiguous()

        logits = model(x_in)  # (B, S-1, V)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update lr
        scale = lr_schedule(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * scale

        optimizer.step()

        # calculate accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()

        # loss smoothing
        l = loss.item()
        ema_loss = l if ema_loss is None else (0.9 * ema_loss + 0.1 * l)
        ppl = math.exp(min(20.0, ema_loss))

        if step % log_every == 0 or step == 1:
            dt = time.time() - t0
            tok_per_step = batch_size * (seq_len - 1)
            print(
                f"[train] step {step:5d}/{steps} | "
                f"loss {l:.4f} (ema {ema_loss:.4f}) | ppl~{ppl:.2f} | acc {acc*100:.2f}% | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"{dt:.1f}s"
            )

        if step % eval_every == 0:
            val_loss, val_ppl, val_acc = estimate_metrics(
                model, device, vocab_size, seq_len, batches=30, batch_size=batch_size
            )
            print(
                f"[eval ] step {step:5d} | "
                f"loss {val_loss:.4f} | ppl {val_ppl:.2f} | acc {val_acc*100:.2f}%"
            )
            model.train()

    # --------- training done ----------
    # --------- random generation demo ----------
    prompt = torch.randint(0, vocab_size, (1, 8), device=device)
    out = greedy_generate(model, device, prompt, max_new_tokens=24)

    print("\nPrompt token ids:", prompt[0].tolist())
    print("Generated ids    :", out[0].tolist())

    # save model(optional)
    # torch.save(model.state_dict(), "minigpt.pt")


if __name__ == "__main__":
    main()