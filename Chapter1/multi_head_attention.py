import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model int = 512,
        num_heads int = 8,
        dropout float = 0.1,
        bias bool = True
    ):
        supper.__init__()

        assert d_model % num_heads == 0, "d_model must be divsible by num_heads"

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
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
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
        batch_size = query.shape[0]

        # Linear projections
        Q = self.q_proj(query)  # (batch_size, seq_len, d_model)
        K = self.k_proj(key)    # (batch_size, seq_len, d_model
        V = self.v_proj(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)  
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        if causal:
            seq_len = query.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_score = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_score, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)  # (batch_size, seq_len, d_model)

        return output         
        
        
class FeedForward(nn.Module):
    def __init__(
        self,
        d_model int = 512,
        d_ff int = 2048,
        dropout float = 0.1,
        activation = F.relu,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for feed-forward network.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        d_model int = 512,
        num_heads int = 8,
        d_ff int = 2048,
        dropout float = 0.1,
        activation = F.relu,
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass for Transformer block.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-Head Attention
        attn_output = self.mha(x, x, x, causal=True)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x