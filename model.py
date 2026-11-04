"""Transformer decoder com os componentes centrais do GPT-2."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256
    context: int = 64
    layers: int = 2
    heads: int = 4
    width: int = 64

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.layers, self.heads, self.width) < 1:
            raise ValueError("As dimensões devem ser positivas.")
        if self.width % self.heads:
            raise ValueError("A dimensão interna deve ser divisível pelo número de cabeças.")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        mask = torch.ones(config.context, config.context, dtype=torch.bool).tril()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        batch, time, width = x.shape
        head_width = width // self.heads
        # Separa Q, K e V antes de mover as cabeças para o segundo eixo.
        projected = self.qkv(x).reshape(batch, time, 3, self.heads, head_width)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (q @ k.transpose(-2, -1)) / head_width**0.5
        visible = self.mask[:time, :time]
        scores = scores.masked_fill(~visible, float("-inf"))
        attention_weights = scores.softmax(dim=-1)
        head_outputs = attention_weights @ v
        combined = head_outputs.transpose(1, 2).reshape(batch, time, width)
        return self.output(combined)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class GPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.head.weight = self.tokens.weight
        # Reduz a contribuição inicial das projeções nos ramos residuais.
        residual_std = 0.02 / (2 * config.layers)**0.5
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=residual_std)
            nn.init.normal_(block.mlp[-1].weight, std=residual_std)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.size(1) <= self.config.context:
            raise ValueError("Esperado tensor [lote, tempo] dentro da janela de contexto.")
        if targets is not None and targets.shape != ids.shape:
            raise ValueError("Os alvos precisam ter a mesma forma dos IDs.")
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, new_tokens, temperature=1.0, top_k=None):
        if ids.ndim != 2 or ids.size(1) == 0:
            raise ValueError("Forneça um prefixo não vazio por sequência.")
        if new_tokens < 0 or temperature <= 0:
            raise ValueError("Quantidade negativa ou temperatura inválida.")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k precisa estar entre 1 e o tamanho do vocabulário.")
        for _ in range(new_tokens):
            logits, _ = self(ids[:, -self.config.context:])
            # A última posição prevê o byte logo depois do prefixo.
            scores = logits[:, -1, :] / temperature
            if top_k is not None:
                values, indices = scores.topk(top_k, dim=-1)
                scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, values)
            next_id = torch.multinomial(scores.softmax(dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
        return ids

