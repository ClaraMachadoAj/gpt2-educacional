"""Carrega um checkpoint local e completa um prefixo em UTF-8."""
import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt:
        parser.error("Informe um prefixo não vazio.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    prefix = torch.tensor([list(args.prompt.encode("utf-8"))], dtype=torch.long)
    result = model.generate(prefix, args.tokens, args.temperature, args.top_k)
    print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()

