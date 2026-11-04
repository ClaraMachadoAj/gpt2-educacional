"""Treina, avalia e salva um modelo pequeno para experimentos locais."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig


def batch_from(data, batch_size, context, generator, device):
    # Uma janela precisa de T+1 bytes para formar T pares entrada/alvo.
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    windows = data[starts[:, None] + torch.arange(context + 1)]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("gpt2.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.context) < 1:
        parser.error("steps, batch-size e context devem ser positivos.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    config = GPTConfig(context=args.context)
    data = torch.tensor(list(args.data.read_bytes()), dtype=torch.long)
    cut = len(data) * 9 // 10
    train, validation = data[:cut], data[cut:]
    if min(len(train), len(validation)) <= config.context:
        parser.error("O corpus deve ter pelo menos context + 1 bytes em cada divisão.")

    model = GPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    training_rng = torch.Generator().manual_seed(args.seed)

    for step in range(1, args.steps + 1):
        x, y = batch_from(train, args.batch_size, config.context, training_rng, args.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            # Reutiliza as mesmas janelas de validação em cada medição.
            model.eval()
            validation_rng = torch.Generator().manual_seed(args.seed + 1)
            with torch.no_grad():
                losses = []
                for _ in range(4):
                    vx, vy = batch_from(
                        validation, args.batch_size, config.context, validation_rng, args.device
                    )
                    _, val_loss = model(vx, vy)
                    losses.append(val_loss)
                mean_validation = torch.stack(losses).mean().item()
            model.train()
            print(f"passo {step:04d} | treino {loss.item():.4f} | validação {mean_validation:.4f}")

    torch.save({"config": asdict(config), "weights": model.state_dict()}, args.output)
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()

