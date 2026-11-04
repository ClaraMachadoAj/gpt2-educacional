# GPT-2 educacional em PyTorch

Implementação de um Transformer decoder que prevê o próximo token. A configuração padrão tem 2 blocos, 4 cabeças, largura 64, janela de 64 tokens e vocabulário de 256 bytes. O projeto treina do zero com um corpus local e usa as operações do PyTorch para implementar a arquitetura.

## Rodar localmente

Requer Python 3.10 ou superior e uma versão compatível do PyTorch. Na pasta do projeto:

```sh
python -m venv .venv
```

Ative com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` em Linux/macOS. Depois:

```sh
python -m pip install -r requirements.txt
python test_model.py -v
python train.py --steps 100 --output gpt2.pt
python generate.py gpt2.pt "Na biblioteca" --tokens 80 --top-k 20
```

Para um teste rápido, use `python train.py --steps 2 --output gpt2.pt`. O treinamento usa CPU por padrão. É possível usar `--device cuda` com uma instalação compatível. Para outro corpus, passe `--data caminho.txt` e use um texto UTF-8 com pelo menos `context + 1` bytes em cada divisão. O arquivo `gpt2.pt` é criado pelo treinamento.

## Organização

| Arquivo | Função |
| --- | --- |
| model.py | Embeddings, atenção causal, blocos, logits, perda e geração |
| train.py | Janelas deslocadas, treino, validação e checkpoint |
| generate.py | Carregamento do modelo e geração a partir de um prefixo |
| corpus.txt | Corpus sintético e repetitivo para demonstração |
| test_model.py | Seis testes sobre formas, perda, causalidade, gradientes, geração, dados e configuração |
| relatorio.md | Versão editável do relatório técnico |

O corpus repete quatro frases, inclusive na divisão de validação. As perdas ilustram o funcionamento do treinamento e da avaliação em janelas separadas, sem medir generalização para texto novo. Um token é um byte. Um caractere acentuado pode ocupar vários tokens e amostras UTF-8 inválidas aparecem com o caractere de substituição.

## Publicação no GitHub

Crie um repositório na sua conta e envie os arquivos desta pasta, incluindo o README e o relatório. A URL exibida pelo GitHub após o envio é o link da entrega. O checkpoint local pode ser recriado pelos comandos acima.

## Referências

- [Andrej Karpathy, Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU)
- [Código de referência da aula](https://github.com/karpathy/build-nanogpt)
- [Implementação oficial do GPT-2](https://github.com/openai/gpt-2)

