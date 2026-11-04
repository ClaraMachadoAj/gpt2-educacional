# Implementação do GPT-2 em PyTorch

## Da entrada aos embeddings

Este projeto implementa em PyTorch um modelo que aprende a prever o próximo byte de um texto. O objetivo é acompanhar o caminho da entrada até a previsão e verificar o que acontece em cada etapa. O modelo tem dois blocos, quatro cabeças de atenção, largura de 64 e uma janela de até 64 bytes. Essa configuração permite rodar o experimento em CPU.

Os scripts representam cada byte UTF-8 por um inteiro de 0 a 255. Por exemplo, "gato" vira [103, 97, 116, 111], enquanto "é" ocupa dois bytes, [195, 169]. Essa escolha dispensa um tokenizador externo e muda a unidade de previsão em relação ao GPT-2 original, que usa BPE sobre bytes.

A entrada de GPT.forward é uma matriz [B, T] de IDs inteiros, com B sequências e T tokens por sequência. A tabela tokens transforma cada ID em um vetor de C valores. A tabela positions representa os índices de 0 a T-1 e produz [T, C]. O PyTorch soma os mesmos vetores de posição a cada sequência do lote. O resultado tem forma [B, T, C].

O embedding dá ao modelo uma representação treinável de cada símbolo. A informação de posição permite distinguir ocorrências do mesmo símbolo em locais diferentes e fornece ao modelo uma noção da ordem. Depois da soma, os blocos transformam os vetores mantendo sua forma.

## O que a máscara muda na atenção

CausalSelfAttention projeta [B, T, C] em [B, T, 3C] e divide a última dimensão em Q, K e V. Após reorganizar os eixos, cada tensor tem forma [B, H, T, d]. A consulta Q representa a informação que uma posição procura. A chave K determina sua compatibilidade com a consulta e o valor V carrega a informação que será combinada.

O produto QKᵀ gera [B, H, T, T]. A linha i contém as compatibilidades da posição i com as posições de origem. Dividir por √d reduz a escala dos produtos à medida que a dimensão cresce, o que ajuda a evitar a saturação do softmax.

A posição i pode consultar os tokens de 0 até i. Em “gat”, por exemplo, a posição do “a” vê “g” e “a”. O “t” só aparece como alvo dessa posição durante o treino.

Considere a entrada “gat”. Para a consulta da posição 1, suponha que os escores já escalados sejam [0, 1, 2]. São valores ilustrativos. Antes da máscara, a terceira chave teria o maior escore. Como ela corresponde ao “t”, que ainda está no futuro dessa consulta, o código transforma a linha em [0, 1, −∞].

O softmax resulta em aproximadamente [0,27; 0,73; 0]. Se os dois primeiros valores fossem vetores [2, 0] e [0, 2], a saída da cabeça nessa posição seria aproximadamente [0,54; 1,46]. Mudar o terceiro valor deixaria essa conta igual.

O resultado tem um vetor por cabeça em cada posição. O código reúne as quatro cabeças e aplica uma projeção linear para voltar à largura original de 64. Assim, o bloco seguinte recebe um tensor com o mesmo formato.

## O restante do bloco e a saída

O bloco aplica LayerNorm antes de cada ramo. A normalização usa a média e a variância das C características de cada posição, seguidas de escala e deslocamento aprendidos. Isso ajusta a escala dos valores de cada token separadamente.

O código calcula x + atenção(norm(x)) e, com o resultado atualizado, x + mlp(norm(x)). As somas preservam um caminho direto para a representação e para os gradientes. O MLP opera em cada posição de modo independente. Ele expande C para 4C, aplica GELU e projeta de volta para C. A GELU introduz não linearidade, permitindo combinar características além do que duas projeções lineares poderiam representar juntas.

A LayerNorm final prepara as representações para a cabeça linear, que retorna um escore para cada ID do vocabulário. A cabeça usa a mesma matriz do embedding de tokens. Assim, a linha de um ID serve tanto para representá-lo na entrada quanto para pontuá-lo na saída. O compartilhamento também reduz o número de parâmetros.

Para um lote de duas sequências com cinco tokens cada, a entrada tem forma [2, 5] e os embeddings têm forma [2, 5, 64]. Em cada bloco, Q, K e V têm forma [2, 4, 5, 16] e a matriz de atenção tem forma [2, 4, 5, 5]. Depois do último bloco e da cabeça, os logits têm forma [2, 5, 256]. São dez previsões, cada uma com um escore para cada byte possível.

## Como o modelo aprende

Imagine uma janela de cinco bytes que contém “gatos”. Com contexto quatro, x recebe “gato” e y recebe “atos”. As quatro previsões são g → a, ga → t, gat → o e gato → s. A máscara garante que essas previsões possam ser calculadas juntas, usando em cada coluna só os bytes já vistos até ali.

batch_from sorteia a posição inicial de cada janela. O último início possível é len(data) − context − 1. A chamada a torch.randint usa limite superior exclusivo, então nenhuma janela passa do fim do tensor.

Para calcular a perda, o código coloca os logits no formato [B×T, 256] e os alvos no formato [B×T]. A entropia cruzada calcula a média de −log p(alvo). Uma previsão uniforme daria a cada byte probabilidade 1/256 e produziria uma perda de cerca de 5,55.

train.py separa os primeiros 90% dos bytes para treino e os demais para validação, antes de recortar as janelas. Em cada passo, o script limpa os gradientes, calcula a perda, chama backward e limita a norma global do gradiente a 1. O AdamW atualiza os parâmetros a partir desses gradientes. A validação reaproveita as mesmas janelas a cada medição e usa no_grad. O checkpoint salva configuração e pesos para reconstruir o modelo.

## Gerar texto com os pesos salvos

generate.py codifica o prefixo, carrega o checkpoint e chama GPT.generate. A cada iteração, o modelo recebe somente a janela mais recente, usa os logits da última posição, divide-os pela temperatura e pode reter apenas os k maiores. Depois do softmax, sorteia um byte e o acrescenta ao prefixo. Temperaturas mais baixas concentram a massa nos escores maiores. Com top-k igual a 1, a seleção equivale à escolha do maior escore.

O código recalcula a janela a cada passo e reinicia os índices de posição dentro dela. A sequência completa é preservada na saída. Como o sorteio ocorre por byte, a decodificação substitui combinações UTF-8 inválidas.

## O que aconteceu na execução

Os seis testes passaram. Eles verificam o formato da saída, o cálculo da perda, os gradientes, o compartilhamento dos pesos, a geração com janela móvel, os alvos deslocados e a rejeição de dimensões inválidas.

O teste troca dois IDs no fim da sequência e compara os logits das três primeiras posições. A comparação atravessa o modelo inteiro, da primeira projeção à cabeça de linguagem. Uma máscara invertida ou aplicada no eixo errado tende a aparecer como diferença nesses logits.

Em CPU, o treino rodou por 100 passos com PyTorch 2.15.0 e semente 42. A tabela mostra a perda do lote de treino de cada passo e a média de quatro lotes fixos de validação.

| Passo | Treino | Validação |
| --- | --- | --- |
| 1 | 5,5565 | 5,4446 |
| 25 | 4,5193 | 4,4918 |
| 50 | 3,8675 | 3,8044 |
| 75 | 3,3512 | 3,3205 |
| 100 | 3,0646 | 3,0175 |

O script de geração abriu o arquivo salvo, reconstruiu o modelo com a configuração armazenada e completou o prefixo. O início da saída foi “Na bibliotecalvpe h.toiAul.”. O checkpoint abriu e o programa gerou uma continuação. O texto ainda saiu quase sem sentido.

## O que esses resultados permitem concluir

Os testes mostram que o modelo usa apenas o prefixo para prever o próximo byte. A queda da perda mostra que ele aprendeu padrões do corpus. Como as mesmas quatro frases aparecem nos dois trechos do arquivo, seria preciso reservar documentos diferentes para avaliar se o modelo generaliza.

Para cada sequência, a atenção guarda T × T escores por cabeça. Se a janela passar de 64 para 128, essa matriz passa de 4.096 para 16.384 posições por cabeça. A implementação deixa essa matriz explícita, o que facilita seguir o cálculo. Na geração, ela é refeita para a janela atual a cada token.

O projeto reproduz os componentes centrais da arquitetura em escala pequena. O GPT-2 da aula usa BPE e uma escala de treino maior. Aqui, os bytes individuais e o corpus curto tornam mais fácil inspecionar os cálculos e repetir a execução.

## Referências

Andrej Karpathy. Let's reproduce GPT-2 (124M). https://youtu.be/l8pRSuU81PU

Código da aula. https://github.com/karpathy/build-nanogpt

OpenAI. Implementação oficial do GPT-2. https://github.com/openai/gpt-2
