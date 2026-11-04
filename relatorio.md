## Arquitetura GPT-2 em PyTorch

O modelo aprende uma tarefa simples de enunciar: dado um pedaço de texto, qual byte vem depois? A arquitetura segue o GPT-2, com dois blocos e um corpus pequeno para rodar em CPU.

### Configuração e tokenização

GPTConfig reúne o vocabulário V, a janela máxima, o número de blocos, o número de cabeças H e a largura C. O modelo padrão usa V = 256, janela de 64, dois blocos, H = 4 e C = 64. Cada cabeça recebe d = C/H = 16 características. A divisão exige que C seja múltiplo de H.

Os scripts representam cada byte UTF-8 por um inteiro de 0 a 255. Por exemplo, "gato" vira [103, 97, 116, 111], enquanto "é" ocupa dois bytes, [195, 169]. Essa escolha dispensa um tokenizador externo e muda a unidade de previsão em relação ao GPT-2 original, que usa BPE sobre bytes.

### Entradas, embeddings e posições

A entrada de GPT.forward é uma matriz [B, T] de IDs inteiros, com B sequências e T tokens por sequência. A tabela tokens transforma cada ID em um vetor de C valores. A tabela positions representa os índices de 0 a T-1 e produz [T, C]. A soma com broadcasting aplica as mesmas posições a todo o lote e retorna [B, T, C].

O embedding dá ao modelo uma representação treinável de cada símbolo. A informação de posição permite distinguir ocorrências do mesmo símbolo em locais diferentes e fornece ao modelo uma noção da ordem. Depois da soma, os blocos transformam os vetores mantendo sua forma.

### Percurso de uma entrada

IDs [B, T] → embeddings [B, T, C] → blocos [B, T, C] → normalização final → logits [B, T, V]. Cada vetor de V logits expressa as preferências do modelo pelo próximo byte naquela posição.

## Atenção causal e transformação

### Consultas, chaves e valores

CausalSelfAttention projeta [B, T, C] em [B, T, 3C] e divide a última dimensão em Q, K e V. Após reorganizar os eixos, cada tensor tem forma [B, H, T, d]. A consulta Q representa a informação que uma posição procura. A chave K determina sua compatibilidade com a consulta e o valor V carrega a informação que será combinada.

O produto QKᵀ gera [B, H, T, T]. A linha i contém as compatibilidades da posição i com as posições de origem. Dividir por √d reduz a escala dos produtos à medida que a dimensão cresce, o que ajuda a evitar a saturação do softmax.

### Máscara e múltiplas cabeças

A máscara triangular inferior permite que a posição i consulte as posições de 0 até i. Antes do softmax, os escores futuros recebem −∞ e passam a ter peso zero. Com três posições, os conjuntos visíveis são {0}, {0, 1} e {0, 1, 2}. Por exemplo, ao ler o “t” em “gat”, o modelo pode usar o próprio “t” para prever “o”. O “o” só entra como alvo nesse passo.

O softmax normaliza cada linha e a multiplicação pelos valores produz uma média ponderada por cabeça. As cabeças aprendem projeções diferentes, o que permite usar mais de um padrão de combinação entre posições. Seus resultados são concatenados e uma projeção linear retorna [B, T, C].

### LayerNorm, conexões residuais e MLP

O bloco aplica LayerNorm antes de cada ramo. A normalização usa a média e a variância das C características de cada posição, seguidas de escala e deslocamento aprendidos. Ela estabiliza a escala da entrada de cada transformação sem misturar tokens.

O código calcula x + atenção(norm(x)) e, com o resultado atualizado, x + mlp(norm(x)). As somas preservam um caminho direto para a representação e para os gradientes. O MLP opera em cada posição de modo independente. Ele expande C para 4C, aplica GELU e projeta de volta para C. A GELU introduz não linearidade, permitindo combinar características além do que duas projeções lineares poderiam representar juntas.

## De logits a um novo token

### Cabeça de linguagem e função de perda

A LayerNorm final prepara as representações para a cabeça linear, que retorna um escore para cada ID do vocabulário. A cabeça usa a mesma matriz do embedding de tokens. Assim, a linha de um ID serve tanto para representá-lo na entrada quanto para pontuá-lo na saída. O compartilhamento também reduz o número de parâmetros.

batch_from retira janelas de T+1 bytes. Para "gato", com T = 3, a entrada é [g, a, t] e o alvo é [a, t, o]. A posição que lê "t" recebe como alvo "o". GPT.forward reorganiza os logits em [B×T, V] e os alvos em [B×T]. A entropia cruzada calcula a média de −log p(alvo) usando os logits diretamente. Quanto maior a probabilidade atribuída ao alvo correto, menor a perda.

### Otimização e validação

train.py separa os primeiros 90% dos bytes para treino e os demais para validação, antes de recortar as janelas. Em cada passo, o script limpa os gradientes, calcula a perda, chama backward e limita a norma global do gradiente a 1. O AdamW atualiza os parâmetros a partir desses gradientes. A validação reaproveita as mesmas janelas a cada medição e usa no_grad. O checkpoint salva configuração e pesos para reconstruir o modelo.

O corpus incluído é sintético e repetitivo. Embora as janelas de validação estejam em outra região do arquivo, frases iguais aparecem em ambas as divisões. A perda nesse corpus ajuda a verificar o pipeline. Para estudar generalização, seria necessário avaliar em documentos distintos.

### Geração autoregressiva

generate.py codifica o prefixo, carrega o checkpoint e chama GPT.generate. A cada iteração, o modelo recebe somente a janela mais recente, usa os logits da última posição, divide-os pela temperatura e pode reter apenas os k maiores. Depois do softmax, sorteia um byte e o acrescenta ao prefixo. Temperaturas mais baixas concentram a massa nos escores maiores. Com top-k igual a 1, a seleção equivale à escolha do maior escore.

O código recalcula a janela a cada passo e reinicia os índices de posição dentro dela. A sequência completa é preservada na saída. Como o sorteio ocorre por byte, a decodificação substitui combinações UTF-8 inválidas.

## O que os testes mostram

### Como verificar o comportamento

test_model.py verifica a forma dos logits e a perda contra o cálculo direto pela log-probabilidade do alvo. O teste de causalidade altera o sufixo e exige que os logits do prefixo permaneçam iguais. O teste de backward verifica gradientes finitos e confirma que embedding e cabeça compartilham o mesmo parâmetro.

Os demais testes conferem a geração com janela móvel e top-k igual a 1, o deslocamento de um token entre entrada e alvo e a rejeição de dimensões inválidas. Os seis testes passaram. Em 100 passos na CPU, a perda de validação caiu de 5,4446 para 3,0175. Após recarregar os pesos, a geração começou com “Na bibliotecalvpe h.toiAul.”. O programa conseguiu gerar bytes, mas a amostra ainda é pouco legível.

### Conceitos teóricos no código

• Produto escalar e softmax: em CausalSelfAttention.forward, a similaridade é convertida em pesos sobre os valores.

• Fatoração autoregressiva: cada fator da probabilidade da sequência condiciona o próximo token ao prefixo. A máscara e os alvos deslocados implementam essa condição mesmo quando todas as posições são calculadas em paralelo no treino.

• Retropropagação: loss.backward propaga a derivada da entropia cruzada pela cabeça, pelos blocos e pelos embeddings.

• Capacidade e custo: a configuração reduzida facilita testar em CPU. A matriz de atenção cresce quadraticamente com T, e o exemplo usa pesos inicializados localmente e um corpus demonstrativo.

### O que dá para concluir

O teste do prefixo é a checagem mais direta da máscara. Mesmo com um final diferente, as primeiras previsões continuam iguais. Já a queda da perda mostra que o modelo ajustou seus pesos às frases do corpus. Como as mesmas frases aparecem no treino e na validação, a próxima checagem seria separar textos inteiros e avaliar em frases inéditas.

### Referências

Andrej Karpathy. Let's reproduce GPT-2 (124M). https://youtu.be/l8pRSuU81PU

Código da aula. https://github.com/karpathy/build-nanogpt

OpenAI. Implementação oficial do GPT-2. https://github.com/openai/gpt-2
