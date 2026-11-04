# GPT-2 em PyTorch

## 1. O que o modelo faz

O modelo aprende uma tarefa simples: prever o próximo byte de um trecho de texto. A arquitetura segue o GPT-2, com dois blocos e um corpus pequeno para rodar em CPU.

### Configuração e tokenização

GPTConfig reúne o vocabulário V, a janela máxima, o número de blocos, o número de cabeças H e a largura C. O modelo padrão usa V = 256, janela de 64, dois blocos, H = 4 e C = 64. Cada cabeça recebe d = C/H = 16 características. A divisão exige que C seja múltiplo de H.

Os scripts representam cada byte UTF-8 por um inteiro de 0 a 255. Por exemplo, "gato" vira [103, 97, 116, 111], enquanto "é" ocupa dois bytes, [195, 169]. Essa escolha dispensa um tokenizador externo e muda a unidade de previsão em relação ao GPT-2 original, que usa BPE sobre bytes.

### Entradas, embeddings e posições

A entrada de GPT.forward é uma matriz [B, T] de IDs inteiros, com B sequências e T tokens por sequência. A tabela tokens transforma cada ID em um vetor de C valores. A tabela positions representa os índices de 0 a T-1 e produz [T, C]. O PyTorch soma os mesmos vetores de posição a cada sequência do lote. O resultado tem forma [B, T, C].

O embedding dá ao modelo uma representação treinável de cada símbolo. A informação de posição permite distinguir ocorrências do mesmo símbolo em locais diferentes e fornece ao modelo uma noção da ordem. Depois da soma, os blocos transformam os vetores mantendo sua forma.

### Percurso de uma entrada

IDs [B, T] → embeddings [B, T, C] → blocos [B, T, C] → normalização final → logits [B, T, V]. Cada vetor de V logits expressa as preferências do modelo pelo próximo byte naquela posição.

## 2. Como a atenção funciona

### Consultas, chaves e valores

CausalSelfAttention projeta [B, T, C] em [B, T, 3C] e divide a última dimensão em Q, K e V. Após reorganizar os eixos, cada tensor tem forma [B, H, T, d]. A consulta Q representa a informação que uma posição procura. A chave K determina sua compatibilidade com a consulta e o valor V carrega a informação que será combinada.

O produto QKᵀ gera [B, H, T, T]. A linha i contém as compatibilidades da posição i com as posições de origem. Dividir por √d reduz a escala dos produtos à medida que a dimensão cresce, o que ajuda a evitar a saturação do softmax.

### Máscara e múltiplas cabeças

A máscara triangular inferior permite que a posição i consulte as posições de 0 até i. Antes do softmax, os escores futuros recebem −∞ e passam a ter peso zero. Com três posições, os conjuntos visíveis são {0}, {0, 1} e {0, 1, 2}. Por exemplo, ao ler o “t” em “gat”, o modelo pode usar o próprio “t” para prever “o”. O “o” só entra como alvo nesse passo.

O softmax normaliza cada linha e a multiplicação pelos valores produz uma média ponderada por cabeça. As cabeças aprendem projeções diferentes, o que permite usar mais de um padrão de combinação entre posições. Seus resultados são concatenados e uma projeção linear retorna [B, T, C].

### LayerNorm, conexões residuais e MLP

O bloco aplica LayerNorm antes de cada ramo. A normalização usa a média e a variância das C características de cada posição, seguidas de escala e deslocamento aprendidos. Isso ajusta a escala dos valores de cada token separadamente.

O código calcula x + atenção(norm(x)) e, com o resultado atualizado, x + mlp(norm(x)). As somas preservam um caminho direto para a representação e para os gradientes. O MLP opera em cada posição de modo independente. Ele expande C para 4C, aplica GELU e projeta de volta para C. A GELU introduz não linearidade, permitindo combinar características além do que duas projeções lineares poderiam representar juntas.

## 3. Treino e geração

### Cabeça de linguagem e função de perda

A LayerNorm final prepara as representações para a cabeça linear, que retorna um escore para cada ID do vocabulário. A cabeça usa a mesma matriz do embedding de tokens. Assim, a linha de um ID serve tanto para representá-lo na entrada quanto para pontuá-lo na saída. O compartilhamento também reduz o número de parâmetros.

batch_from retira janelas de T+1 bytes. Para "gato", com T = 3, a entrada é [g, a, t] e o alvo é [a, t, o]. A posição que lê "t" recebe como alvo "o". GPT.forward reorganiza os logits em [B×T, V] e os alvos em [B×T]. A entropia cruzada calcula a média de −log p(alvo) usando os logits diretamente. Quanto maior a probabilidade atribuída ao alvo correto, menor a perda.

### Otimização e validação

train.py separa os primeiros 90% dos bytes para treino e os demais para validação, antes de recortar as janelas. Em cada passo, o script limpa os gradientes, calcula a perda, chama backward e limita a norma global do gradiente a 1. O AdamW atualiza os parâmetros a partir desses gradientes. A validação reaproveita as mesmas janelas a cada medição e usa no_grad. O checkpoint salva configuração e pesos para reconstruir o modelo.

O corpus incluído é sintético e repetitivo. Embora as janelas de validação estejam em outra região do arquivo, frases iguais aparecem em ambas as divisões. A perda nesse corpus ajuda a verificar o pipeline. Para estudar generalização, seria necessário avaliar em documentos distintos.

### Geração autoregressiva

generate.py codifica o prefixo, carrega o checkpoint e chama GPT.generate. A cada iteração, o modelo recebe somente a janela mais recente, usa os logits da última posição, divide-os pela temperatura e pode reter apenas os k maiores. Depois do softmax, sorteia um byte e o acrescenta ao prefixo. Temperaturas mais baixas concentram a massa nos escores maiores. Com top-k igual a 1, a seleção equivale à escolha do maior escore.

O código recalcula a janela a cada passo e reinicia os índices de posição dentro dela. A sequência completa é preservada na saída. Como o sorteio ocorre por byte, a decodificação substitui combinações UTF-8 inválidas.

## 4. O que foi testado

### Como verificar o comportamento

test_model.py verifica a forma dos logits e a perda contra o cálculo direto pela log-probabilidade do alvo. O teste de causalidade altera o sufixo e exige que os logits do prefixo permaneçam iguais. O teste de backward verifica gradientes finitos e confirma que embedding e cabeça compartilham o mesmo parâmetro.

Os demais testes conferem a geração com janela móvel e top-k igual a 1, o deslocamento de um token entre entrada e alvo e a rejeição de dimensões inválidas. Os seis testes passaram. Em 100 passos na CPU, a perda de validação caiu de 5,4446 para 3,0175. Após recarregar os pesos, a geração começou com “Na bibliotecalvpe h.toiAul.”. O programa conseguiu gerar bytes, mas a amostra ainda é pouco legível.

### Conceitos teóricos no código

• Produto escalar e softmax: em CausalSelfAttention.forward, a similaridade é convertida em pesos sobre os valores.

• Fatoração autoregressiva: cada fator da probabilidade da sequência condiciona o próximo token ao prefixo. A máscara e os alvos deslocados implementam essa condição mesmo quando todas as posições são calculadas em paralelo no treino.

• Retropropagação: loss.backward propaga a derivada da entropia cruzada pela cabeça, pelos blocos e pelos embeddings.

• Capacidade e custo: a configuração reduzida facilita testar em CPU. A matriz de atenção cresce quadraticamente com T, e o exemplo usa pesos inicializados localmente e um corpus demonstrativo.

### Como interpretar o resultado

O teste do prefixo é a checagem mais direta da máscara. Mesmo com um final diferente, as primeiras previsões continuam iguais. Já a queda da perda mostra que o modelo ajustou seus pesos às frases do corpus. Como as mesmas frases aparecem no treino e na validação, a próxima checagem seria separar textos inteiros e avaliar em frases inéditas.

## 5. Um exemplo de máscara

### O que acontece com três posições

Considere a entrada “gat”. Para a consulta da posição 1, suponha que os escores já escalados sejam [0, 1, 2]. São valores ilustrativos. Antes da máscara, a terceira chave teria o maior escore. Como ela corresponde ao “t”, que ainda está no futuro dessa consulta, o código transforma a linha em [0, 1, −∞].

O softmax resulta em aproximadamente [0,27; 0,73; 0]. Se os dois primeiros valores fossem vetores [2, 0] e [0, 2], a saída da cabeça nessa posição seria aproximadamente [0,54; 1,46]. Mudar o terceiro valor deixaria essa conta igual.

### Onde conferir no código

Em model.py, scores guarda uma linha para cada consulta. A transposição de k troca os eixos de tempo e profundidade para que a multiplicação compare cada consulta com todas as chaves. A máscara registrada no módulo acompanha o dispositivo do modelo. Ela é recortada para T × T, pois a entrada pode ser menor que a janela máxima.

### O que o teste mede

O teste troca dois IDs no fim da sequência e compara os logits das três primeiras posições. A comparação atravessa o modelo inteiro, da primeira projeção à cabeça de linguagem. Uma máscara invertida ou aplicada no eixo errado tende a aparecer como diferença nesses logits.

## 6. O formato dos tensores

### Um lote de duas sequências

Imagine B = 2 e T = 5 na configuração padrão. Os IDs têm forma [2, 5]. Depois da soma dos embeddings, há cinco vetores de 64 números em cada sequência, ou seja, [2, 5, 64]. A projeção de atenção produz [2, 5, 192], com 64 características para cada um de Q, K e V.

Separando as quatro cabeças, cada Q, K e V passa a ter forma [2, 4, 5, 16]. A atenção tem [2, 4, 5, 5], porque cada uma das cinco consultas tem cinco chaves possíveis. O produto com V restaura a profundidade 16 de cada cabeça. Reunir as quatro cabeças devolve os 64 números por posição.

### Onde o MLP entra

No MLP, cada vetor de 64 características passa por 256 unidades intermediárias. A GELU age nesses valores e a segunda projeção volta a 64, de modo que a soma residual tenha dimensões compatíveis. A atenção combina posições. O MLP transforma as características de cada posição após essa combinação.

### Na saída do modelo

A cabeça retorna [2, 5, 256]. São dez previsões, cada uma com um escore para cada byte possível. No treino, a perda compara essas dez previsões com dez IDs alvo. Na geração, o código usa apenas o vetor da última posição para decidir o que acrescentar.

## 7. Como montar os alvos

### Da janela ao lote

Imagine uma janela de cinco bytes que contém “gatos”. Com contexto quatro, x recebe “gato” e y recebe “atos”. As quatro previsões são g → a, ga → t, gat → o e gato → s. A máscara garante que essas previsões possam ser calculadas juntas, usando em cada coluna só os bytes já vistos até ali.

batch_from sorteia a posição inicial de cada janela. O último início possível é len(data) − context − 1. A chamada a torch.randint usa limite superior exclusivo, então nenhuma janela passa do fim do tensor.

### O que significa uma perda de 5,55

Se todos os 256 bytes fossem igualmente prováveis, a chance do byte correto seria 1/256 e a perda seria ln(256), cerca de 5,55. A primeira perda de treino ficou perto desse valor. A perda final menor indica que, em média, o modelo passou a atribuir mais probabilidade aos alvos avaliados.

### Por que separar o corpus antes de fazer janelas

Se a divisão fosse feita depois de sortear janelas, uma janela de treino e outra de validação poderiam compartilhar quase todos os bytes. O script separa primeiro os trechos. Neste corpus, as mesmas frases reaparecem nos dois trechos por construção, então a avaliação continua sendo uma demonstração do pipeline.

## 8. Uma execução em CPU

### Como reproduzir

Instale as dependências conforme o README. Execute python test_model.py -v. Para repetir a rodada, use python train.py --steps 100 --output gpt2.pt. Depois execute python generate.py gpt2.pt "Na biblioteca" --tokens 80 --top-k 20. A execução verificada usou PyTorch 2.15.0 em CPU e a semente padrão 42.

### Perda registrada

Passo 1    | treino 5,5565 | validação 5,4446

Passo 25  | treino 4,5193 | validação 4,4918

Passo 50  | treino 3,8675 | validação 3,8044

Passo 75  | treino 3,3512 | validação 3,3205

Passo 100 | treino 3,0646 | validação 3,0175

### Como ler esses números

A perda de treino mostrada é a do lote daquele passo. A de validação é a média de quatro lotes fixos. Comparar a coluna de validação ao longo da execução usa as mesmas janelas e torna mais fácil ver se os pesos estão atribuindo mais probabilidade aos bytes esperados.

### O checkpoint funciona?

O script de geração abriu o arquivo salvo, reconstruiu o modelo com a configuração armazenada e completou o prefixo. O início da saída foi “Na bibliotecalvpe h.toiAul.”. O checkpoint abriu e o programa gerou uma continuação. O texto ainda saiu quase sem sentido.

## 9. Limites do experimento

### O que veio da arquitetura GPT-2

A implementação usa blocos decoder, máscara causal, posições aprendidas, LayerNorm antes dos ramos, MLP com GELU e compartilhamento entre embedding e cabeça de saída. O exercício usa dois blocos pequenos, um corpus curto e bytes individuais. O vídeo de referência apresenta a reprodução da variante de 124 milhões de parâmetros, com outra escala de dados e treino.

### O custo de olhar para o prefixo

Para cada sequência, a atenção guarda T × T escores por cabeça. Se a janela passar de 64 para 128, essa matriz passa de 4.096 para 16.384 posições por cabeça. A implementação deixa essa matriz explícita, o que facilita seguir o cálculo. Na geração, ela é refeita para a janela atual a cada token.

### Qual seria o próximo teste

Para medir generalização, valeria usar um corpus com documentos distintos e reservar alguns documentos inteiros para validação. Para entender a amostragem, vale manter o mesmo checkpoint e prefixo e comparar top-k igual a 1 com top-k igual a 20. O primeiro sempre escolhe o maior escore de cada passo. O segundo pode escolher outros candidatos e produzir continuações diferentes.

### Conclusão

O resultado mais claro é o teste de causalidade. Ele mostra que trocar o fim da entrada deixa as previsões do começo iguais. Junto dos testes de perda e gradientes, isso verifica a regra que sustenta o treino: cada posição aprende a prever o byte seguinte com a informação disponível até ali.

### Referências

Andrej Karpathy. Let's reproduce GPT-2 (124M). https://youtu.be/l8pRSuU81PU

Código da aula. https://github.com/karpathy/build-nanogpt

OpenAI. Implementação oficial do GPT-2. https://github.com/openai/gpt-2
