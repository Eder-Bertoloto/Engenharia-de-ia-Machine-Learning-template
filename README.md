# Classificação de pessoas com TensorFlow.js

Projeto didático em Node.js que treina uma pequena rede neural para classificar pessoas em três categorias: `premium`, `medium` e `basic`. O código está concentrado em [`index.js`](./index.js).

## Requisitos

- Node.js em uma versão recente, com suporte ao modo `--watch`.
- npm.

## Instalação e execução

Na raiz do projeto, instale as dependências e inicie o exemplo:

```bash
npm install
npm start
```

O comando `npm start` executa `index.js` com o modo de observação do Node.js. O programa treina o modelo em cada execução e imprime as probabilidades previstas para a pessoa de exemplo.

## Como funciona

1. Os dados de treino contêm três exemplos, representados por tensores:
   - Entradas: idade normalizada e codificações one-hot de cor e localização.
   - Saídas: rótulos one-hot na ordem `premium`, `medium`, `basic`.
2. `trainModel` cria uma rede sequencial com uma camada densa de 80 neurônios (`relu`) e uma camada de saída de 3 neurônios (`softmax`).
3. O modelo é treinado com Adam e `categoricalCrossentropy` por 1.000 épocas.
4. `predict` transforma a entrada em tensor e retorna as probabilidades das três classes, que são exibidas da maior para a menor.

### Formato das entradas

Cada pessoa é representada por sete valores, nesta ordem:

```text
[idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
```

As categorias de cor e localização devem ser codificadas em one-hot: o valor correspondente à opção escolhida é `1`, e os demais são `0`. A idade usada no exemplo é normalizada para o intervalo de 0 a 1. A entrada deve seguir a mesma ordem e o mesmo padrão usados no treino.

## Arquivos e comandos

- `index.js`: dados de exemplo, definição e treino do modelo, e previsão.
- `package.json`: metadados, dependência TensorFlow.js e scripts npm.
- `package-lock.json`: versões resolvidas das dependências.
- `npm start`: executa o exemplo com reinicialização automática ao detectar alterações.
- `npm test`: ainda não está implementado; o script atual termina com erro intencional.

## Limitações conhecidas

- O conjunto de treino tem somente três exemplos inventados. As previsões não são confiáveis para uso real e o modelo pode simplesmente memorizar esses dados.
- O modelo é treinado novamente a cada execução; não há salvamento nem carregamento de pesos.
- A pessoa declarada no exemplo é João, de 28 anos, com cor verde e localização Curitiba. Porém, o vetor fornecido à previsão codifica azul e Rio. Para que a previsão corresponda à pessoa declarada, atualize os valores one-hot do vetor em `index.js`.
- Não há validação de entrada, avaliação com dados independentes ou interface/API.

Este código serve como ponto de partida para estudo, não como sistema de classificação pronto para produção.
