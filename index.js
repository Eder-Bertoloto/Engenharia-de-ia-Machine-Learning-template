//REDE NEURAL PARA CLASSIFICAR PESSOAS EM CATEGORIAS (premium, medium, basic)

import tf from '@tensorflow/tfjs';

async function trainModel(inputXs, inputYs) {
    // Criamos um modelo de fluxo sequencial
    const model = tf.sequential();

    // primeira camada da rede
    //entrada de 7 posicoes (idade_normalizada + 3 cor + 3 localização)

    //80 neurônios = aqui coloquei tudo isso, pq tem pouca base de treino
    //quanto mais neuronios, mais complexidade a rede pode aprender
    //e consequentemente, mais processamento ela vai usar

    //a ReLu age como filtro: e como se ela deixasse somente os dados interessantes seguirem
    //viagem na rede
    //se a informacao chegou nesse neuronio, ela 'e positiva, passa para frente!
    //se for 0 ou negativa, pode jogar fora, nao vai servir para nada
    model.add(tf.layers.dense({
        inputShape: [7],//7 posicoes de entrada
        activation: 'relu',//
        units: 80//80 neuronios
    }));

    //saida: 3 neuronios, pq temos 3 categorias de saida (premium, medium, basic)
    //activation: softmax, pq queremos que a saida seja uma probabilidade, e nao um valor qualquer
    model.add(tf.layers.dense({
        activation: 'softmax',//softmax: pega a saida e transforma em probabilidade
        units: 3//3 neuronios de saida
    }));

    //compilando o modelo
    //optimizer Adam (adaptive moment estimation) 
    //e' um treinador pessoal moderno para redes neurais:
    //ajusta os pesos de forma inteligente e eficiente
    //aprende com o historico de erros e acertos

    //loss: categoricalCrossentropy (entropia cruzada categórica)
    //ele compara o que o modelo "acha" (os scores de cada categoria) com a resposta certa
    //ex: a categoria premium sera' sempre [1, 0, 0], medium [0, 1, 0], basic [0, 0, 1]

    //quanto mais distante da previsao do moldelo da rasposta correta, maior o erro
    //ex classico: classificacao de imagens, recomendacao, categorizacao de usuario, etc
    //qualquer coisa em que a resposta certa e' "apenas uma entre v'arias possiveis"

    //metrics: accuracy (acurácia)
    //mede a porcentagem de acertos do modelo, ou seja, quantas vezes ele acertou a resposta correta

    model.compile({
        optimizer: 'adam',//adam: algoritmo de otimização, que vai ajustar os pesos da rede
        loss: 'categoricalCrossentropy',//categoricalCrossentropy: função de perda, que mede o quão ruim a rede está se saindo 
        metrics: ['accuracy']//metrics: métrica de avaliação, que mede a acurácia da rede
    });

    //treinando o modelo
    //verbose: 0 (não mostra o progresso do treinamento)
    //epochs: 1000 (quantas vezes o modelo vai ver os dados de treino)
    //shuffle: true (embaralha os dados a cada época, para evitar overfitting)
    // (overfitting: nao ficar viciado em um padrão, e sim aprender a generalizar)
    await model.fit(inputXs, inputYs, {
        verbose: 0,
        epochs: 1000,
        shuffle: true,
        callbacks: {
            //onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`)
        }
    });

    return model;
}    

async function predict(model, pessoaTensorNormalizado) {
    //transformar o array js para o tensor(tfjs)
    const tfInput = tf.tensor2d(pessoaTensorNormalizado);

    //fazendo a previsao(output sera' um vetor de 3 probavilidades)
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// inputXs.print();
// outputYs.print();


//quanto mais dados melhor!
//assim o algoritmo consegue entender melhor os padroes complexos
//os dados
const model = await trainModel(inputXs, outputYs);

const pessoa = {
    nome: "João",
    idade: 28,
    cor: "verde",
    localizacao: "Curitiba"
}

//normalizandoa idade da nova pessoa usando o mesmo padrao de treino
//valor_atual - valor_minimo / (valor_maximo - valor_minimo)
//Exemplo: idade_min = 25, idade_max = 40, entao (28 -25) / (40 - 25) = 0.2
const pessoaTensorNormalizado = [
    [
        0.2,//idade normalizada
        1, //cor azul
        0, //cor vermelho
        0, //cor verde
        0, //localizacao São Paulo
        1, //localizacao Rio
        0 //localizacao Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
        .sort((a, b) => b.prob - a.prob)//ordenar do maior para o menor
        .map(p => `${labelsNomes[p.index]}  (${(p.prob * 100).toFixed(2)}%)`)//formatar a saida
        .join('\n');//juntar tudo em uma string, separando por quebra de linha

console.log(results);
