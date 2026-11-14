# Detecção de Movimento em Videomonitoramento — CDnet 2014 *pedestrians*

Trabalho de Processamento de Imagens com OpenCV.

O sistema separa as regiões em movimento do fundo da cena, desenha caixas delimitadoras sobre os pedestres e gera uma série temporal da atividade. São comparadas duas abordagens: diferença entre frames consecutivos e um modelo temporal de fundo com atualização progressiva, implementado pelo grupo.

## Integrantes

- Lucas Carreño
- Melissa Pizzato

## Requisitos

- Python 3.12.2
- Bibliotecas em `requirements.txt` (OpenCV, NumPy, pandas e Matplotlib)

## Instalação

Na raiz do projeto:

```bash
pip install -r requirements.txt
```

## Base de dados

A sequência **pedestrians** já está incluída no repositório, na pasta `data/pedestrians/`, com a seguinte estrutura:

```
data/pedestrians/
├── input/
├── groundtruth/
├── ROI.bmp
└── temporalROI.txt
```

Fonte original: https://changedetection.vitalab.usherbrooke.ca/dataset2014/ (categoria Baseline).

## Estrutura do projeto

```
├── data/pedestrians/        base de dados
├── src/
│   ├── config.py            caminhos, divisão de frames e parâmetros
│   ├── dataset.py           leitura de frames, ground truth e ROI
│   ├── preprocess.py        tons de cinza e filtro Gaussiano
│   ├── frame_diff.py        diferença entre frames
│   ├── background_model.py  modelo temporal de fundo
│   ├── postprocess.py       morfologia
│   ├── regions.py           componentes conexos e caixas
│   ├── evaluation.py        métricas
│   └── visualization.py     anotação dos frames
├── check_data.py
├── run_pipeline.py
├── tune.py
├── evaluate.py
├── outputs/
└── requirements.txt
```

## Execução

Todos os comandos devem ser executados a partir da raiz do projeto.

### 1. Verificar a base

```bash
python check_data.py
```

### 2. Executar um método

```bash
python run_pipeline.py --metodo fundo --intervalo ajuste --mostrar
```

- `--metodo`: `diff` (diferença de frames) ou `fundo` (modelo de fundo)
- `--intervalo`: `ajuste` (frames 300–599), `teste` (600–1099) ou `completo` (300–1099)
- `--mostrar`: exibe o vídeo anotado durante a execução
- `--salvar`: salva máscaras, vídeo e série temporal em `outputs/`

### 3. Ajustar parâmetros

```bash
python tune.py
```

Testa combinações de parâmetros somente nos frames de ajuste (300–599). Os valores escolhidos ficam registrados em `src/config.py`.

### 4. Avaliação final

```bash
python evaluate.py
```

Executa os dois métodos uma única vez nos frames de teste (600–1099) e salva as métricas, vídeos e gráficos em `outputs/`.
