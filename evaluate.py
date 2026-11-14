import matplotlib
matplotlib.use("Agg")  # gera os gráficos direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PASTA_SAIDA, FRAMES_TESTE
from run_pipeline import executar, PARAMS_PADRAO

METODOS = ["diff", "fundo"]
NOMES = {"diff": "Diferença de frames", "fundo": "Modelo de fundo"}


def f_por_frame(serie):
    # Calcula o F-measure de cada frame individualmente,
    # para ver em quais momentos do vídeo cada método acerta ou falha
    p = serie["tp"] / (serie["tp"] + serie["fp"])
    r = serie["tp"] / (serie["tp"] + serie["fn"])
    return (2 * p * r / (p + r)).fillna(0)


def main():
    pasta_fig = PASTA_SAIDA / "figures"
    pasta_fig.mkdir(parents=True, exist_ok=True)

    resultados = []
    series = {}

    # Roda cada método uma única vez no trecho de teste, com os parâmetros congelados
    # Também salva máscaras, vídeo anotado e série temporal em outputs/
    for metodo in METODOS:
        print(f"Avaliando {NOMES[metodo]}...")
        serie, resultado = executar(metodo, PARAMS_PADRAO[metodo], FRAMES_TESTE, salvar=True)
        series[metodo] = serie
        resultados.append({"metodo": NOMES[metodo], **resultado})

    # Tabela comparativa final, usada no relatório
    tabela = pd.DataFrame(resultados).set_index("metodo")
    tabela.to_csv(PASTA_SAIDA / "csv" / "comparacao_teste.csv")
    print("\nResultados no teste (frames 600–1099):")
    print(tabela.T.round(4).to_string())

    # Gráfico 1: série temporal da atividade (quantidade de regiões e área em movimento)
    fig, eixos = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for metodo, serie in series.items():
        eixos[0].plot(serie["frame"], serie["n_regioes"], label=NOMES[metodo], linewidth=0.8)
        eixos[1].plot(serie["frame"], serie["area_pct"], label=NOMES[metodo], linewidth=0.8)
    eixos[0].set_ylabel("Nº de regiões")
    eixos[1].set_ylabel("Área em movimento (%)")
    eixos[1].set_xlabel("Frame")
    eixos[0].legend()
    fig.tight_layout()
    fig.savefig(pasta_fig / "serie_temporal.png", dpi=150)

    # Gráfico 2: F-measure frame a frame no trecho de teste
    fig, eixo = plt.subplots(figsize=(10, 4))
    for metodo, serie in series.items():
        avaliados = serie.dropna(subset=["tp"])
        eixo.plot(avaliados["frame"], f_por_frame(avaliados), label=NOMES[metodo], linewidth=0.8)
    eixo.set_xlabel("Frame")
    eixo.set_ylabel("F-measure")
    eixo.set_ylim(0, 1)
    eixo.legend()
    fig.tight_layout()
    fig.savefig(pasta_fig / "f_por_frame.png", dpi=150)

    print(f"\nGráficos salvos em {pasta_fig}")


if __name__ == "__main__":
    main()