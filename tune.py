import itertools
import time

import pandas as pd

from src.config import PASTA_SAIDA, FRAMES_AJUSTE
from run_pipeline import executar, PARAMS_PADRAO

# Valores testados para cada método

# Valores iniciais para cada parâmetro na primeira rodada de tuning, foram ajustados posteriormente
# GRADES = {
#     "diff": {
#         "limiar": [8, 12, 16, 20, 25],
#         "kernel": [3, 5, 7],
#         "area_min": [100, 300],
#     },
#     "fundo": {
#         "k": [2.0, 2.5, 3.0, 3.5],
#         "alpha": [0.005, 0.01, 0.02],
#         "kernel": [3, 5],
#     },
# }

# Valores ajustados após o tuning, utilizados na segunda rodada
GRADES = {
    "diff": {
        "limiar": [4, 6, 8, 10],
        "kernel": [5],
        "area_min": [50, 100],
    },
    "fundo": {
        "k": [3.0, 3.5],
        "alpha": [0.001, 0.0025, 0.005],
        "kernel": [1, 3],
    },
}

def ajustar(metodo):
    grade = GRADES[metodo]
    nomes = list(grade.keys())
    resultados = []

    # Gera todas as combinações possíveis dos valores da grade
    combinacoes = list(itertools.product(*grade.values()))
    print(f"\n{metodo}: testando {len(combinacoes)} combinações")

    for valores in combinacoes:
        # Parte dos parâmetros padrão e substitui só os que estão sendo testados
        params = dict(PARAMS_PADRAO[metodo])
        params.update(dict(zip(nomes, valores)))

        # Roda o pipeline sem salvar nada, avaliando somente o trecho de ajuste
        _, resultado = executar(metodo, params, FRAMES_AJUSTE)
        resultados.append({**dict(zip(nomes, valores)), **resultado})
        print(f"  {dict(zip(nomes, valores))} -> F = {resultado['f_measure']:.4f}")

    # Ordena da melhor para a pior configuração e salva a tabela completa
    tabela = pd.DataFrame(resultados).sort_values("f_measure", ascending=False)
    (PASTA_SAIDA / "csv").mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PASTA_SAIDA / "csv" / f"ajuste_{metodo}.csv", index=False)

    melhor = tabela.iloc[0]
    print(f"\nMelhor configuração ({metodo}):")
    for nome in nomes:
        print(f"  {nome}: {melhor[nome]}")
    print(f"  F-measure: {melhor['f_measure']:.4f}")


if __name__ == "__main__":
    inicio = time.perf_counter()
    for metodo in GRADES:
        ajustar(metodo)
    print(f"\nTempo total: {time.perf_counter() - inicio:.1f} s")