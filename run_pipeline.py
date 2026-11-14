import argparse
import time

import cv2
import numpy as np
import pandas as pd

from src.config import PASTA_SAIDA, FRAMES_AJUSTE, FRAMES_TESTE, PARAMS_DIFF, PARAMS_FUNDO
from src.dataset import iterar_frames, ler_gt, ler_roi, ler_temporal_roi
from src.preprocess import preprocessar
from src.frame_diff import DiferencaFrames
from src.background_model import ModeloFundoGaussiano
from src.postprocess import limpar_mascara
from src.regions import filtrar_regioes
from src.evaluation import matriz_confusao, metricas
from src.visualization import anotar, lado_a_lado

PARAMS_PADRAO = {"diff": PARAMS_DIFF, "fundo": PARAMS_FUNDO}

# Velocidade da exibição na tela e do vídeo salvo
FPS_EXIBICAO = 60


def criar_detector(metodo, params):
    # Cria o detector de movimento escolhido com seus parâmetros
    if metodo == "diff":
        return DiferencaFrames(params["limiar"])
    if metodo == "fundo":
        return ModeloFundoGaussiano(
            params["alpha"], params["alpha_frente"], params["k"],
            params["n_init"], params["var_min"],
        )
    raise ValueError(f"Método desconhecido: {metodo}")


def intervalo_avaliacao(nome):
    # Traduz o nome do intervalo para a faixa de frames correspondente
    if nome == "ajuste":
        return FRAMES_AJUSTE
    if nome == "teste":
        return FRAMES_TESTE
    return ler_temporal_roi()


def executar(metodo, params, intervalo, salvar=False, mostrar=False, fps_exibicao=FPS_EXIBICAO):
    aval_ini, aval_fim = intervalo
    roi = ler_roi()
    detector = criar_detector(metodo, params)
    gravador = None

    # Cria as pastas de saída quando os resultados forem salvos
    if salvar:
        pasta_mascaras = PASTA_SAIDA / "masks" / metodo
        for pasta in (pasta_mascaras, PASTA_SAIDA / "videos", PASTA_SAIDA / "csv"):
            pasta.mkdir(parents=True, exist_ok=True)

    linhas = []
    # Percorre o vídeo sempre desde o primeiro frame e em ordem,
    # porque os métodos dependem do histórico anterior ao trecho avaliado
    for n, frame in iterar_frames(1, aval_fim):
        # Mede apenas o tempo de processamento, sem contar a leitura do disco
        inicio = time.perf_counter()
        cinza = preprocessar(frame, params["blur"])
        mascara = detector.aplicar(cinza)
        mascara = limpar_mascara(mascara, params["kernel"])
        mascara, caixas = filtrar_regioes(mascara, params["area_min"])
        tempo_ms = (time.perf_counter() - inicio) * 1000

        # Registra a atividade do frame: quantidade de regiões e área em movimento
        area_px = int(np.count_nonzero(mascara))
        linha = {
            "frame": n,
            "n_regioes": len(caixas),
            "area_px": area_px,
            "area_pct": 100 * area_px / mascara.size,
            "tempo_ms": tempo_ms,
        }

        # Compara com o ground truth apenas dentro do intervalo de avaliação
        if aval_ini <= n <= aval_fim:
            linha.update(matriz_confusao(mascara, ler_gt(n), roi))
        linhas.append(linha)

        # Monta o painel com o frame anotado ao lado da máscara
        if salvar or mostrar:
            anotado = anotar(frame, caixas, f"{metodo} | frame {n} | regioes: {len(caixas)}")
            painel = lado_a_lado(anotado, mascara)

        # Salva a máscara do frame e adiciona o painel ao vídeo de saída
        if salvar:
            cv2.imwrite(str(pasta_mascaras / f"bin{n:06d}.png"), mascara)
            if gravador is None:
                h, w = painel.shape[:2]
                gravador = cv2.VideoWriter(
                    str(PASTA_SAIDA / "videos" / f"{metodo}.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"), fps_exibicao, (w, h),
                )
            gravador.write(painel)

        # Exibe o painel na tela, esperando o tempo necessário para manter o FPS definido
        if mostrar:
            cv2.imshow("Pipeline", painel)
            gasto = time.perf_counter() - inicio
            espera_ms = max(1, int((1 / fps_exibicao - gasto) * 1000))
            cv2.waitKey(espera_ms)

    if gravador is not None:
        gravador.release()
    if mostrar:
        cv2.destroyAllWindows()

    serie = pd.DataFrame(linhas)

    # Soma os acertos e erros de todos os frames avaliados e calcula as métricas finais
    avaliados = serie.dropna(subset=["tp"])
    tot = avaliados[["tp", "fp", "fn", "tn", "fp_sombra", "n_sombra"]].sum()
    resultado = metricas(tot.tp, tot.fp, tot.fn, tot.tn)
    resultado["fpr_sombra"] = tot.fp_sombra / tot.n_sombra if tot.n_sombra else 0.0
    resultado["n_sombra"] = tot.n_sombra

    # Tempo médio considerando apenas os frames avaliados
    resultado["tempo_medio_ms"] = avaliados["tempo_ms"].mean()
    resultado["fps_processamento"] = 1000 / resultado["tempo_medio_ms"]

    if salvar:
        serie.to_csv(PASTA_SAIDA / "csv" / f"{metodo}_serie.csv", index=False)

    return serie, resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metodo", default="diff", choices=["diff", "fundo"])
    parser.add_argument("--intervalo", default="ajuste", choices=["ajuste", "teste", "completo"])
    parser.add_argument("--salvar", action="store_true")
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()

    _, resultado = executar(
        args.metodo, PARAMS_PADRAO[args.metodo],
        intervalo_avaliacao(args.intervalo), args.salvar, args.mostrar,
    )
    print(f"\nMétodo: {args.metodo} | intervalo: {args.intervalo}")
    for chave, valor in resultado.items():
        print(f"{chave:>18}: {valor:.4f}")