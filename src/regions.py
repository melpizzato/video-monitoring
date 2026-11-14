import cv2
import numpy as np


def filtrar_regioes(mascara, area_min):
    # Rotula cada região branca conectada (connectivity=8: vizinhos nas diagonais também contam)
    # stats para cada rótulo: [x, y, largura, altura, área]
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)

    # O rótulo 0 é sempre o fundo, mantém só as regiões com área mínima
    validos = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min]

    # Reconstrói a máscara apenas com as regiões válidas, para que a máscara
    # avaliada e as caixas desenhadas fiquem sempre consistentes entre si
    mascara_filtrada = np.where(np.isin(rotulos, validos), 255, 0).astype(np.uint8)

    caixas = [tuple(int(v) for v in stats[i, :4]) for i in validos]
    return mascara_filtrada, caixas