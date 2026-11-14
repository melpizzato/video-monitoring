import cv2
import numpy as np


def anotar(frame, caixas, texto):
    saida = frame.copy()
    # Desenha uma caixa verde em volta de cada região em movimento
    for x, y, w, h in caixas:
        cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(saida, texto, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return saida


def lado_a_lado(anotado, mascara):
    # A máscara tem 1 canal, converte para 3 para poder juntar com o frame colorido
    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    return np.hstack([anotado, mascara_bgr])