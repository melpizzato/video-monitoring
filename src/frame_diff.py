import cv2
import numpy as np


class DiferencaFrames:
    """Solução inicial: compara cada frame com o imediatamente anterior."""

    def __init__(self, limiar):
        self.limiar = limiar
        self.anterior = None

    def aplicar(self, cinza):
        # No primeiro frame ainda não existe referência, retorna máscara vazia
        if self.anterior is None:
            self.anterior = cinza
            return np.zeros_like(cinza, dtype=np.uint8)

        # Diferença absoluta pixel a pixel (só o que mudou entre os dois instantes fica claro)
        diferenca = cv2.absdiff(self.anterior, cinza)

        # Diferenças acima do limiar viram movimento (255), o resto vira fundo (0)
        _, mascara = cv2.threshold(diferenca, self.limiar, 255, cv2.THRESH_BINARY)

        # O frame atual vira a referência da próxima iteração
        self.anterior = cinza
        return mascara