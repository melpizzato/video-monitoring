import cv2


def limpar_mascara(mascara, kernel):
    if kernel <= 1:
        return mascara

    # Elemento elíptico se adapta melhor ao formato arredondado das silhuetas
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

    # Abertura: elimina pontos isolados menores que o
    # elemento, como ruído e pequenos reflexos, sem encolher os objetos grandes
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)

    # Fechamento: preenche buracos internos e une
    # fragmentos do mesmo pedestre que ficaram separados (ex.: tronco e pernas)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, elemento, iterations=2)
    return mascara