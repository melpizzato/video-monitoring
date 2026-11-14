import cv2


def preprocessar(frame, blur):
    # Converte para tons de cinzapara detectar movimento e reduz o processamento de 3 canais para 1
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suaviza com Gaussiano para reduzir ruído que poderia ser confundido com movimento
    # O tamanho do kernel precisa ser ímpar; blur <= 1 desativa a etapa
    if blur > 1:
        cinza = cv2.GaussianBlur(cinza, (blur, blur), 0)
    return cinza