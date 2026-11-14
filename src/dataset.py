import cv2
import numpy as np
from src.config import PASTA_SEQ

def ler_temporal_roi(pasta=PASTA_SEQ):
    inicio, fim = map(int, (pasta / "temporalROI.txt").read_text().split())
    return inicio, fim

def ler_roi(pasta=PASTA_SEQ):
    roi = cv2.imread(str(pasta / "ROI.bmp"), cv2.IMREAD_GRAYSCALE)
    return roi > 0

def ler_frame(n, pasta=PASTA_SEQ):
    return cv2.imread(str(pasta / "input" / f"in{n:06d}.jpg"))

def ler_gt(n, pasta=PASTA_SEQ):
    for ext in ("png", "bmp"):
        caminho = pasta / "groundtruth" / f"gt{n:06d}.{ext}"
        if caminho.exists():
            return cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    raise FileNotFoundError(f"GT do frame {n} não encontrado")

def total_frames(pasta=PASTA_SEQ):
    return len(list((pasta / "input").glob("in*.jpg")))

def iterar_frames(inicio=1, fim=None, pasta=PASTA_SEQ):
    # Ordem temporal garantida pelo índice numérico, não pela ordem do sistema de arquivos
    fim = fim or total_frames(pasta)
    for n in range(inicio, fim + 1):
        yield n, ler_frame(n, pasta)