import numpy as np
from src.config import GT_ESTATICO, GT_SOMBRA, GT_MOVIMENTO


def matriz_confusao(mascara, gt, roi):
    predito = mascara > 0

    # Positivo: apenas pixels anotados como movimento (255)
    positivo = (gt == GT_MOVIMENTO) & roi

    # Negativo: estático (0) e sombra (50). Sombra não faz parte do objeto,
    # então detectá-la é contado como falso positivo (critério oficial do CDnet)
    negativo = ((gt == GT_ESTATICO) | (gt == GT_SOMBRA)) & roi

    # Fora da ROI (85) e desconhecido (170) não entram em nenhum dos conjuntos,
    # logo são ignorados automaticamente
    sombra = (gt == GT_SOMBRA) & roi

    return {
        "tp": int(np.sum(predito & positivo)),
        "fp": int(np.sum(predito & negativo)),
        "fn": int(np.sum(~predito & positivo)),
        "tn": int(np.sum(~predito & negativo)),
        # Contagens separadas para analisar especificamente o erro causado por sombras
        "fp_sombra": int(np.sum(predito & sombra)),
        "n_sombra": int(np.sum(sombra)),
    }


def _div(a, b):
    return a / b if b > 0 else 0.0

def metricas(tp, fp, fn, tn):
    # true positive (tp), false positive (fp), false negative (fn), true negative (tn)
    recall = _div(tp, tp + fn)
    precisao = _div(tp, tp + fp)
    return {
        "recall": recall, # dos pixels que realmente são pedestre, quantos o método encontrou
        "precisao": precisao, # de tudo que o método marcou como movimento, quanto realmente era pedestre
        "f_measure": _div(2 * precisao * recall, precisao + recall), # média harmônica entre precisão e recall
        "especificidade": _div(tn, tn + fp), # dos pixels de fundo, quantos ficaram corretamente como fundo
        "fpr": _div(fp, fp + tn), # taxa de falso positivo
        "fnr": _div(fn, tp + fn), # taxa de falso negativo
        "pwc": 100 * _div(fn + fp, tp + fn + fp + tn),  # % de pixels classificados errado
    }