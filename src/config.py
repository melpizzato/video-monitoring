from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SEQ = RAIZ / "data" / "pedestrians"
PASTA_SAIDA = RAIZ / "outputs"

# Rótulos do ground truth CDnet 2014
GT_ESTATICO = 0
GT_SOMBRA = 50          # tratado como negativo
GT_FORA_ROI = 85        # ignorado
GT_DESCONHECIDO = 170   # ignorado
GT_MOVIMENTO = 255      # positivo

# Divisão temporal (confirmar com temporalROI.txt)
FRAMES_INICIALIZACAO = (1, 299)
FRAMES_AJUSTE = (300, 599)
FRAMES_TESTE = (600, 1099)

# Parâmetros ótimos, escolhidos após o tuning
PARAMS_DIFF = {"blur": 15, "limiar": 6, "kernel": 5, "area_min": 100}
PARAMS_FUNDO = {
    "blur": 5, "alpha": 0.005, "alpha_frente": 0.001, "k": 3.0,
    "n_init": 50, "var_min": 16.0, "kernel": 3, "area_min": 300,
}