import numpy as np
from src.dataset import ler_temporal_roi, ler_roi, ler_frame, ler_gt, total_frames

print("Frames:", total_frames())
print("Temporal ROI:", ler_temporal_roi())
roi = ler_roi()
print("ROI: shape", roi.shape, "| pixels válidos:", roi.sum())
frame = ler_frame(1)
print("Frame:", frame.shape, frame.dtype)

inicio, fim = ler_temporal_roi()
valores = set()
for n in range(inicio, fim + 1, 50):
    valores |= set(np.unique(ler_gt(n)).tolist())
print("Rótulos encontrados no GT:", sorted(valores))