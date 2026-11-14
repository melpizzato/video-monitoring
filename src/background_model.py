import numpy as np


class ModeloFundoGaussiano:
    """Modelo de fundo que aprende como cada pixel costuma ser e vai se
    atualizando ao longo do vídeo."""

    def __init__(self, alpha, alpha_frente, k, n_init, var_min):
        self.alpha = alpha                # velocidade de aprendizado nas áreas de fundo
        self.alpha_frente = alpha_frente  # velocidade menor nas áreas com movimento
        self.k = k                        # tolerância para considerar um pixel diferente do fundo
        self.n_init = n_init              # quantidade de frames usados para montar o fundo inicial
        self.var_min = var_min            # oscilação mínima aceita, evita excesso de sensibilidade
        self.buffer = []
        self.media = None
        self.var = None

    def _inicializar(self):
        pilha = np.stack(self.buffer).astype(np.float32)

        # Monta o fundo inicial com a mediana dos primeiros frames
        # Como os pedestres estão sempre passando, eles somem do resultado e sobra só o cenário
        self.media = np.median(pilha, axis=0)

        # Mede o quanto cada pixel oscila naturalmente nesses frames (ruído, iluminação)
        self.var = np.maximum(np.mean((pilha - self.media) ** 2, axis=0), self.var_min)
        self.buffer = []

    def aplicar(self, cinza):
        # Enquanto não tem frames suficientes, só guarda as imagens e não detecta nada
        # Isso não afeta a avaliação, que começa no frame 300
        if self.media is None:
            self.buffer.append(cinza)
            if len(self.buffer) == self.n_init:
                self._inicializar()
            return np.zeros(cinza.shape, dtype=np.uint8)

        # Converte para float para fazer as contas sem estourar o limite de 0 a 255
        atual = cinza.astype(np.float32)
        diferenca = atual - self.media
        dist2 = diferenca ** 2

        # Compara o frame atual com o fundo aprendido
        # Um pixel é marcado como movimento quando foge muito do seu comportamento normal
        # Pixels que oscilam mais naturalmente precisam de uma mudança maior para serem marcados
        frente = dist2 > (self.k ** 2) * self.var

        # Define a velocidade de atualização de cada pixel:
        # o fundo se atualiza rápido, para acompanhar mudanças de luz;
        # as áreas com movimento se atualizam devagar, para o pedestre não virar parte do fundo
        taxa = np.where(frente, self.alpha_frente, self.alpha).astype(np.float32)

        # Atualiza aos poucos o fundo e a oscilação de cada pixel com o frame atual
        self.media += taxa * diferenca
        self.var = (1 - taxa) * self.var + taxa * dist2
        np.maximum(self.var, self.var_min, out=self.var)

        # Devolve a máscara: branco onde há movimento, preto onde é fundo
        return frente.astype(np.uint8) * 255