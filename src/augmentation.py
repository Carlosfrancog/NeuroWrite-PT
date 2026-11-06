"""Aumento de dados para caracteres manuscritos, aplicado SOMENTE em mini-lotes de TREINO.

As transformações usam a imagem em tons de cinza em grade fixa (28x28).
Não inventam escritores independentes. Não devem ser aplicadas ao teste.
NÃO implementam gradientes de imagem: o backprop atualiza os pesos da MLP,
e uma nova variação da mesma amostra é apresentada no mini-lote seguinte.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, grey_dilation, grey_erosion

LADO = 28
FEATURES = LADO * LADO


def aumentar_lote(
    X: np.ndarray,
    rng: np.random.Generator,
    probabilidade: float = 0.75,
) -> np.ndarray:
    """Gera novas variantes do lote, preservando original, forma e ordem.

    Parâmetros
    ----------
    X: float array (n, 784), pixels de 0 a 1, traço claro em fundo escuro.
    rng: gerador externo para reprodutibilidade.
    probabilidade: fração esperada de amostras alteradas por chamada.

    Nunca modifica `X` in-place. Os rótulos `y` permanecem inalterados,
    desde que as deformações suaves não alterem visualmente o caractere.
    """
    dados = np.asarray(X, dtype=np.float32)
    if dados.ndim != 2 or dados.shape[1] != FEATURES:
        raise ValueError("Esperado lote (n, 784), imagens 28x28.")
    if not 0.0 <= probabilidade <= 1.0:
        raise ValueError("A probabilidade precisa ficar entre 0 e 1.")
    if not np.isfinite(dados).all() or np.any(dados < 0) or np.any(dados > 1):
        raise ValueError("Os pixels devem ser finitos e estar em [0,1].")
    resultado = dados.copy().reshape(-1, LADO, LADO)
    centro = np.array([(LADO - 1) / 2, (LADO - 1) / 2], dtype=float)

    for i in range(len(resultado)):
        if rng.random() > probabilidade:
            continue
        origem = resultado[i]
        angulo = np.deg2rad(rng.uniform(-7.0, 7.0))
        c, s = np.cos(angulo), np.sin(angulo)
        escala = rng.uniform(0.94, 1.06)
        cisalhamento = rng.uniform(-0.045, 0.045)
        transformacao = np.array(
            [[c * escala, -s * escala + cisalhamento],
             [s * escala,  c * escala]], dtype=float
        )
        inversa = np.linalg.inv(transformacao)
        deslocamento = rng.uniform(-1.0, 1.0, size=2)
        offset = centro - inversa @ (centro + deslocamento)
        nova = affine_transform(
            origem, inversa, offset=offset,
            output_shape=(LADO, LADO), order=1,
            mode="constant", cval=0.0, prefilter=False,
        )
        # Espessura variável apenas ocasionalmente. Manter acentos legíveis.
        opcao = rng.random()
        if opcao < 0.06:
            nova = grey_dilation(nova, size=(2, 2))
        elif opcao < 0.12:
            nova = grey_erosion(nova, size=(2, 2))
        if rng.random() < 0.2:
            nova = gaussian_filter(nova, sigma=0.35)
        resultado[i] = np.clip(nova, 0, 1)

    return resultado.reshape(-1, FEATURES).copy()
