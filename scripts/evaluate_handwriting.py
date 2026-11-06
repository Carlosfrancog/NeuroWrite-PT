"""Avaliação diagnóstica da caligrafia real sobre modelos públicos EMNIST.

A amostra de entrada é uma máscara binária derivada da fotografia encaminhada
pelo grupo. O arquivo original NÃO é publicado aqui. As 38 posições da foto
foram conferidas manualmente. Esta rotina NÃO pretende segmentar páginas
arbitrárias de caderno e NÃO executa retreinamento.

Somente 26 caracteres A–Z são avaliados: Ç e acentos estão FORA das classes
conhecidas pelos modelos EMNIST.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import zlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXTRAS = list("ÇÁÀÂÃÉÊÍÓÔÕÚ")
LABELS = [list("ABCDEFGHIJKLM"), list("NOPQRSTUVWXYZ"), EXTRAS]
# Centros de cada caractere, medidos NA imagem particular enviada pelo grupo.
CENTROS = [
    [30, 76, 123, 165, 196, 232, 266, 309, 354, 384, 413, 448, 476],
    [29, 76, 124, 164, 200, 230, 266, 305, 352, 389, 414, 449, 474],
    [29, 77, 118, 165, 198, 230, 264, 306, 351, 390, 418, 453],
]
FAIXAS_Y = [(15, 41), (41, 63), (63, 89)]
ALTURA, LARGURA = 99, 514


def carregar_mascara(caminho: Path) -> np.ndarray:
    pacote = base64.b64decode(caminho.read_text(encoding="utf-8").strip())
    bits = np.frombuffer(zlib.decompress(pacote), dtype=np.uint8)
    pixels = np.unpackbits(bits)[: ALTURA * LARGURA]
    if len(pixels) != ALTURA * LARGURA:
        raise ValueError("Dimensões incompatíveis na amostra de teste.")
    return pixels.reshape((ALTURA, LARGURA)).astype(np.uint8) * 255


def recortar_caracteres(mascara: np.ndarray):
    imagens, rotulos, linhas = [], [], []
    for linha, (xs, chars, (y0, y1)) in enumerate(zip(CENTROS, LABELS, FAIXAS_Y), start=1):
        assert len(xs) == len(chars)
        for i, (centro, caractere) in enumerate(zip(xs, chars)):
            x0 = int((xs[i-1] + centro) / 2) if i else max(0, centro - 23)
            x1 = int((centro + xs[i+1]) / 2) if i + 1 < len(xs) else min(LARGURA, centro + 23)
            retalho = mascara[y0:y1, x0:x1]
            yy, xx = np.where(retalho > 0)
            if len(xx) == 0:
                raise ValueError(f"Recorte vazio: linha {linha}, letra {caractere}")
            recorte = retalho[yy.min():yy.max()+1, xx.min():xx.max()+1]
            alto, largo = recorte.shape
            escala = 20.0 / max(alto, largo)
            novo_largo = max(1, round(largo * escala))
            novo_alto = max(1, round(alto * escala))
            reduzido = np.asarray(Image.fromarray(recorte).resize(
                (novo_largo, novo_alto), Image.Resampling.LANCZOS
            ), dtype=np.uint8)
            canvas = np.zeros((28, 28), dtype=np.uint8)
            oy = (28 - novo_alto) // 2
            ox = (28 - novo_largo) // 2
            canvas[oy:oy+novo_alto, ox:ox+novo_largo] = reduzido
            imagens.append(canvas)
            rotulos.append(caractere)
            linhas.append(linha)
    assert len(imagens) == 38
    assert rotulos[:26] == list(ALFABETO)
    assert rotulos[26:] == EXTRAS
    return np.stack(imagens), rotulos, linhas


def avaliar(nome: str, modelo, imagens, rotulos, linhas, pasta_saida: Path):
    entradas = imagens[:26].reshape(26, 784).astype(np.float32) / 255.0
    pred = np.asarray(modelo.predict(entradas), dtype=int)
    if (pred < 0).any() or (pred >= 26).any():
        raise ValueError(f"Rótulos inesperados de {nome}: {sorted(set(pred))}")
    previsto = [ALFABETO[k] for k in pred]
    tabela = pd.DataFrame({
        "linha": linhas[:26], "real": rotulos[:26],
        "previsto": previsto,
        "correto": [a == b for a, b in zip(rotulos[:26], previsto)],
    })
    tabela.to_csv(pasta_saida / f"predicoes_{nome.lower()}.csv", index=False)
    acertos = int(tabela["correto"].sum())

    fig, eixos = plt.subplots(2, 13, figsize=(16, 4.5))
    for i, ax in enumerate(eixos.flat):
        ax.imshow(imagens[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{rotulos[i]} → {previsto[i]}",
                     color="green" if tabela.iloc[i]["correto"] else "firebrick", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{nome} | Caligrafia inédita: {acertos}/26 letras reconhecidas", fontsize=13)
    fig.tight_layout()
    fig.savefig(pasta_saida / f"letras_{nome.lower()}.png", dpi=150)
    plt.close(fig)
    return {"modelo": nome, "acertos": acertos, "total": 26, "acuracia_diagnostica": acertos / 26}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path,
                        default=Path("tests/fixtures/linha_caderno_mask.b64"))
    parser.add_argument("--models-dir", type=Path,
                        default=Path("results/emnist"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/handwriting"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mascara = carregar_mascara(args.fixture)
    imagens, rotulos, linhas = recortar_caracteres(mascara)
    Image.fromarray(mascara).save(args.output_dir / "amostra_binarizada.png")

    fig, eixos = plt.subplots(3, 13, figsize=(16, 5.5))
    for i, ax in enumerate(eixos.flat):
        ax.axis("off")
        if i < 38:
            ax.imshow(imagens[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(rotulos[i], fontsize=10)
    fig.suptitle("38 recortes conferidos — 26 A–Z e 12 fora do domínio EMNIST")
    fig.tight_layout()
    fig.savefig(args.output_dir / "recortes_conferidos.png", dpi=150)
    plt.close(fig)

    metricas = []
    for nome in ("SVM", "MLP"):
        caminho = args.models_dir / f"modelo_{nome.lower()}.joblib"
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo gerado pela Action não encontrado: {caminho}")
        modelo = joblib.load(caminho)  # somente modelos que a própria Action treinou
        metricas.append(avaliar(nome, modelo, imagens, rotulos, linhas, args.output_dir))
    resumo = pd.DataFrame(metricas)
    resumo.to_csv(args.output_dir / "resumo_diagnostico.csv", index=False)
    (args.output_dir / "AVISO.txt").write_text(
        "Avaliação diagnóstica: 26 letras A-Z de uma única foto submetida pelo grupo.\n"
        "A segmentação foi feita para este layout específico; a amostra foi inspecionada, "
        "portanto este NÃO é um teste cego de generalização.\n"
        "Os modelos NÃO foram retreinados com a imagem de teste.\n"
        "Os outros 12 caracteres (Ç e acentos) NÃO são classes suportadas pelo EMNIST Letters.\n"
        "Novos testes independentes, com outros escritores e fotos, serão necessários.\n",
        encoding="utf-8"
    )
    print(resumo.to_string(index=False))
    print("Amostras fora do domínio A–Z (sem previsão):", " ".join(EXTRAS))


if __name__ == "__main__":
    main()
