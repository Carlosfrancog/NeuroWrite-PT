"""Teste diagnóstico de A–Z manuscrito na MLP incremental, sem retreinar.

A foto original NÃO é versionada: usa apenas máscara anonimizada já existente.
Terceira linha (Ç e acentos) é descartada antes da inferência.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluate_handwriting import ALFABETO, carregar_mascara, recortar_caracteres


def avaliar_az(modelo, imagens, rotulos, linhas, destino: Path):
    if rotulos[:26] != list(ALFABETO):
        raise ValueError("A ordem esperada das duas primeiras linhas é A–Z.")
    # Nenhum pixel da terceira linha participa de predict_proba.
    entradas = imagens[:26].reshape(26, 784).astype(np.float32) / 255.0
    probabilidades = np.asarray(modelo.predict_proba(entradas))
    classes = np.asarray(modelo.classes_, dtype=int)
    if probabilidades.shape != (26, 26) or set(classes) != set(range(26)):
        raise ValueError("Modelo incremental incompatível: esperadas 26 classes EMNIST A–Z.")
    ordem = np.argsort(-probabilidades, axis=1)
    vencedores = classes[ordem[:, 0]]
    previsoes = [ALFABETO[i] for i in vencedores]
    linhas_tabela = []
    for idx, esperado in enumerate(rotulos[:26]):
        candidatos = [ALFABETO[classes[j]] for j in ordem[idx, :3]]
        confs = probabilidades[idx, ordem[idx, :3]]
        linhas_tabela.append({
            "linha": linhas[idx], "esperado": esperado,
            "previsto": previsoes[idx], "correto": previsoes[idx] == esperado,
            "confianca": float(confs[0]),
            "top_1": candidatos[0], "p_top_1": float(confs[0]),
            "top_2": candidatos[1], "p_top_2": float(confs[1]),
            "top_3": candidatos[2], "p_top_3": float(confs[2]),
        })
    tabela = pd.DataFrame(linhas_tabela)
    destino.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino / "predicoes_mlp_incremental_AZ.csv", index=False)
    acertos = int(tabela["correto"].sum())
    pd.DataFrame([{"modelo": "MLP_incremental_com_augmentacao",
                   "acertos": acertos, "total": 26,
                   "acuracia_diagnostica": acertos / 26}]).to_csv(
        destino / "resumo_mlp_incremental_AZ.csv", index=False)
    fig, eixos = plt.subplots(2, 13, figsize=(20, 5))
    for idx, ax in enumerate(eixos.flat):
        ax.imshow(imagens[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{rotulos[idx]} → {previsoes[idx]} ({tabela.iloc[idx]['confianca']:.0%})",
                     fontsize=9, color="seagreen" if tabela.iloc[idx]["correto"] else "firebrick")
        ax.axis("off")
    fig.suptitle(f"Sua caligrafia A–Z · MLP incremental · {acertos}/26 acertos · sem acentos",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(destino / "letras_mlp_incremental_AZ.png", dpi=155)
    plt.close(fig)
    (destino / "AVISO_MLP_INCREMENTAL.txt").write_text(
        "Avaliação diagnóstica com uma única fotografia pré-segmentada (26 A–Z). "
        "Não é avaliação cega e não mede generalização por escritor. "
        "O modelo foi treinado somente com EMNIST; a amostra NÃO participou de treino, "
        "augmentação, escolha de hiperparâmetros nem ajuste de pesos. "
        "Ç e 11 outros caracteres acentuados foram excluídos.\n",
        encoding="utf-8")
    return tabela


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path,
                        default=Path("tests/fixtures/linha_caderno_mask.b64"))
    parser.add_argument("--modelo", type=Path,
                        default=Path("results/online/com_augmentacao.joblib"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/handwriting"))
    args = parser.parse_args()
    modelo = joblib.load(args.modelo)  # arquivo gerado pela própria Action
    imagens, rotulos, linhas = recortar_caracteres(carregar_mascara(args.fixture))
    tabela = avaliar_az(modelo, imagens, rotulos, linhas, args.output_dir)
    print(f"Caligrafia A-Z: {int(tabela.correto.sum())}/26 letras corretas.")
    print(tabela[["esperado", "previsto", "confianca", "top_2", "top_3"]].to_string(index=False))


if __name__ == "__main__":
    main()
