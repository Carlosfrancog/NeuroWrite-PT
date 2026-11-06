"""Desenha TODOS os nós e TODOS os pesos de uma MLP real em alto zoom.

Não cria rede simplificada. Cada aresta representa um peso em coefs_,
cada neurônio representa uma entrada/unidade real. As cores dos nós
representam ativações calculadas por forward pass REAL do classificador.

O arquivo PNG tem resolução alta para ampliar detalhes no Colab.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from sklearn.pipeline import Pipeline


def propagacao_real(pipeline: Pipeline, entrada: np.ndarray):
    """Ativações de todas as camadas; verificadas contra predict_proba."""
    scaler = pipeline.named_steps["escala"]
    rede = pipeline.named_steps["modelo"]
    pixels = np.asarray(entrada, dtype=np.float32).reshape(1, -1)
    if len(rede.coefs_) == 0 or pixels.shape[1] != rede.coefs_[0].shape[0]:
        raise ValueError("A entrada não corresponde à rede treinada.")
    ativacoes = [scaler.transform(pixels)[0]]
    for camada, (pesos, bias) in enumerate(zip(rede.coefs_, rede.intercepts_)):
        z = ativacoes[-1] @ pesos + bias
        if camada < len(rede.coefs_) - 1:
            if rede.activation == "relu":
                z = np.maximum(0, z)
            elif rede.activation == "tanh":
                z = np.tanh(z)
            elif rede.activation == "logistic":
                z = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
            else:
                raise ValueError(f"Ativação oculta não suportada: {rede.activation}")
        else:
            if rede.out_activation_ != "softmax":
                raise ValueError("Visualização atual requer saída softmax multiclasses.")
            exp = np.exp(z - np.max(z))
            z = exp / exp.sum()
        ativacoes.append(z)
    if not np.allclose(ativacoes[-1], pipeline.predict_proba(pixels)[0], atol=1e-5):
        raise RuntimeError("Ativações desenhadas diferentes das previsões do modelo!")
    return ativacoes


def desenhar_rede_completa(
    pipeline: Pipeline,
    exemplo: np.ndarray,
    destino: str | Path,
    dpi: int = 190,
) -> dict:
    """Desenha a topologia completa (por exemplo 784 → 128 → 26).

    NENHUM neurônio nem peso é omitido. Renderização de 100 mil arestas
    pode demorar e gerar arquivo grande; use PNG em vez de SVG por padrão.
    """
    rede = pipeline.named_steps["modelo"]
    ativacoes = propagacao_real(pipeline, exemplo)
    tamanhos = [len(a) for a in ativacoes]
    coordenadas = [np.linspace(-1, 1, n) for n in tamanhos]
    total_pesos = int(sum(p.size for p in rede.coefs_))
    cores_nos = plt.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(22, 13))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")

    for camada, pesos in enumerate(rede.coefs_):
        p, q = pesos.shape
        a = np.column_stack((
            np.full(p * q, camada, dtype=np.float32),
            np.repeat(coordenadas[camada], q),
        ))
        b = np.column_stack((
            np.full(p * q, camada + 1, dtype=np.float32),
            np.tile(coordenadas[camada + 1], p),
        ))
        segmentos = np.stack((a, b), axis=1)
        valores = pesos.reshape(-1)
        limite = max(float(np.percentile(np.abs(valores), 98)), 1e-8)
        intensidade = np.clip(np.abs(valores) / limite, 0, 1)
        rgba = np.empty((len(valores), 4), dtype=np.float32)
        rgba[:, 0] = np.where(valores >= 0, 0.18, 0.95)
        rgba[:, 1] = np.where(valores >= 0, 0.70, 0.40)
        rgba[:, 2] = np.where(valores >= 0, 0.97, 0.62)
        rgba[:, 3] = 0.004 + 0.033 * intensidade
        colecao = LineCollection(
            segmentos, colors=rgba, linewidths=0.1 + 0.24 * intensidade,
            rasterized=True, zorder=1,
        )
        ax.add_collection(colecao)
        ax.text(
            camada + 0.5, 1.04, f"{p*q:,} pesos".replace(",", "."),
            color="#cbd5e1", ha="center", fontsize=11,
        )

    for camada, (coordenada, ativacao) in enumerate(zip(coordenadas, ativacoes)):
        # A escala de cor é independente por camada, para que nós de
        # amplitudes diferentes permaneçam visíveis. São valores REAIS.
        ativacao = np.asarray(ativacao, dtype=np.float64)
        minimo, maximo = np.percentile(ativacao, [2, 98])
        if abs(maximo - minimo) < 1e-10:
            maximo = minimo + 1
        normalizadas = np.clip((ativacao - minimo) / (maximo - minimo), 0, 1)
        s = 4 if len(ativacao) > 500 else (14 if len(ativacao) > 60 else 43)
        ax.scatter(
            np.full(len(ativacao), camada), coordenada,
            s=s, c=cores_nos(normalizadas), edgecolors="none", zorder=3,
            rasterized=True,
        )
        ax.text(
            camada, 1.13, f"Camada {camada}: {len(ativacao)} neurônios",
            color="white", ha="center", weight="bold", fontsize=14,
        )

    classe = int(np.argmax(ativacoes[-1]))
    classes = np.asarray(rede.classes_)
    ax.text(
        len(tamanhos) - 1, -1.13,
        f"Classe prevista: {classes[classe]} | confiança: {ativacoes[-1][classe]:.1%}",
        color="#a7f3d0", fontsize=12, ha="center",
    )
    ax.text(
        (len(tamanhos) - 1)/2, -1.24,
        f"Rede REAL: {' → '.join(map(str, tamanhos))}  •  {total_pesos:,} pesos".replace(",", "."),
        color="#e2e8f0", fontsize=15, ha="center",
    )
    ax.set_xlim(-0.25, len(tamanhos) - 0.75)
    ax.set_ylim(-1.31, 1.19)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "topologia": tamanhos,
        "total_pesos": total_pesos,
        "previsao": str(classes[classe]),
        "confianca": float(ativacoes[-1][classe]),
        "arquivo": str(destino),
    }
