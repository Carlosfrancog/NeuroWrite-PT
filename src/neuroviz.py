"""Visualização leve e reproduzível da MLP real, compatível com Colab e GitHub Actions.

NumPy calcula contribuições e seleções; Matplotlib/Agg renderiza sem interface Tk.
Linhas filtradas NÃO são conexões inventadas: todas vêm de coefs_.
Snapshots registram pesos efetivos após partial_fit; NÃO são gradientes.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.full_network import propagacao_real


@dataclass(frozen=True)
class Snapshot:
    epoca: int
    entrada: np.ndarray
    ativacoes: tuple[np.ndarray, ...]
    pesos: tuple[np.ndarray, ...]
    classes: tuple


def capturar_snapshot(pipeline, exemplo: np.ndarray, epoca: int) -> Snapshot:
    """Cópia independente dos pesos e ativações reais, sem alterar o modelo."""
    rede = pipeline.named_steps["modelo"]
    if len(rede.coefs_) != 2:
        raise ValueError("Este visualizador suporta uma camada oculta (784→128→26).")
    entrada = np.asarray(exemplo, dtype=np.float32).reshape(-1)
    ativacoes = propagacao_real(pipeline, entrada)
    return Snapshot(
        epoca=int(epoca), entrada=entrada.copy(),
        ativacoes=tuple(np.asarray(a).copy() for a in ativacoes),
        pesos=tuple(np.asarray(w).copy() for w in rede.coefs_),
        classes=tuple(rede.classes_),
    )


def _rotulo(classe) -> str:
    """Classes do EMNIST são inteiros 0..25, mapeados para A..Z."""
    try:
        k = int(classe)
        if k == classe and 0 <= k < 26:
            return chr(65 + k)
    except (TypeError, ValueError):
        pass
    return str(classe)


def _estilo(ax, titulo):
    ax.set_facecolor("#0b1220")
    ax.figure.patch.set_facecolor("#0b1220")
    ax.set_title(titulo, color="white", loc="left", fontsize=14, pad=20)
    ax.axis("off")


def _linha(ax, a, b, valor, limite=1.0):
    cor = "#6fd1e3" if valor >= 0 else "#f994a5"
    espessura = .5 + 2.2 * min(abs(float(valor)) / max(limite, 1e-9), 1)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=cor, linewidth=espessura,
            alpha=.36, zorder=1)


def _figura(snapshot: Snapshot, modo: str, top_k: int):
    if modo not in ("grande", "compacta"):
        raise ValueError("Modo deve ser 'grande' ou 'compacta'.")
    if top_k < 1:
        raise ValueError("top_k precisa ser positivo.")
    entrada, escondida, saida = snapshot.ativacoes
    w0, w1 = snapshot.pesos
    n_in, n_hid, n_out = len(entrada), len(escondida), len(saida)
    assert w0.shape == (n_in, n_hid) and w1.shape == (n_hid, n_out)
    # Ranking de ativação absoluta; contribuição até cada saída é a_h * W_h,o.
    destaques = np.argsort(-np.abs(escondida))[:min(top_k, n_hid)]
    classes = [_rotulo(c) for c in snapshot.classes]
    vencedor = int(np.argmax(saida))
    titulo = (f"NeuroWrite | Época {snapshot.epoca} | "
              f"{classes[vencedor]}: {saida[vencedor]:.1%} | {modo}")
    fig, ax = plt.subplots(figsize=(20, 12) if modo == "grande" else (15, 9))
    _estilo(ax, titulo)
    y_saida = np.linspace(.95, .05, n_out)
    if modo == "grande":
        y_entrada = np.linspace(.95, .05, n_in)
        y_oculta = np.linspace(.95, .05, n_hid)
        ax.scatter(np.zeros(n_in), y_entrada, s=2.1, c="#87a5ba", alpha=.65, zorder=3)
        ax.scatter(np.full(n_hid, 1.5), y_oculta, s=13, c="#466076", alpha=.5, zorder=3)
        ax.scatter(np.full(len(destaques), 1.5), y_oculta[destaques],
                   s=40, c="#f1bb69", zorder=4)
        for h in destaques:
            contribuicoes = entrada * w0[:, h]
            fontes = np.argsort(-np.abs(contribuicoes))[:6]
            lim = max(float(np.max(np.abs(contribuicoes[fontes]))), 1e-10)
            for i in fontes:
                _linha(ax, (0, y_entrada[i]), (1.5, y_oculta[h]),
                       contribuicoes[i], lim)
        ax.text(0, 1.03, f"{n_in} entradas", color="#e2e8f0", ha="center")
        ax.text(1.5, 1.03, f"{n_hid} ocultos · top {len(destaques)} destacado",
                color="#e2e8f0", ha="center")
    else:
        # A imagem, em vez das 784 linhas, preserva a interpretação de pixels.
        ax.imshow(snapshot.entrada.reshape(28, 28), cmap="gray",
                  extent=(-.20, .75, .20, .80), aspect="auto", zorder=3)
        y_oculta = np.linspace(.92, .08, len(destaques))
        ax.scatter(np.full(len(destaques), 1.5), y_oculta,
                   s=65 + 95 * np.abs(escondida[destaques]) /
                   max(float(np.max(np.abs(escondida[destaques]))), 1e-10),
                   c="#f1bb69", zorder=4)
        for p, h in enumerate(destaques):
            ax.text(1.65, y_oculta[p], f"H{h:03d}", color="#e2e8f0",
                    fontsize=9, va="center")
        ax.text(.27, 1.03, "Entrada 28×28", color="#e2e8f0", ha="center")
        ax.text(1.5, 1.03, f"Top {len(destaques)} ativos",
                color="#e2e8f0", ha="center")

    x_saida = 3.25
    ax.scatter(np.full(n_out, x_saida), y_saida, s=36,
               c=["#f1bb69" if i == vencedor else "#6b829d" for i in range(n_out)],
               zorder=4)
    for i, prob in enumerate(saida):
        ax.text(x_saida + .13, y_saida[i], f"{classes[i]}  {prob:5.1%}",
                color="#f1bb69" if i == vencedor else "#d7e2ec",
                fontsize=10, va="center", weight="bold" if i == vencedor else "normal")

    for p, h in enumerate(destaques):
        y_h = y_oculta[h] if modo == "grande" else y_oculta[p]
        contrib = escondida[h] * w1[h]
        # Exibir somente conexões com maior contribuição para cada neurônio.
        destinos = np.argsort(-np.abs(contrib))[:3]
        lim = max(float(np.max(np.abs(contrib[destinos]))), 1e-10)
        for o in destinos:
            _linha(ax, (1.5, y_h), (x_saida, y_saida[o]), contrib[o], lim)
    ax.text(x_saida, 1.03, f"{n_out} saídas · probabilidades", color="#e2e8f0",
            ha="center")
    n_pesos = int(sum(p.size for p in snapshot.pesos))
    ax.text(1.65, -.035,
            f"Rede real: {n_in}→{n_hid}→{n_out} · {n_pesos:,} pesos"
            " · linhas exibidas = filtro por contribuição (não é o grafo completo)".replace(",", "."),
            color="#9db5ca", fontsize=10, ha="center")
    ax.text(1.65, -.085,
            "Azul: contribuição positiva · rosa: negativa · H = índice do neurônio oculto",
            color="#9db5ca", fontsize=9, ha="center")
    ax.set_xlim(-.45, 4.4)
    ax.set_ylim(-.13, 1.1)
    fig.subplots_adjust(left=.02, right=.98, top=.9, bottom=.06)
    return fig


def exportar_png(snapshot: Snapshot, destino: str | Path, modo="compacta",
                 top_k=12, dpi=135) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig = _figura(snapshot, modo, top_k)
    try:
        fig.savefig(destino, dpi=dpi, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return destino


def exportar_gif(snapshots: list[Snapshot], destino: str | Path,
                 duracao_ms=650, top_k=10) -> Path:
    """Timelapse dos estados após cada época; não representa gradientes ao vivo."""
    if not snapshots:
        raise ValueError("São necessários checkpoints reais para gerar o GIF.")
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for snap in snapshots[:30]:  # limite de memória/tempo
        fig = _figura(snap, "compacta", top_k)
        try:
            memoria = BytesIO()
            fig.savefig(memoria, format="png", dpi=75, facecolor=fig.get_facecolor())
            memoria.seek(0)
            with Image.open(memoria) as imagem:
                frames.append(imagem.convert("RGB").copy())
        finally:
            plt.close(fig)
    frames[0].save(destino, save_all=True, append_images=frames[1:],
                   duration=max(100, int(duracao_ms)), loop=0, optimize=True)
    return destino
