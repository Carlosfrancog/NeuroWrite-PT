"""Executa comparação exploratória de treino incremental COM e SEM aumentação.

Executar pela raiz do repositório:
    python -m scripts.run_online_emnist

Reutiliza o download feito pelo notebook EMNIST durante a GitHub Action.
As métricas no teste oficial são calculadas apenas APÓS todos os treinos.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import struct
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.augmentation import aumentar_lote
from src.online_training import ConfigTreino, treinar_incremental
from src.full_network import desenhar_rede_completa

RAIZ = Path("notebooks/data/public/emnist_letters")
SAIDA = Path("results/online")
SAIDA.mkdir(parents=True, exist_ok=True)
SEMENTE = 42


def ler_idx(nome: str) -> np.ndarray:
    arquivo = RAIZ / nome
    if not arquivo.is_file():
        raise FileNotFoundError(
            f"{arquivo} não encontrado. Execute antes o notebook público EMNIST."
        )
    with gzip.open(arquivo, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic not in (2049, 2051):
            raise ValueError(f"IDX inválido: {arquivo.name}")
        dimensoes = magic & 0xFF
        shape = struct.unpack(">" + "I" * dimensoes, f.read(4 * dimensoes))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).copy()


def subset(nome: str, por_classe: int, seed: int):
    images = ler_idx(f"emnist-letters-{nome}-images-idx3-ubyte.gz")
    labels = ler_idx(f"emnist-letters-{nome}-labels-idx1-ubyte.gz")
    assert images.shape[0] == len(labels)
    indices = []
    rng = np.random.default_rng(seed)
    for classe in range(1, 27):
        indices.extend(rng.choice(np.flatnonzero(labels == classe), por_classe, replace=False))
    rng.shuffle(indices)
    # Mesmo tratamento de orientação do notebook público.
    X = images[indices].transpose(0, 2, 1).reshape(-1, 784).astype(np.float32) / 255
    y = labels[indices].astype(int) - 1
    return X, y


def main():
    X_publico, y_publico = subset("train", por_classe=100, seed=SEMENTE)
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_publico, y_publico, test_size=0.2,
        random_state=SEMENTE, stratify=y_publico,
    )
    # O conjunto oficial de TESTE só é carregado após criar treino/validação.
    # Nunca é passado para o gerador ou função de treinamento.
    historicos, modelos = [], {}
    for ativar in (False, True):
        nome = "com_augmentacao" if ativar else "sem_augmentacao"
        modelo, historico = treinar_incremental(
            X_treino, y_treino, X_val, y_val,
            ConfigTreino(epocas=6, tamanho_lote=128,
                         camada_oculta=(128,), semente=SEMENTE,
                         augmentar=ativar),
        )
        historicos.append(historico)
        modelos[nome] = modelo
        joblib.dump(modelo, SAIDA / f"{nome}.joblib", compress=3)
        print(nome, "F1 validação:", round(historico.iloc[-1]["f1_macro_validacao"], 4))
    historia = pd.concat(historicos, ignore_index=True)
    historia.to_csv(SAIDA / "historico_por_epoca.csv", index=False)

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    for nome, grupo in historia.groupby("treino"):
        eixos[0].plot(grupo["epoca"], grupo["f1_macro_validacao"], marker="o", label=nome)
        eixos[1].plot(grupo["epoca"], grupo["loss_ultimo_lote"], marker="o", label=nome)
    eixos[0].set(title="F1 macro — validação interna", xlabel="Época", ylabel="F1 macro")
    eixos[1].set(title="Loss do último mini-lote (não média da época)", xlabel="Época", ylabel="Loss")
    for ax in eixos:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_epocas.png", dpi=160)
    plt.close(fig)

    # Ilustrar a mesma entrada real em sua forma original e três variantes.
    exemplo = X_treino[:1].copy()
    rng = np.random.default_rng(2026)
    var = [aumentar_lote(exemplo, rng, probabilidade=1.0)[0] for _ in range(3)]
    fig, axs = plt.subplots(1, 4, figsize=(9, 2.8))
    for ax, vetor, titulo in zip(axs, [exemplo[0]] + var,
                                 ["Original", "Variação 1", "Variação 2", "Variação 3"]):
        ax.imshow(vetor.reshape(28, 28), cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SAIDA / "variacoes_sinteticas.png", dpi=160)
    plt.close(fig)

    X_teste, y_teste = subset("test", por_classe=20, seed=SEMENTE + 1)
    linhas = []
    for nome, modelo in modelos.items():
        previsto = modelo.predict(X_teste)
        linhas.append({
            "modelo": nome,
            "acuracia_teste": float(accuracy_score(y_teste, previsto)),
            "f1_macro_teste": float(f1_score(y_teste, previsto, average="macro")),
            "f1_macro_validacao_ultima_epoca": float(
                historia[historia["treino"] == nome].iloc[-1]["f1_macro_validacao"]
            ),
        })
    comparacao = pd.DataFrame(linhas)
    comparacao.to_csv(SAIDA / "comparacao_teste.csv", index=False)
    print("Avaliação única das variantes no conjunto de teste reservado:")
    print(comparacao.to_string(index=False))

    trace = desenhar_rede_completa(
        modelos["com_augmentacao"], X_teste[0],
        SAIDA / "rede_real_completa.png", dpi=180
    )
    (SAIDA / "rede_real_metadados.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Rede MLP real desenhada:", trace["topologia"], "pesos:", trace["total_pesos"])


if __name__ == "__main__":
    main()
