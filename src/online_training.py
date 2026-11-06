"""Treinamento incremental de rede neural com variações on-the-fly.

Fluxo: lote real -> (opcional) augmentação -> StandardScaler treinado SOMENTE no
treino -> MLP.partial_fit -> backprop nos pesos -> próximo lote transformado.

Validação e teste não entram no gerador nem nos cálculos do scaler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.augmentation import aumentar_lote


@dataclass(frozen=True)
class ConfigTreino:
    epocas: int = 6
    tamanho_lote: int = 128
    camada_oculta: tuple[int, ...] = (128,)
    semente: int = 42
    augmentar: bool = False
    chance_augmentacao: float = 0.75


def treinar_incremental(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_validacao: np.ndarray,
    y_validacao: np.ndarray,
    configuracao: ConfigTreino = ConfigTreino(),
) -> tuple[Pipeline, pd.DataFrame]:
    """Retorna rede treinada e histórico com perda do último lote por época.

    O histórico da validação tem finalidade diagnóstica; o teste final NÃO é
    recebido por esta função. A seleção de hiperparâmetros deve acontecer
    antes de avaliar o teste reservado.
    """
    Xt = np.asarray(X_treino, dtype=np.float32)
    Xv = np.asarray(X_validacao, dtype=np.float32)
    yt = np.asarray(y_treino)
    yv = np.asarray(y_validacao)
    if len(Xt) != len(yt) or len(Xv) != len(yv):
        raise ValueError("Incompatibilidade entre entradas e rótulos.")
    if Xt.ndim != 2 or Xv.ndim != 2 or Xt.shape[1] != 784 or Xv.shape[1] != 784:
        raise ValueError("Esperadas imagens 28x28 vetorizadas (784 atributos).")
    if len(Xt) == 0 or len(Xv) == 0:
        raise ValueError("Treino e validação não podem ser vazios.")
    if not (np.isfinite(Xt).all() and np.isfinite(Xv).all()):
        raise ValueError("Os pixels devem ser números finitos.")
    if configuracao.epocas < 1 or configuracao.tamanho_lote < 1:
        raise ValueError("Épocas e tamanho do lote precisam ser positivos.")

    classes = np.unique(yt)
    if len(classes) < 2 or not np.isin(yv, classes).all():
        raise ValueError("A validação deve conter apenas classes presentes no treino.")
    scaler = StandardScaler()
    scaler.fit(Xt)  # nunca ajustar na validação/teste

    modelo = MLPClassifier(
        hidden_layer_sizes=configuracao.camada_oculta,
        solver="adam", learning_rate_init=0.001,
        early_stopping=False, shuffle=False,
        random_state=configuracao.semente,
        max_iter=1,
    )
    rng = np.random.default_rng(configuracao.semente)
    linhas = []
    primeiro = True

    for epoca in range(1, configuracao.epocas + 1):
        indices = rng.permutation(len(Xt))
        for inicio in range(0, len(indices), configuracao.tamanho_lote):
            idx = indices[inicio:inicio + configuracao.tamanho_lote]
            lote = Xt[idx]
            if configuracao.augmentar:
                lote = aumentar_lote(
                    lote, rng, probabilidade=configuracao.chance_augmentacao
                )
            lote_normalizado = scaler.transform(lote)
            if primeiro:
                modelo.partial_fit(lote_normalizado, yt[idx], classes=classes)
                primeiro = False
            else:
                modelo.partial_fit(lote_normalizado, yt[idx])
        previsao = modelo.predict(scaler.transform(Xv))
        linhas.append({
            "epoca": epoca,
            "treino": "com_augmentacao" if configuracao.augmentar else "sem_augmentacao",
            "loss_ultimo_lote": float(modelo.loss_),
            "acuracia_validacao": float(accuracy_score(yv, previsao)),
            "f1_macro_validacao": float(f1_score(yv, previsao, average="macro", zero_division=0)),
        })

    pipeline = Pipeline([("escala", scaler), ("modelo", modelo)])
    return pipeline, pd.DataFrame(linhas)
