"""Testes locais rápidos: python -m unittest discover -s tests -p 'test_*.py'."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.augmentation import aumentar_lote
from src.online_training import ConfigTreino, treinar_incremental
from src.full_network import propagacao_real, desenhar_rede_completa


class TesteAugmentacao(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((3, 784), dtype=np.float32)
        self.X[:, 9*28+10:9*28+14] = 1.0

    def test_sem_mudanca_quando_probabilidade_zero(self):
        saida = aumentar_lote(self.X, np.random.default_rng(12), probabilidade=0)
        np.testing.assert_array_equal(self.X, saida)
        self.assertIsNot(self.X, saida)

    def test_reprodutivel_e_sem_mutacao_dos_originais(self):
        original = self.X.copy()
        a = aumentar_lote(self.X, np.random.default_rng(42), 1.0)
        b = aumentar_lote(self.X, np.random.default_rng(42), 1.0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.X, original)
        self.assertEqual(a.shape, self.X.shape)
        self.assertTrue((a >= 0).all() and (a <= 1).all())

    def test_validacao_nao_passa_pelo_gerador(self):
        rng = np.random.default_rng(2026)
        X_train = rng.random((15, 784), dtype=np.float32)
        y_train = np.arange(15) % 3
        # Validação com valores diferentes para detectar vazamento no scaler.
        X_val = np.full((3, 784), .99, dtype=np.float32)
        y_val = np.array([0, 1, 2])
        copia = X_val.copy()
        pipeline, hist = treinar_incremental(
            X_train, y_train, X_val, y_val,
            ConfigTreino(epocas=2, tamanho_lote=5, camada_oculta=(4,),
                         semente=42, augmentar=True),
        )
        np.testing.assert_array_equal(X_val, copia)
        np.testing.assert_allclose(
            pipeline.named_steps["escala"].mean_, X_train.mean(axis=0), atol=1e-6
        )
        self.assertEqual(len(hist), 2)


class TesteRedeReal(unittest.TestCase):
    def test_forward_deve_bater_com_sklearn(self):
        rng = np.random.default_rng(13)
        X = rng.random((12, 784), dtype=np.float32)
        y = np.arange(12) % 3
        scaler = StandardScaler().fit(X)
        modelo = MLPClassifier(hidden_layer_sizes=(3,), random_state=13,
                               early_stopping=False)
        modelo.partial_fit(scaler.transform(X), y, classes=np.array([0, 1, 2]))
        pipe = Pipeline([("escala", scaler), ("modelo", modelo)])
        ativacoes = propagacao_real(pipe, X[0])
        self.assertEqual(list(map(len, ativacoes)), [784, 3, 3])
        with TemporaryDirectory() as pasta:
            saida = Path(pasta) / "rede.png"
            meta = desenhar_rede_completa(pipe, X[0], saida, dpi=60)
            self.assertTrue(saida.is_file() and saida.stat().st_size > 1000)
            self.assertEqual(meta["total_pesos"], 784*3 + 3*3)
            self.assertEqual(meta["topologia"], [784, 3, 3])


if __name__ == "__main__":
    unittest.main()
