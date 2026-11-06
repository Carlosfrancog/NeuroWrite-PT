"""Garante que a amostra manuscrita de 3 linhas só use as 26 letras A-Z."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.evaluate_handwriting import carregar_mascara, recortar_caracteres
from scripts.evaluate_handwriting_online import avaliar_az


class ModeloControlado:
    classes_ = np.arange(26)

    def predict_proba(self, entradas):
        assert entradas.shape == (26, 784), "A terceira linha não deve ser enviada à rede"
        probabilidades = np.full((26, 26), 0.1 / 26)
        probabilidades[np.arange(26), np.arange(26)] += 0.9
        return probabilidades


class TesteEscritaSemAcentos(unittest.TestCase):
    def test_26_letras_sem_terceira_linha(self):
        mascara = carregar_mascara(Path("tests/fixtures/linha_caderno_mask.b64"))
        imagens, rotulos, linhas = recortar_caracteres(mascara)
        self.assertEqual(len(imagens), 38)
        self.assertEqual(rotulos[:26], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        with TemporaryDirectory() as pasta:
            destino = Path(pasta)
            tabela = avaliar_az(ModeloControlado(), imagens, rotulos, linhas, destino)
            self.assertEqual(len(tabela), 26)
            self.assertTrue(tabela["correto"].all())
            self.assertEqual(set(tabela["linha"]), {1, 2})
            self.assertTrue((destino / "letras_mlp_incremental_AZ.png").is_file())
            self.assertTrue((destino / "predicoes_mlp_incremental_AZ.csv").is_file())


if __name__ == "__main__":
    unittest.main()
