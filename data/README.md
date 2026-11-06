# Dados do NeuroWrite PT

**Não versionar fotografias de manuscritos nem informações privadas de seus autores.** O dataset original é recebido por pasta privada compartilhada no Drive.

Estrutura local/Colab durante o desenvolvimento:

```text
data/
  raw/
    ian/
    isadora/
    miguel/
  processed/
    manifest.csv
    chars/
      ian_01_r01_A.png
      ...
```

Cada linha de `manifest.csv` representa um caractere individual recortado:

| coluna | exemplo | significado |
|---|---|---|
| `arquivo` | `data/processed/chars/ian_01_r01_A.png` | caminho de imagem individual |
| `classe` | `A` | caractere esperado (rótulo) |
| `escritor` | `ian` | ID estável do autor (grupo de validação) |
| `folha` | `01` | identificador da folha original |
| `repeticao` | `1` | primeira ou segunda sequência da folha |

O código do notebook assume que **as imagens já estão segmentadas e conferidas**. O processo automático de recorte será definido depois de inspecionar fotos piloto. Não supor que linhas pautadas do caderno desaparecem sem processamento.

O Colab é usado durante a execução do estudo, mas a apresentação do professor deve abrir o `.ipynb` **já executado**, sem login e sem execução ao vivo.

Fonte pública complementar opcional: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), separado por variante e classe para evitar incompatibilidades de rótulos.
