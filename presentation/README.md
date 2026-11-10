# NeuroWrite-PT — recursos de apresentação

Esta pasta contém os arquivos reais necessários para executar o notebook no Google Colab, **sem autenticação do Google Drive, sem ZIP e sem dados Base64 no código**.

## Arquivos
- `data/Isadora_Xy_1063_37_classes.npz` — 1.063 amostras normalizadas de caracteres (37 classes).
- `assets/metricas.csv` — referência do experimento público EMNIST.
- `assets/comparacao.png`, `assets/matriz_confusao_svm.png`, `assets/matriz_confusao_mlp.png`, `assets/distribuicao.png`, `assets/amostras.png`, `assets/resumo_diagnostico.csv`.

## Download no Colab
O notebook usa a primeira célula para obter os recursos por HTTPS do próprio GitHub, recriando as pastas locais `data/` e `assets/`. Os arquivos são verificados por SHA-256 antes de uso.

**Importante:** as amostras manuscritas pertencem a uma única escritora; os resultados em fotos reservadas não demonstram generalização entre escritores.

Na execução pública, mantenha o notebook e a pasta de recursos na mesma revisão do GitHub. A versão de apresentação fixa o hash do commit para não depender de alterações futuras na branch.
