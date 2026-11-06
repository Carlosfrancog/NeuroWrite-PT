# Foto manuscrita — amostra diagnóstica

A foto enviada ao projeto em 06/11/2026 continha três linhas de caracteres manuscritos em folha pautada:
- 1ª linha: `ABCDEFGHIJKLM`
- 2ª linha: `NOPQRSTUVWXYZ`
- 3ª linha: `Ç Á À Â Ã É Ê Í Ó Ô Õ Ú`

## O que está versionado
O arquivo `linha_caderno_mask.b64` contém apenas a **máscara binária do traço**, sem cores do caderno, comprimida por `zlib` após `np.packbits` e codificada em Base64. Dimensões: **99 × 514** pixels. O original da fotografia não foi publicado no repositório.

O script `scripts/evaluate_handwriting.py` utiliza **posições medidas manualmente para esta foto** para criar 38 recortes 28 × 28. Não é segmentação automática geral.

## Cuidados com o experimento
- **Apenas A–Z** (26 letras) podem ser avaliadas pelos classificadores treinados em EMNIST Letters.
- Os 12 caracteres da terceira linha (Ç e acentos) aparecem somente na visualização; os modelos públicos ainda **não possuem essas classes**.
- **A Action não treina com esta foto**. Ela apenas carrega os modelos obtidos do EMNIST e executa predições.
- Por ser uma foto conhecida e com recortes ajustados à disposição da página, o teste é **diagnóstico de integração, não teste cego independente**.
- Retropropagação já ocorre durante o treinamento do MLP. Adaptar o modelo com novas amostras rotuladas exige separar exemplos próprios de treinamento, validação e um **teste final que nunca tenha sido usado para ajustes**.

Para consultar a avaliação, baixar o artefato da execução da Action na pasta `handwriting/`. Ele inclui os recortes conferidos, um CSV por modelo e uma figura comparativa por letra.
