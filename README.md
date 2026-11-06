# NeuroWrite PT

**Projeto-terça — projeto independente do EduPredict/EPA.**

Estudo de **classificação supervisionada de caracteres manuscritos em português** realizado em Python/Google Colab, seguindo o notebook oficial da disciplina *Ciência de Dados e Aprendizado de Máquina*.

## Objetivo
Comparar **SVM (SVC)** e **Rede Neural MLP (MLPClassifier)** para reconhecer letras maiúsculas escritas à mão, incluindo caracteres acentuados, investigando a generalização entre diferentes escritores.

**Pergunta de pesquisa:** modelos treinados com a caligrafia de algumas pessoas conseguem classificar a escrita de uma pessoa não vista durante o treinamento?

## Integrantes
- **Carlos** — desenvolvimento, limpeza/rotulagem, experimentos e integração do notebook.
- **Ian** — coleta manuscrita, documentação do dataset e apresentação da preparação.
- **Isadora** — coleta e estudo das métricas/erros para apresentação.
- **Miguel** — coleta e estudo dos algoritmos e hiperparâmetros para apresentação.

## Primeiro marco
1. Ian, Isadora e Miguel produzem **1 folha piloto** cada (ver [protocolo](docs/protocolo_coleta.md)).
2. Carlos testa o recorte/rotulagem em 1 folha de cada pessoa.
3. Após validar as fotos, cada integrante completa **10 folhas**, com duas repetições dos 38 caracteres por folha.
4. Carlos gera o `manifest.csv` e avalia SVM e MLP usando separação por escritor (Leave-One-Writer-Out).

Quantidade **planejada**, antes de limpeza: 3 pessoas × 10 folhas × 2 repetições × 38 caracteres = **2.280 imagens individuais**.

## Estrutura
- `notebooks/NeuroWrite_PT.ipynb`: relatório e experimento no formato do professor.
- `docs/protocolo_coleta.md`: instruções para os autores das folhas.
- `data/README.md`: formato do dataset processado, sem dados pessoais.
- `requirements.txt`: bibliotecas Python usadas no experimento.

## Restrições da entrega
- Entrega: **09/11/2026**; apresentações: **10/11 ou 17/11**.
- O arquivo entregue é um **.ipynb já executado**, com código, tabelas, figuras e saídas persistidas.
- **Não executar nem autenticar durante a apresentação**. O notebook final deve ser legível offline, sem montagem do Google Drive ou downloads necessários para exibir resultados.
- Não enviar fotos originais, nomes pessoais em imagens ou segredos ao GitHub público.
- Não confundir classificação de **caracteres isolados** com OCR irrestrito de texto cursivo. Reconhecer frases curtas é um extra condicionado à segmentação e tempo disponível.

## Fonte pública opcional
[EMNIST — NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset): referência complementar para letras A–Z. As classes com acentos/cedilha são de coleta própria; **não misturar classes incompatíveis nem afirmar que EMNIST contém caracteres portugueses acentuados**.

## Situação
**Etapa inicial:** documentação e notebook estruturado; *nenhum modelo treinado ou resultado empírico disponível ainda*.
