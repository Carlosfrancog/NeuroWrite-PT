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


## Treinamento automatizado com dataset público (GitHub Actions)

Podemos começar **antes das fotos do grupo**. A workflow [Treinar modelos - EMNIST publico](https://github.com/Carlosfrancog/NeuroWrite-PT/actions/workflows/train-emnist.yml) roda automaticamente quando o notebook público muda, ou manualmente em **Actions → Run workflow**.

A Action executa o notebook [01_EMNIST_publico.ipynb](notebooks/01_EMNIST_publico.ipynb), baixa o EMNIST Letters, utiliza amostras balanceadas das divisões oficiais de treino/teste (100/20 exemplos por classe), realiza validação cruzada e comparação **SVM vs MLP**. Ao final, publica em **Artifacts** um pacote com:
- notebook `.ipynb` **já executado**, com saídas;
- `metricas.csv`, metadados e gráficos (amostras, distribuição, comparação, matrizes de confusão);
- modelos treinados `.joblib`.

**Limitações importantes:** a variante *EMNIST Letters* agrega minúsculas e maiúsculas em 26 classes A–Z; não possui Ç nem caracteres acentuados. O estudo dos colegas com **38 classes** exigirá reprocessamento e retreino. O modelo público não é OCR geral de frases, nem representa validação por escritor.

**Entrega acadêmica:** o `.ipynb` final precisa permanecer autossuficiente e com resultados salvos, sem login, Drive ou execução ao vivo na apresentação.

Ver também [workflow YAML](.github/workflows/train-emnist.yml).


## Experimento incremental: aumentação de escrita durante o treinamento

Implementado em 06/11/2026:
- [Gerador `src/augmentation.py`](src/augmentation.py): rotações de até 7°, escala ±6%, translações de até 1 pixel, pequeno cisalhamento e alterações leves de espessura. Mantém rótulos, número e ordem dos exemplos.
- [Treinamento `src/online_training.py`](src/online_training.py): a cada mini-lote, gera novas variantes **somente do treino**, aplica `partial_fit` na MLP e registra F1 da validação por época.
- [Experimento `scripts/run_online_emnist.py`](scripts/run_online_emnist.py): compara **sem** e **com** aumentação no mesmo recorte do EMNIST, com validação interna e teste oficial isolado até o fim.
- [Rede completa `src/full_network.py`](src/full_network.py): visualiza **938 neurônios e 103.680 conexões reais**, para arquitetura **784 → 128 → 26**, com pesos extraídos de `coefs_` e ativações reais verificadas contra `predict_proba`. PNG de alta resolução (zoom via Colab).
- [Testes `tests/test_online_training.py`](tests/test_online_training.py): garantem reprodutibilidade, não mutação dos dados, isolamento da validação e correspondência do cálculo de propagação direta.
- [Action automatizada](.github/workflows/train-emnist.yml): publica esses resultados em `results/online/` no pacote de artifacts.

**Primeiro resultado:** os resultados verificados da execução [42303717843](https://github.com/Carlosfrancog/NeuroWrite-PT/actions/runs/42303717843) foram:

| MLP incremental (6 épocas) | Acurácia teste oficial | F1 macro teste |
|---|---:|---:|
| Sem aumentação | 64,62% | 0,6436 |
| Com aumentação | 63,65% | 0,6273 |

A **augmentação aleatória não melhorou os resultados nessa configuração**. Isso não contradiz o objetivo: seu efeito deve ser investigado, não pressuposto. O experimento incremental também não usa o mesmo protocolo de ajustes do baseline SVM×MLP do notebook público; portanto, não comparar as pontuações como se fossem uma disputa controlada entre os três treinamentos.

**Observação importante sobre feedback:** a retropropagação atualiza os pesos da rede. O gerador on-the-fly básico **não recebe gradientes**; ele cria uma nova variante do treino no próximo mini-lote. A evolução adaptativa por erro do treino está planejada em [issue #3](https://github.com/Carlosfrancog/NeuroWrite-PT/issues/3). O teste real nunca entra no gerador.

### Próximas etapas (issues em português)
- [#1 Vídeo/timelapse com a rede REAL e zoom](https://github.com/Carlosfrancog/NeuroWrite-PT/issues/1)
- [#2 Pipeline das 30 folhas e recortes quadriculados](https://github.com/Carlosfrancog/NeuroWrite-PT/issues/2)
- [#3 Gerador adaptativo orientado por erro de treinamento](https://github.com/Carlosfrancog/NeuroWrite-PT/issues/3)
- [#4 Integração no notebook final do professor](https://github.com/Carlosfrancog/NeuroWrite-PT/issues/4)

O vídeo e o treinamento com os 38 caracteres do grupo ainda não estão implementados; são trabalhos planejados.
