# Histórico do desenvolvimento de aprendizado por reforço

Este documento reúne o caminho percorrido no projeto, os motivos das mudanças, os resultados obtidos e a próxima hipótese experimental. As métricas de simulação se referem ao ambiente Python que emula o simulador Java; energia é um proxy do ambiente.

## 1. Ponto de partida

O projeto já possuía um ambiente Gymnasium (`drone_fleet_env.py`), uma réplica de referência da Busca Binária DT (`dt_baseline.py`), scripts de treino e avaliação, e checkpoints DQN/PPO.

O ambiente original tinha estado com 9 dimensões, sete ações discretas de ajuste da frota (−5, −3, −1, 0, +1, +3, +5) e recompensa em degraus baseada em EWMA do MMT: +10 abaixo do alvo, 0 dentro do SLA, −50 acima do SLA, além de penalidade de frota alugada. O `reset()` reaplicava a mesma seed inicial em todos os episódios quando havia `default_seed`, fazendo o agente repetir o mesmo cenário.

Os números históricos do README antes das alterações indicavam, em cinco cenários, DQN/RL com MMT médio 55,2 min, SLA 81,5% e frota 62,2; DT com 42,1 min, 89,7% e 68,0; frota estática 60 com 65,7 min e SLA 67,7%.

## 2. Primeira revisão da recompensa e do estado

Foram feitas as seguintes alterações:

- `drone_fleet_env.py`: a seed é usada para inicializar o RNG do Gymnasium no primeiro reset; resets sucessivos passam a gerar cenários diferentes, de forma reproduzível.
- O estado cresceu de 9 para 11 dimensões com MMT normalizado e fração de violações do SLA na janela recente.
- A recompensa em degraus foi substituída por sinal gradual baseado em MMT/SLA recentes, frota operacional e pressão de fila. Os pesos são `r_quality`, `r_sla`, `r_per_drone` e `r_queue`.
- `train_rl.py`: DQN passou para rede `[128, 128]`, learning rate `3e-4`, buffer 100 mil, batch 128, 10 mil passos de aquecimento, `gamma=0.995`, exploração ao longo de 35% dos passos e alvo atualizado a cada 2 mil passos. O orçamento padrão foi elevado a 500 mil.
- Checkpoints antigos de 9 dimensões tornaram-se incompatíveis; preservamos os arquivos antigos e salvamos os novos com nomes v2.

### Primeira avaliação v2

Com cinco seeds de teste, DQN obteve MMT 42,29 min, SLA 89,91% e frota 66,42. PPO obteve 42,79 min, 89,27% e 66,68. DT teve 42,10 min, 89,72% e 67,95. A frota estática teve 65,74 min e 67,74% de SLA. As avaliações foram preservadas em `rl_resultados/avaliacao_v2_dqn/` e `avaliacao_v2_ppo/`, pois o avaliador anterior sobrescrevia os mesmos arquivos.

Interpretação: ambos os agentes RL melhoraram muito sobre a frota estática. DQN e DT ficaram praticamente empatados em serviço; DQN usou menos frota. PPO foi ligeiramente inferior ao DQN. Cinco seeds não sustentavam afirmação de superioridade estatística.

## 3. Protocolo reproduzível implementado

Após a análise, foram implementados:

- `train_rl.py`: semeadura reproduzível do RNG do ambiente, avaliação periódica em ambiente separado e salvamento do melhor checkpoint pela recompensa média de validação, além do checkpoint final.
- `evaluate_rl.py`: 30 seeds por padrão, saída direcionada por `--out-dir`, CSV por seed, IC t de 95% e diferenças pareadas usando seeds comuns.
- `experiment_rl.py`: cinco treinamentos independentes por padrão (42–46), cada modelo testado nas mesmas 30 seeds, com diretórios e logs exclusivos.
- Os checkpoints automáticos antigos deixaram de ser escolhidos por padrão, evitando erro de shape por incompatibilidade de observação.
- `README.md`, `MELHORIAS_RL.md` e `ANALISE_RESULTADOS_RL_V2.md` documentam uso, alterações e limitações.

## 4. Experimento DQN com cinco treinos × 30 cenários

Executado em `rl_resultados/experimentos/dqn/20261023_101426/`. Foram concluídos cinco treinos independentes de 500 mil passos e 150 execuções de avaliação por política (cinco checkpoints × 30 cenários). Todos os dados esperados foram encontrados, sem métricas ausentes. Os melhores checkpoints ocorreram em 450k, 280k, 200k, 140k e 420k passos.

Média por modelo; DP de DQN é a variação entre as cinco seeds de treinamento:

| Política | MMT médio | MMT P95 | SLA | Frota média | Fila média | Drone-h alugados |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 42,77 ± 0,82 min | 97,02 ± 1,14 min | 89,88 ± 0,45% | 66,73 ± 0,63 | 1,78 ± 0,92 | 134,68 ± 12,52 |
| Busca Binária DT | 41,84 min | 96,63 min | 90,13% | 68,07 | 0,32 | 161,45 |
| Estática 60 | 65,15 min | 132,47 min | 67,93% | 60,00 | 27,48 | 0 |

Bootstrap pareado cruzado (10 mil réplicas, reamostrando seeds de treino e teste) para DQN − DT:

| Métrica | Diferença | IC bootstrap 95% |
|---|---:|---:|
| MMT médio | +0,94 min | [+0,31; +1,69] |
| MMT P95 | +0,39 min | [−0,92; +1,85] |
| SLA | −0,25 ponto percentual | [−0,71; +0,19] |
| Frota média | −1,34 drone | [−1,92; −0,83] |
| Fila média | +1,46 pedidos | [+0,83; +2,26] |
| Drone-h alugados | −26,77 h | [−38,29; −16,29] |

Conclusão: DQN economiza frota e aluguel frente ao DT, mas apresenta MMT médio um pouco maior e fila maior; SLA e P95 não diferiram de forma clara no intervalo cruzado. É uma troca custo–serviço, não uma vitória geral da RL. Nenhum controlador atingiu SLA de 95%. A comparação com frota estática mostra ganho claro do dimensionamento adaptativo, mas não separa RL de DT. PPO não foi repetido neste protocolo completo.

O relatório detalhado desta rodada está em `rl_resultados/experimentos/dqn/20261023_101426/ANALISE_RESULTADOS.md`.

## 5. Alternativa de algoritmo: QR-DQN

Foi adicionado QR-DQN como próximo candidato. Ele preserva o espaço discreto e estima a distribuição dos retornos por quantis; a documentação oficial confirma suporte a `Discrete` e uso de `MlpPolicy`. Essa escolha testa uma representação de valor diferente em ambiente estocástico sem alterar estado, ações, recompensa ou protocolo de avaliação. QR-DQN padrão continua escolhendo pela expectativa dos retornos; portanto, a técnica não é por si só uma otimização de cauda, SLA ou CVaR.

Configuração inicial para comparação justa: mesma rede oculta `[128, 128]`, buffer 100k, batch 128, warm-up 10k, `gamma=0.995`, exploração 35%, target update 2k e 50 quantis. Uma nova avaliação deve usar cinco seeds de treino e as mesmas 30 seeds de teste usadas no DQN.

O código aceita `qrdqn` em `train_rl.py`, `evaluate_rl.py` e `experiment_rl.py`. A dependência foi adicionada em `requirements.txt`; neste ambiente, `sb3-contrib` não estava instalado quando a integração foi preparada. Os treinos QR-DQN ainda não foram executados.

### Execução possível

```bash
cd rl
pip install -r requirements.txt
python3 experiment_rl.py --algo qrdqn --timesteps 500000 \
  --train-seeds 42 43 44 45 46 \
  --eval-seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

Cada execução cria uma pasta de experimento exclusiva em `rl_resultados/experimentos/qrdqn/`. QR-DQN permanece implementado como alternativa, mas a próxima comparação priorizada passou a ser MARL, conforme a decisão de explorar responsabilidades cooperativas.

## 6. Próxima linha priorizada: MARL cooperativo

Antes de executar QR-DQN, foi proposta uma política multiagente. O simulador não possui regiões ou drones autônomos com observações espaciais próprias; inventar tais agentes não seria fiel ao domínio. Foi então implementada uma decomposição cooperativa com duas decisões que já existem no problema:

1. **Agente de capacidade** escolhe um delta de frota entre os sete valores já usados pelo DQN.
2. **Agente de reserva/devolução** escolhe a reserva de drones alugados ociosos (0, 2, 4 ou 6) quando a fila está vazia, regulando quantos alugados permanecem disponíveis.

Os agentes atuam simultaneamente, recebem o estado global de 12 dimensões (o estado original mais o número normalizado de alugados ociosos) e aprendem com recompensa compartilhada. Cada agente tem rede Q própria e replay buffer próprio; o treino usa Independent Q-Learning cooperativo, replay 100k, batch 128, warm-up 10k, learning rate `3e-4`, `gamma=0.995`, epsilon decaindo em 35% do treino e target network atualizado a cada 2k passos. Checkpoint é escolhido por retorno compartilhado em validação separada.

Arquivos adicionados/alterados para esta linha:

- `drone_fleet_env.py`: observação estendida exclusiva do MARL e `step_multiagent`, que aplica as ações simultâneas e respeita fila, reserva, frota mínima e disponibilidade de drones alugados.
- `marl_train.py`: duas redes DQN independentes, treino cooperativo, validação, checkpoints e logs.
- `evaluate_rl.py`: carregamento do checkpoint MARL e avaliação nas mesmas métricas e seeds dos outros controladores.
- `experiment_rl.py`: opção `--algo marl`, cinco treinos por padrão e 30 seeds de teste compartilhadas.

### Próxima execução recomendada

```bash
cd rl
python3 experiment_rl.py --algo marl --timesteps 500000 \
  --train-seeds 42 43 44 45 46 \
  --eval-seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

A saída fica sob `rl_resultados/experimentos/marl/`. A execução foi concluída e analisada na seção seguinte. A formulação continua sendo uma hipótese experimental: agentes podem competir indiretamente pelas ações apesar da recompensa compartilhada, então não há garantia de melhora.

## 7. Resultado da execução MARL

Execução registrada em `rl_resultados/experimentos/marl/20261023_111409/`: cinco equipes treinadas (seeds 42–46), 500 mil passos por treino, validação separada e avaliação em 30 seeds de teste comuns. Foram preservados checkpoints, logs, séries por seed, resumos por modelo e agregados entre treinos. Os valores abaixo são médias dos cinco resumos de modelo; `±` é o desvio padrão entre checkpoints treinados, não o erro padrão dos 150 episódios.

| Política | MMT médio (min) | MMT P95 (min) | SLA (%) | Frota média | Fila média | Aluguel (drone-h) |
|---|---:|---:|---:|---:|---:|---:|
| MARL | 42,14 ± 0,31 | 96,29 ± 0,52 | 90,20 ± 0,18 | 67,11 ± 0,28 | 1,06 ± 0,38 | 142,29 ± 5,63 |
| Busca Binária DT | 41,84 | 96,63 | 90,13 | 68,07 | 0,32 | 161,45 |
| Estático 60 | 65,15 | 132,47 | 67,93 | 60,00 | 27,48 | 0,00 |

Leitura: o MARL ficou muito próximo da busca DT em MMT, P95 e SLA; a diferença pontual de SLA foi +0,08 ponto percentual, sem evidência suficiente nesta tabela para declarar superioridade. Usou cerca de 0,96 drone a menos e 19,16 drone-h alugados a menos por cenário, com fila média cerca de 0,74 pedido maior. Frente à frota estática, melhorou substancialmente serviço e fila, mas manteve frota média maior. Seu tempo médio de decisão foi aproximadamente 0,18 ms. O DQN da rodada anterior teve MMT médio 42,77 min e SLA 89,88%; a comparação serve como referência, mas a análise estatística deve usar pareamento por seed e contabilizar a variação entre treinos.

Arquivos de resultado principais: `resumo_entre_treinos.csv`, `resumo_por_modelo.csv` e `resultados_por_seed_treino.csv` na pasta da execução. A análise não sustenta que MARL seja globalmente melhor: houve troca entre serviço, fila, frota e aluguel, e nenhum método alcançou SLA de 95%.

## 8. QR-DQN e Decision Transformer para comparação

QR-DQN já está integrado via SB3-Contrib em `train_rl.py`, `evaluate_rl.py` e `experiment_rl.py`, com 50 quantis e parâmetros alinhados ao DQN. Não alterar observação, ação ou recompensa permite isolar a mudança de algoritmo. Sua execução ainda depende de `sb3-contrib` instalado no ambiente.

Foi adicionado `Decision Transformer` como algoritmo offline de sequência. `train_decision_transformer.py` coleta trajetórias de demonstração do controlador Busca Binária DT com 15% de exploração aleatória por padrão, separa episódios para validação e treina uma Transformer causal condicionada ao retorno-ainda-a-obter, estados e ações anteriores. `decision_transformer.py` contém a rede, adaptador de inferência com contexto de 20 passos e atualização do retorno restante. `experiment_rl.py --algo all` preserva o dataset offline de cada seed junto do modelo. O avaliador reinicia o contexto a cada cenário e compara métricas nas mesmas seeds.

O conjunto offline é criado a partir de seeds de coleta deslocadas em 200 mil em relação à seed de treino; as seeds de teste (1–30) não são usadas na coleta. O checkpoint é selecionado por perda de validação de previsão de ação. Como o dataset deriva do controlador DT e exploração aleatória, esta primeira versão testa aprendizado por sequência sobre demonstrações com ruído; não equivale a um benchmark offline amplo, e a perda de ação não garante maior recompensa na simulação.

Para treinar todos os métodos (DQN, PPO, QR-DQN, MARL e Decision Transformer) com cinco seeds e avaliar os mesmos 30 cenários:

```bash
cd rl
python3 experiment_rl.py --algo all --timesteps 500000 \
  --train-seeds 42 43 44 45 46 \
  --eval-seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

O comando sequencial cria pastas distintas em `rl_resultados/experimentos/` e uma consolidação em `rl_resultados/experimentos/comparacao_all/`. Para QR-DQN, instale antes as dependências de `requirements.txt`. A execução completa pode ser longa; nenhum treino novo foi iniciado nesta etapa.

## 9. Resultado observado da rodada comparativa `20261023_122550`

A execução iniciada com `--algo all` foi conferida nos arquivos de saída. DQN, PPO, QR-DQN e MARL concluíram cinco treinos (seeds 42–46) cada e as avaliações nos mesmos 30 cenários. Decision Transformer não deixou checkpoint, resumo de modelo ou avaliação; só foi encontrado o dataset offline de coleta da seed 42 (200 episódios). A pasta de consolidação `comparacao_all` também não foi criada, então os resumos separados são a fonte da análise. Não foi reiniciado nenhum treino.

Médias dos cinco modelos (`±` DP entre seeds de treino):

| Política | MMT médio (min) | MMT P95 (min) | SLA (%) | Frota média | Fila média | Aluguel (drone-h) | CPU/decisão (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | 42,775 ± 0,816 | 97,019 ± 1,138 | 89,882 ± 0,448 | 66,734 ± 0,626 | 1,785 ± 0,915 | 134,682 ± 12,522 | 0,167 ± 0,002 |
| PPO | 42,660 ± 0,343 | 96,460 ± 0,700 | 89,921 ± 0,216 | 67,185 ± 0,780 | 1,577 ± 0,368 | 143,699 ± 15,613 | 0,300 ± 0,026 |
| QR-DQN | 42,915 ± 0,510 | 97,556 ± 1,035 | 89,833 ± 0,305 | 66,440 ± 0,486 | 1,895 ± 0,428 | 128,791 ± 9,733 | 0,290 ± 0,041 |
| MARL | 42,140 ± 0,315 | 96,294 ± 0,517 | 90,203 ± 0,178 | 67,115 ± 0,281 | 1,061 ± 0,383 | 142,289 ± 5,629 | 0,188 ± 0,005 |
| Busca Binária DT | 41,840 | 96,626 | 90,127 | 68,073 | 0,320 | 161,453 | 0,044 |
| Estático 60 | 65,146 | 132,468 | 67,930 | 60,000 | 27,476 | 0,000 | 0,001 |

O MARL teve melhores valores pontuais entre os aprendidos para MMT, P95, SLA e fila; QR-DQN teve menor uso de frota e aluguel, com serviço ligeiramente pior. Busca Binária DT ainda tem menor MMT e fila pontuais que os algoritmos aprendidos. Nenhuma política alcançou SLA de 95%, e não há vencedor global sem priorização explícita de objetivos. A análise completa, incluindo intervalos pareados, está em `ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md`; o notebook `visualizacao_resultados_rl.ipynb` gera painéis separados para DQN, PPO, QR-DQN e MARL.

### Correção do fluxo de treino Decision Transformer

O traceback posterior mostrou que a avaliação tentava abrir `dt_seed42.pt` sem o arquivo existir. O problema estava na interação do padding à esquerda com as máscaras causal e de padding no Transformer: posições vazias podiam ficar sem chaves de atenção válidas e contaminar a perda de validação. O treino só gravava o checkpoint ao observar uma perda menor que a melhor anterior; por isso, sem perda válida, o script podia terminar sem modelo.

`train_decision_transformer.py` agora usa padding à direita, grava o dataset offline logo após a coleta e interrompe o processo se a validação produzir valor não finito ou se o checkpoint não for criado. `experiment_rl.py` confere a existência do checkpoint antes de avaliar e reporta a etapa/algoritmo que falhou durante `--algo all`, preservando os resultados anteriores. A retomada deve usar um `--name` novo porque a pasta da tentativa anterior já existe. O relatório `ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md` registra o comando de retomada exclusivo para DT. O treino corrigido ainda precisa ser executado para produzir resultados comparáveis.

## 10. Comparação final incluindo Decision Transformer

Após corrigir o padding do treino e retomar somente o DT, foram concluídos cinco checkpoints para seeds 42–46 e 150 cenários por checkpoint. Os cinco algoritmos agora podem ser comparados: DQN/PPO/QR-DQN/MARL da rodada `20261023_122550` e Decision Transformer da pasta `dt_reexec_20261023`.

| Política | MMT médio (min) | MMT P95 (min) | SLA (%) | Frota média | Fila média | Aluguel (drone-h) | CPU/decisão (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | 42,775 ± 0,816 | 97,019 ± 1,138 | 89,882 ± 0,448 | 66,734 ± 0,626 | 1,785 ± 0,915 | 134,682 ± 12,522 | 0,167 ± 0,002 |
| PPO | 42,660 ± 0,343 | 96,460 ± 0,700 | 89,921 ± 0,216 | 67,185 ± 0,780 | 1,577 ± 0,368 | 143,699 ± 15,613 | 0,300 ± 0,026 |
| QR-DQN | 42,915 ± 0,510 | 97,556 ± 1,035 | 89,833 ± 0,305 | 66,440 ± 0,486 | 1,895 ± 0,428 | 128,791 ± 9,733 | 0,290 ± 0,041 |
| MARL | 42,140 ± 0,315 | 96,294 ± 0,517 | 90,203 ± 0,178 | 67,115 ± 0,281 | 1,061 ± 0,383 | 142,289 ± 5,629 | 0,188 ± 0,005 |
| Decision Transformer | 41,774 ± 0,146 | 96,496 ± 0,528 | 90,180 ± 0,122 | 68,011 ± 0,140 | 0,330 ± 0,012 | 160,214 ± 2,788 | 2,065 ± 0,479 |
| Busca Binária DT | 41,840 | 96,626 | 90,127 | 68,073 | 0,320 | 161,453 | — |
| Estático 60 | 65,146 | 132,468 | 67,930 | 60,000 | 27,476 | 0,000 | — |

O Decision Transformer ficou estatisticamente próximo do demonstrador Busca Binária DT nas diferenças pareadas por cenário. Para DT − Busca Binária DT, MMT foi −0,066 min (IC bootstrap 95% [−0,222; 0,087]) e SLA +0,053 ponto percentual ([-0,087; 0,183]); todos os seis intervalos pareados incluem zero. Como o DT foi treinado sobre demonstrações desse baseline, o resultado sugere imitação bem-sucedida, sem demonstrar melhoria operacional. Sua latência média foi 2,065 ms, acima dos outros métodos. Nenhum algoritmo atingiu SLA de 95%; a leitura completa e as limitações estão em `ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md`.

O notebook `visualizacao_resultados_rl.ipynb` detecta agora o resumo do DT e gera também `comparacao_por_algoritmo_dt.png`. A execução `--algo all` original não criou a pasta de consolidação automática, por isso a tabela comparativa combina os resultados individuais preservados.
