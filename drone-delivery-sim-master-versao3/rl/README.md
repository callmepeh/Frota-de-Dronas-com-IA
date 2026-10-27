# Skill: RL para Dimensionamento de Frota (Drones-DT)

Substitui a **Busca Binária analítica do Digital Twin** (`SimScheduler.runDigitalTwin`)
por uma política de **Aprendizado por Reforço** treinada sobre um ambiente
Gymnasium que emula o simulador Java com fidelidade validada.

> **O projeto do simulador Java não é alterado nem executado por esta skill.**
> Os CSVs já existentes (`ad_mmt_log*.csv`, `system_state_log*.csv`) são
> reutilizados como referência de calibração/validação.

## Arquivos

| Arquivo | Papel |
|---|---|
| `drone_fleet_env.py` | Ambiente Gymnasium (MDP) + núcleo de eventos discretos réplica do Java |
| `dt_baseline.py` | Baseline: réplica da Busca Binária do DT (MMT estacionário M/M/n por Erlang-C) |
| `validate_env.py` | Valida o ambiente contra os CSVs reais do Java |
| `experiment_rl.py` | Treino multi-seed e avaliação estatística em seeds de teste comuns |
| `marl_train.py` | Dois agentes cooperativos: capacidade de frota e reserva/devolução de alugados |
| `decision_transformer.py` | Transformer causal condicionado ao retorno e ao histórico de estados/ações |
| `train_decision_transformer.py` | Coleta offline de trajetórias, treino e seleção por validação |
| `HISTORICO_DESENVOLVIMENTO_RL.md` | Histórico completo das políticas, mudanças e experimentos |
| `train_rl.py` | Treino DQN/PPO/QR-DQN com checkpoint selecionado por validação |
| `evaluate_rl.py` | RL × BuscaBináriaDT × Estático60 nos mesmos cenários (mesmas seeds) |
| `modelos/*.zip` | Modelos treinados |
| `rl_resultados/` | CSVs de validação, treino, séries por seed e gráfico comparativo |

## 1. Definição do MDP

| Componente | Definição |
|---|---|
| **Estado S** | fila/maxQueue, intensidade de demanda (`adMin/AD`), frota operacional, ociosos, em recarga, em trânsito, EWMA de MMT, fase do ciclo de demanda (sin/cos), MMT e violação de SLA da janela recente — 11 dimensões |
| **Ação A** | Δ frota ∈ {−5, −3, −1, 0, +1, +3, +5} aplicado a cada intervalo de decisão (5 min), limitada a `[fleet_min, fleet_max]` |
| **Recompensa R** | Sinal gradual por MMT e proporção de entregas dentro do SLA (80 min), com penalidade pequena por frota operacional e pressão de fila |

O Dynamic AD, os tempos de missão (estágios `30s/20min/…/20min/30s`), recarga
(15 s), trânsito de aluguel/devolução (15 s), fila máxima, reserva ociosa e a
regra de devolução são calculados dos parâmetros de `config.properties`.

## 2. Integração com o simulador

Dois níveis, sem tocar no Java:

1. **Ambiente emulado (usado aqui)** — o `DroneSimCore` reproduz, evento a
   evento, a mecânica de `DiscreteSimMain`/`SimScheduler`/`Drone`. Validação
   contra os CSVs do projeto (`validacao_ambiente.csv`): as execuções com frota
   fixa de 60 drones reproduzem os resultados do Java
   (ex.: MMT médio Java 64,1 → Python 62–65 min; SLA 69,1% → 63–74%;
   entregas 1.357 → 1.333–1.442).
2. **Ponte ao vivo (futura)** — `Gymnasium` padrão expõe `reset()`/`step(action)`;
   para treinar contra o Java real basta um wrapper que converse com o
   simulador via WebSocket/REST, mantendo o mesmo espaço de estado/ação.
   Os métodos já são compatíveis com a interface do Gym/Gymnasium.

## 3. Algoritmo

- **DQN** (padrão) — ideal para espaço de ações discreto. Rede MLP com duas camadas de 128 unidades. Treine:
  ```bash
  cd rl
  python3 train_rl.py --algo dqn    --timesteps 500000
  python3 train_rl.py --algo ppo    --timesteps 500000   # alternativa PPO
  python3 train_rl.py --algo qrdqn --timesteps 500000   # distribuição de retornos
  ```
- Biblioteca: Stable Baselines3 (replay buffer de 100 mil transições, batch 128,
  epsilon-greedy com decaimento em 35% do treino). Episódios sucessivos usam
  seeds distintas e reproduzíveis a partir de `--seed`.
- A observação agora tem 11 dimensões; checkpoints treinados com o estado antigo
  de 9 dimensões precisam ser recriados antes da avaliação.
- **QR-DQN** mantém ações discretas e estima a distribuição dos retornos por
  regressão quantílica (50 quantis), em vez de estimar diretamente só o retorno
  médio. É uma hipótese experimental para melhorar o aprendizado sob demanda
  estocástica; não garante resultado melhor nem otimiza CVaR/SLA automaticamente.
  A comparação controlada usa o mesmo experimento multi-seed descrito abaixo.
- **MARL cooperativo** usa dois agentes com redes e replay buffers separados:
  um escolhe o delta de frota e o outro escolhe a reserva de alugados ociosos
  (0/2/4/6). Eles observam o estado global e compartilham a recompensa. Use
  `experiment_rl.py --algo marl` para repetir sua avaliação.
- **Decision Transformer** aprende offline a prever ações com base no retorno a
  obter, estado e ações anteriores. O dataset é coletado do baseline DT com
  exploração aleatória controlada.

## 4. Execução e avaliação

```bash
python3 validate_env.py  # ambiente vs CSVs do Java
python3 evaluate_rl.py --algo dqn --model modelos/dqn_drone_fleet_v2.zip \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
  --out-dir rl_resultados/dqn_v2_teste
```

O avaliador usa 30 seeds comuns por padrão e grava métricas por seed, intervalos
de confiança t de 95% e diferenças pareadas contra DT e frota estática. Informe
`--out-dir` diferente para cada modelo para preservar as avaliações.

### Comparação completa multi-seed

Este fluxo treina cinco modelos independentes por algoritmo e testa cada modelo
nas mesmas 30 seeds. DQN/PPO/QR-DQN e MARL selecionam checkpoints por retorno de
validação; o Decision Transformer seleciona por perda de previsão de ação em
episódios offline separados. As seeds de avaliação não entram na coleta offline.

```bash
python3 experiment_rl.py --algo all --timesteps 500000 \
  --train-seeds 42 43 44 45 46 \
  --eval-seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

O modo `all` executa sequencialmente DQN, PPO, QR-DQN, MARL e Decision
Transformer, cada um com cinco treinos independentes e os mesmos 30 cenários de
teste. Os dados são salvos em pastas por algoritmo e consolidados em
`rl_resultados/experimentos/comparacao_all/`. QR-DQN requer `sb3-contrib`, já
listado em `requirements.txt`. O Decision Transformer coleta trajetórias
offline do baseline DT com exploração controlada. MARL usa agentes para decisões
do controlador central (capacidade da frota e reserva de alugados), pois o
simulador não modela drones independentes por região.

Comparação em cenários idênticos (mesma seed da demanda): entregas, MMT
médio/mediano/P95, SLA (80 min), fila média/máx, frota média, custo em drones,
energia (Wh), `deliveries/Wh`, nº de aluguéis/devoluções e **tempo de CPU por
decisão** (inferência da DQN × ~6 avaliações da busca binária).

### Resultados v2 observados (5 seeds; referência exploratória)

| Política | MMT médio | SLA ok | Frota média | CPU/decisão | Recompensa |
|---|---|---|---|---|---|
| Estático 60 | 65,7 min | 67,7% | 60,0 | 0,002 ms | −0,12 |
| Busca Binária (DT) | 42,1 min | 89,7% | 68,0 | 0,067 ms | +1,16 |
| DQN v2 | 42,3 min | 89,9% | 66,4 | 0,277 ms | +1,22 |
| PPO v2 | 42,8 min | 89,3% | 66,7 | 0,543 ms | +1,15 |
| MARL (5 modelos × 30 seeds) | 42,1 min | 90,2% | 67,1 | 0,177 ms | +1,23 |

Leitura: DQN e DT ficaram próximos no SLA; DQN usou cerca de 1,5 drone a menos
em média nos cinco cenários. Esse lote é exploratório e não demonstra
superioridade estatística. Consulte `ANALISE_RESULTADOS_RL_V2.md` para as
limitações. Parâmetros `r_quality`, `r_sla`, `r_per_drone` e `r_queue` em
`drone_fleet_env.py:RLParams` ajustam o equilíbrio entre qualidade, SLA, frota
e fila.

## Requisitos

```bash
pip install -r requirements.txt
```

## Análise integrada dos experimentos

Abra e execute `analise_integrada_ia.ipynb` de cima para baixo. O notebook
reúne o catálogo das políticas, resultados, tabelas e gráficos em um único
arquivo e lê os CSVs originais da rodada de 2026-10-23. Não treina modelos,
não executa o simulador Java e não cria arquivos de saída auxiliares. As
análises do simulador continuam em `analise_resultados_aprendizado_reforco.ipynb`
e `../analise_resultados_simulacao.ipynb`.

## Notas

- `fleet_min`/`fleet_max` (padrão 30–100) expandem os limites `min=max=60` do
  projeto; com eles a Busca Binária e a RL têm margem real de decisão.
- Os streams de RNG de chegadas são isolados do stream de operações, garantindo
  comparação justa entre políticas na mesma seed.
- A busca binária do baseline replica `runDigitalTwin`, incluindo a regra de
  devolver ociosos acima da reserva (`DT_IDLE_RESERVE`) quando a fila zera.
