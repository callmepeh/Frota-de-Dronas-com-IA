# Melhorias propostas para o aprendizado por reforço

## O que foi observado

O ambiente passava `default_seed` para cada `reset()` sem seed explícita. Como o Stable-Baselines3 reinicia o ambiente entre episódios sem passar uma seed nova em todos eles, o agente treinava repetidamente na mesma sequência de chegadas e operações. Isso favorecia memorizar um único cenário e reduzia a generalização.

A recompensa anterior dependia de uma EWMA global e tinha saltos grandes (`+10`, `0` ou `-50`). Além disso, o estado não informava diretamente o MMT nem a taxa de violações da janela recém-concluída. Isso dificultava ligar uma ação recente ao seu efeito no serviço.

## Alterações implementadas

### `drone_fleet_env.py`

- A primeira reinicialização sem seed continua reproduzível a partir de `--seed`; as seguintes avançam o gerador do Gymnasium e recebem cenários diferentes.
- O espaço de observação passou de 9 para 11 dimensões: foram adicionados MMT normalizado da janela recente e proporção recente de entregas fora do SLA.
- A recompensa agora fornece sinal gradual: considera MMT médio recente e fração dentro do SLA, penaliza frota operacional e pressão da fila. Os pesos ficam em `RLParams` (`r_quality`, `r_sla`, `r_per_drone`, `r_queue`).

### `train_rl.py`

- DQN passa a usar 500 mil passos como padrão, rede MLP `[128, 128]`, taxa de aprendizado `3e-4`, replay buffer de 100 mil transições, batch 128, 10 mil passos de aquecimento e desconto `gamma=0.995`.
- A exploração decai ao longo de 35% do treino; sincronização da rede-alvo a cada 2 mil passos.

### `README.md`

- Atualizada a descrição do estado, recompensa e configuração de treino.
- Registrado que o modelo salvo existente tem arquitetura de observação antiga e precisa ser treinado novamente antes de avaliação; os resultados antigos são mantidos como histórico.

## Protocolo experimental implementado

`train_rl.py` agora semeia também o RNG do Gymnasium no primeiro reset e seleciona o melhor checkpoint por episódios de validação separados do teste. O avaliador usa 30 seeds por padrão, aceita diretório de saída independente, salva dados por seed, intervalos t de 95% e diferenças pareadas. `experiment_rl.py` automatiza cinco treinos independentes e avalia cada modelo na mesma bateria de teste, sem sobrescrever logs ou checkpoints de outra execução.

Para executar o protocolo completo:

```bash
cd rl
python3 experiment_rl.py --algo dqn --timesteps 500000 \
  --train-seeds 42 43 44 45 46 \
  --eval-seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

O protocolo multi-seed já foi executado para DQN e MARL. O checkpoint antigo continua incompatível com o estado de 11 dimensões. QR-DQN e Decision Transformer estão integrados para a comparação seguinte; consulte o histórico completo para os resultados MARL e para o comando que executa todos os algoritmos.

## Limites da mudança

Os pesos da recompensa são um ponto de partida e devem ser ajustados conforme o objetivo do sistema (priorizar SLA ou reduzir frota). A melhoria de arquitetura e diversidade de cenários aumenta a chance de generalização, mas só um novo treino seguido de avaliação comparativa pode confirmar que os resultados melhoraram.

## Resultados MARL e novos algoritmos

O experimento MARL foi executado com cinco seeds de treino e 30 cenários comuns. A equipe alcançou MMT médio de 42,14 min, SLA de 90,20%, frota média 67,11 e 142,29 drone-h alugados. Ficou próxima do baseline DT em serviço, com menor uso de frota e aluguel e uma fila média maior. Detalhes por checkpoint e limitações estão no [histórico completo](HISTORICO_DESENVOLVIMENTO_RL.md#7-resultado-da-execução-marl).

QR-DQN permanece integrado via SB3-Contrib. Foi adicionada também uma implementação offline de Decision Transformer, treinada sobre trajetórias do baseline DT com exploração controlada. O comando `python3 experiment_rl.py --algo all` compara DQN, PPO, QR-DQN, MARL e Decision Transformer em seeds de teste comuns e consolida os CSVs. O protocolo, a configuração e as limitações metodológicas estão registrados no [histórico](HISTORICO_DESENVOLVIMENTO_RL.md#8-qr-dqn-e-decision-transformer-para-comparação).

A análise comparativa final está em [ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md](ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md). DQN, PPO, QR-DQN e MARL foram comparados com Decision Transformer após a retomada `dt_reexec_20261023`; cada método tem cinco modelos avaliados nos mesmos 30 cenários. DT reproduziu o baseline demonstrador sem ganho pareado estatisticamente claro e teve maior latência de inferência. O [notebook de visualização](visualizacao_resultados_rl.ipynb) gera cinco painéis e inclui a análise bootstrap.


### Correção do Decision Transformer

O erro observado e a correção do padding/validação estão descritos no [histórico](HISTORICO_DESENVOLVIMENTO_RL.md#correção-do-fluxo-de-treino-decision-transformer). O avaliador agora só é chamado depois de confirmar que o checkpoint existe. Para retomar, use o comando DT com um nome de experimento novo documentado na [análise comparativa](ANALISE_RESULTADOS_COMPARACAO_20261023_122550.md#diagnóstico-do-erro-do-decision-transformer).
