# Análise dos resultados RL v2

## Síntese

Foram reavaliados os checkpoints `dqn_drone_fleet_v2.zip` e `ppo_drone_fleet_v2.zip` nos mesmos cinco cenários (seeds 1, 2, 3, 5 e 7), junto da busca binária DT e da frota estática de 60 drones. Os resumos e séries foram preservados em `rl_resultados/avaliacao_v2_dqn/` e `rl_resultados/avaliacao_v2_ppo/` porque o avaliador grava sempre nos mesmos nomes e uma execução substitui a anterior.

| Política | MMT médio (min) | P95 MMT (min) | SLA (%) | Frota média | Fila média | Entregas/Wh | CPU decisão (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN v2 | 42,29 | 96,02 | 89,91 | 66,42 | 1,36 | 0,01234 | 0,277 |
| PPO v2 | 42,79 | 97,85 | 89,27 | 66,68 | 1,56 | 0,01224 | 0,543 |
| Busca binária DT | 42,10 | 97,21 | 89,72 | 67,95 | 0,34 | 0,01212 | 0,067 |
| Estática 60 | 65,74 | 134,61 | 67,74 | 60,00 | 27,89 | 0,01230 | 0,002 |

Valores são médias de cinco execuções; `Q_fila_max` também é média dos máximos por execução no CSV agregado.

## Interpretação

1. **DQN é a melhor das duas políticas aprendidas neste lote.** Contra PPO, obteve MMT médio cerca de 0,51 min menor, SLA 0,64 ponto percentual maior, frota 0,26 drone menor e inferência aproximadamente duas vezes mais rápida. As diferenças são modestas.
2. **DQN e DT estão praticamente empatados no serviço observado.** O DQN ficou 0,19 ponto percentual acima em SLA, mas seu MMT médio foi 0,19 min maior e sua fila média foi maior (1,36 contra 0,34). Em contrapartida, reduziu a frota média em 1,53 drone (cerca de 2,3%) e o aluguel acumulado em cerca de 19% (128,48 contra 158,98 drone-horas). Isso sugere uma pequena vantagem de custo/capacidade, sem evidência de uma vantagem geral de desempenho.
3. **RL melhora claramente sobre a frota fixa de 60 no compromisso SLA/latência**, usando em média 6,42 drones adicionais: SLA subiu de 67,74% para 89,91%, MMT caiu de 65,74 para 42,29 min e fila média caiu de 27,89 para 1,36. Essa comparação evidencia o valor do dimensionamento adaptativo, não uma superioridade do RL sobre o controlador DT.
4. Nenhuma política chega perto de 100% no SLA de 80 min: aproximadamente 10% das entregas ainda excedem o limite, e o P95 fica acima de 96 min. O objetivo precisa definir se esse nível é aceitável; se não for, será necessário priorizar mais a cauda/SLA na recompensa ou estabelecer o SLA como restrição.
5. A métrica energia é um proxy do ambiente, não uma medição física completa. As diferenças de `Entregas_por_Wh` são pequenas e não sustentam uma conclusão forte de eficiência energética.

## Variação entre as seeds

O DQN ganhou do DT em frota em todas as cinco seeds (redução de 1,21 a 1,91 drones). Em SLA, ganhou em quatro das cinco, mas perdeu 1,10 ponto percentual na seed 3. Os resultados médios de SLA foram DQN 89,91% (desvio padrão entre seeds 0,67), DT 89,72% (0,39) e PPO 89,27% (0,89). Com apenas cinco seeds, intervalos de confiança são largos; diferenças de décimos de ponto percentual não devem ser tratadas como evidência estatística de superioridade.

Os retornos médios por episódio aumentaram no treino: DQN, média dos últimos 100 episódios 278,79 contra 166,89 dos primeiros 100; PPO 293,32 contra 157,61. Isso indica aprendizado em relação à recompensa configurada, mas não implica automaticamente melhor SLA fora do treino. PPO ilustra essa diferença: maior retorno final médio que DQN no log de treino, mas avaliação operacional ligeiramente inferior.

## Limitações metodológicas encontradas

- **Poucas seeds de avaliação:** cinco cenários são insuficientes para uma comparação estatística robusta. Recomenda-se no mínimo 30 seeds de demanda e apresentar média, desvio padrão e intervalo de confiança pareado.
- **Uma execução de treino por algoritmo:** seed de treinamento única não mede a variabilidade de inicialização/aprendizado. Treinar pelo menos 5 seeds independentes por algoritmo e avaliar cada política na mesma bateria de seeds de teste.
- **Reprodutibilidade dos treinos v2 já realizados:** esses treinos ocorreram antes da correção de propagação da seed para o `np_random` do Gymnasium; eles não devem ser tratados como replicações completamente reproduzíveis. A implementação atual corrige esse ponto e grava logs por execução.
- **Checkpoint final:** o script salva o modelo do último passo, não seleciona checkpoint por validação. Separar seeds de treino/validação/teste e selecionar checkpoint apenas pela validação reduz o risco de sobreajuste.
- **Múltiplos objetivos:** a recompensa agrega MMT, taxa de SLA, frota e fila em pesos escolhidos manualmente. O resultado depende desses pesos; relatar a curva de compromisso (SLA × frota) é mais informativo do que uma única recompensa.
- **Referência Java:** a validação estática existente indica que o simulador emulado é razoável, mas há diferenças (por exemplo, Java MMT 64,1 min versus emulação de 64,9 min em uma seed). As conclusões valem primeiro para o ambiente emulado.

## Recomendação de pesquisa

Para a próxima rodada, manter o DT como baseline forte e usar DQN como candidato. Corrigir primeiro a propagação da seed de treino; então executar pelo menos cinco seeds de treinamento por algoritmo e 30 seeds de teste comuns. Reportar diferenças pareadas e intervalos de confiança para SLA, MMT médio/P95, frota média, drone-horas alugadas, fila e custo computacional. Se o requisito for SLA ≥ 95%, ajustar a formulação para penalizar diretamente violações por janela e P95, e declarar explicitamente a restrição de custo. Não concluir que a RL supera o DT com os dados atuais.

O protocolo para essa próxima rodada foi implementado em `experiment_rl.py`, `train_rl.py` e `evaluate_rl.py`. Ele ainda precisa ser executado para produzir a nova evidência; os números desta análise permanecem os resultados exploratórios anteriores.
