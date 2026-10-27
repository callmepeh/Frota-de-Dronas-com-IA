# Análise do experimento DQN multi-seed

## Desenho e integridade dos dados

- Cinco treinamentos independentes: seeds 42–46; 500 mil passos cada.
- Cinco checkpoints foram escolhidos por avaliação de validação, sem usar as seeds de teste. O melhor passo variou bastante: 450k, 280k, 200k, 140k e 420k, respectivamente.
- Cada modelo foi testado em 30 seeds de demanda (1–30), comparado às mesmas políticas DT e estática nos mesmos cenários.
- O arquivo agregado contém 450 observações: 5 modelos × 30 seeds × 3 políticas. Não há métricas ausentes.
- Os resultados por treinamento e por cenário estão em `resumo_por_modelo.csv` e `resultados_por_seed_treino.csv`; avaliações individuais em `treino_seed_*/teste/`.

## Desempenho médio

As médias de DQN são primeiro calculadas por cada modelo sobre 30 seeds; a coluna DP mostra a variação entre os cinco modelos treinados.

| Política | MMT médio (min) | P95 (min) | SLA (%) | Frota média | Fila média | Drone-h alugados |
|---|---:|---:|---:|---:|---:|---:|
| DQN v2 (média; DP entre treinos) | 42,77; 0,82 | 97,02; 1,14 | 89,88; 0,45 | 66,73; 0,63 | 1,78; 0,92 | 134,68; 12,52 |
| Busca binária DT | 41,84 | 96,63 | 90,13 | 68,07 | 0,32 | 161,45 |
| Frota estática 60 | 65,15 | 132,47 | 67,93 | 60,00 | 27,48 | 0,00 |

## Comparação pareada DQN − DT

Foi calculado um bootstrap pareado cruzado com 10 mil réplicas: reamostragem de modelos treinados e de seeds de teste, mantendo a comparação DQN/DT emparelhada em cada cenário. Os intervalos são percentis de 95% e representam incerteza aproximada nos dois fatores do experimento.

| Métrica (DQN − DT) | Diferença | IC bootstrap 95% | Interpretação |
|---|---:|---:|---|
| MMT médio | +0,94 min | [+0,31; +1,69] | DQN é um pouco mais lento em média |
| MMT P95 | +0,39 min | [−0,92; +1,85] | Sem diferença clara na cauda |
| SLA | −0,25 p.p. | [−0,71; +0,19] | Sem diferença clara; não afirmar empate exato nem superioridade |
| Frota média | −1,34 drones | [−1,92; −0,83] | Redução consistente de frota |
| Fila média | +1,46 pedidos | [+0,83; +2,26] | DQN mantém fila maior |
| Drone-h alugados | −26,77 h | [−38,29; −16,29] | Redução de aproximadamente 16,6% |

O ganho do DQN sobre o DT é sobretudo econômico: frota e horas de aluguel menores. Esse ganho vem com MMT médio ligeiramente pior e mais fila. O SLA fica perto de 90% em ambos, abaixo de qualquer meta de 95% ou 99%. A decisão depende de quanto custa manter capacidade e de quanto a latência média adicional é aceitável.

## Comparação com frota estática

Contra a frota fixa, o DQN melhorou o SLA em 21,95 pontos percentuais (IC cruzado 95%: +19,99 a +24,11), reduziu o MMT em 22,37 minutos (−24,29 a −20,59) e reduziu a fila média em 25,69 pedidos (−28,12 a −23,50). Usou 6,73 drones a mais, em média (+6,13 a +7,28). Isso confirma a utilidade do dimensionamento adaptativo neste simulador; não isola uma vantagem do RL sobre o DT.

## Leitura como resultado de pesquisa

1. **Não há evidência de que DQN supere o DT em qualidade de serviço.** O MMT médio foi estatisticamente pior no bootstrap cruzado, enquanto SLA e P95 não tiveram diferença clara. A fila também foi maior.
2. **Há evidência de redução de recursos.** A frota e as drone-horas alugadas caíram em todos os cinco modelos, com os intervalos cruzados acima inteiramente abaixo de zero.
3. **A conclusão relevante é uma fronteira custo–serviço**, não uma vitória geral do algoritmo. DQN pode ser preferível se economizar aluguel justificar cerca de um minuto adicional de MMT médio e uma fila um pouco maior. Se a prioridade for latência/fila, DT continua preferível.
4. **A seleção de checkpoint variou bastante** (140k a 450k passos), sinal de que a política não estabiliza no mesmo ponto em todos os treinamentos. Avaliar mais seeds de treino, ou maior orçamento, pode alterar a comparação.
5. A eficiência energética é calculada por um proxy do simulador. A pequena diferença de entregas/Wh não é suficiente para alegar economia real de energia.

## Próxima decisão experimental

Antes de ajustar a rede, definir o requisito operacional. Se SLA ≥95% for obrigatório, a configuração atual não atende e deve-se testar uma formulação que imponha ou penalize diretamente essa meta, apresentando também a fronteira entre SLA e custo. Se a meta atual (~90%) for aceitável, comparar explicitamente o valor econômico de 26,8 drone-horas alugadas economizadas contra a degradação de latência/fila. O DT deve continuar como baseline principal.
