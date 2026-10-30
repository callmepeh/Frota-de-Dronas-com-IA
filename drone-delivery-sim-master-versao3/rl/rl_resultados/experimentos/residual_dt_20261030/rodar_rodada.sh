#!/usr/bin/env bash
# Rodada completa do experimento residual_dt_20261030 (sequencial, reproduzível).
set -u
cd "$(dirname "$0")/../../.."   # rl/
EXP="rl_resultados/experimentos/residual_dt_20261030"
mkdir -p "$EXP/logs"
for perfil in legacy servico equilibrado custo; do
  for seed in 42 43; do
    tag="${perfil}_seed${seed}"
    echo "=== [$perfil $tag] treino $(date +%F_%T) ===" >> "$EXP/logs/rodada.log"
    if [[ -f "$EXP/modelos/dqn_res_${perfil}_estado_v3_seed${seed}.zip" ]]; then
      echo "skipped: treino $tag ja concluido" >> "$EXP/logs/rodada.log"; continue
    fi
    python3 train_residual.py --profile "$perfil" --timesteps 300000 \
      --seed "$seed" \
      --out "$EXP/modelos/dqn_res_${perfil}_estado_v3_seed${seed}.zip" \
      --results-dir "$EXP/treino_${perfil}_seed_${seed}/treino" \
      >> "$EXP/logs/treino_${tag}.log" 2>&1 || { echo "FALHA treino $tag" >> "$EXP/logs/rodada.log"; exit 1; }
    echo "=== [$tag] avaliacao $(date +%F_%T) ===" >> "$EXP/logs/rodada.log"
    python3 evaluate_residual.py \
      --model "$EXP/modelos/dqn_res_${perfil}_estado_v3_seed${seed}.zip" \
      --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
      --out-dir "$EXP/avaliacoes/${perfil}_seed_${seed}" \
      >> "$EXP/logs/avaliacao_${tag}.log" 2>&1 || { echo "FALHA avaliacao $tag" >> "$EXP/logs/rodada.log"; exit 1; }
    echo "OK $tag $(date +%F_%T)" >> "$EXP/logs/rodada.log"
  done
done
echo "RODADA COMPLETA $(date +%F_%T)" >> "$EXP/logs/rodada.log"
