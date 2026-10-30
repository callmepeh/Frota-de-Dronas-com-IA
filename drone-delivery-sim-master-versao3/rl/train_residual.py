# -*- coding: utf-8 -*-
"""
Treino da política RESIDUAL sobre o Digital Twin (experimento isolado 20261030).

A IA não substitui o DT: recebe a recomendação corrente do controlador Busca
Binária DT e escolhe apenas uma correção discreta {−4, −2, 0, +2, +4} drones;
frota_final = recomendação_DT + correção_IA, sempre respeitando frota mínima/
máxima e as regras de aluguel/devolução de apply_fleet_target.

Estado aprimorado (v3, 30 dims): tudo disponível no instante da decisão
(fila, idade da fila, demanda estimada + histórico de 4 decisões, previsões
do ciclo de demanda +5/+10/+20 min, frota ativa/ociosa/recarga/trânsitos,
P95/MMT/SLA/violações recentes, recomendação DT e gap residual).

Recompensa configurável (RewardProfile): 'legacy' (original, para ablação),
'servico', 'equilibrado' ou 'custo', ou caminho de JSON com pesos explícitos.

Uso:
  python3 train_residual.py --profile equilibrado --timesteps 150000 \\
      --seed 42 --out rl_resultados/experimentos/residual_dt_20261030/modelos/...
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from drone_fleet_env import DroneFleetEnv, load_params

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Treino DQN residual sobre o DT")
    ap.add_argument("--profile", required=True,
                    help="legacy (recompensa original, para ablação) | servico | "
                         "equilibrado | custo | caminho JSON")
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-seed", type=int, default=None,
                    help="seed base da validação (padrão: treino + 100000)")
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--fleet-min", type=int, default=None)
    ap.add_argument("--fleet-max", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()
    validation_seed = (args.validation_seed if args.validation_seed is not None
                       else args.seed + 100_000)
    if validation_seed == args.seed:
        ap.error("a seed de validação deve ser diferente da seed de treino")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    callback_dir = args.results_dir / "validacao"
    callback_dir.mkdir(parents=True, exist_ok=True)

    P = load_params(fleet_min=args.fleet_min, fleet_max=args.fleet_max)
    profile_arg = None if args.profile == "legacy" else args.profile
    csv_profile = args.profile.replace('/', '_')
    raw_env = DroneFleetEnv(P, seed=args.seed, obs_version="v3",
                            action_mode="residual", reward_profile=profile_arg)
    train_env = Monitor(raw_env)
    eval_params = load_params(fleet_min=args.fleet_min, fleet_max=args.fleet_max)
    eval_env = Monitor(DroneFleetEnv(
        eval_params, seed=validation_seed, obs_version="v3",
        action_mode="residual", reward_profile=profile_arg))

    # Hiperparâmetros idênticos ao DQN de train_rl.py (efeito isolado da
    # mudança de estado/ação/recompensa, não do algoritmo).
    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=128,
        learning_starts=10_000,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_final_eps=0.05,
        gamma=0.995,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        seed=args.seed,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(callback_dir),
        log_path=str(callback_dir),
        eval_freq=max(1, args.eval_freq),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=eval_callback,
                progress_bar=False)
    dt_train = time.perf_counter() - t0

    last_out = args.out.with_name(f"{args.out.stem}_last{args.out.suffix or '.zip'}")
    model.save(last_out)
    best_model_path = callback_dir / "best_model.zip"
    if best_model_path.exists():
        best = DQN.load(best_model_path, env=train_env)
        best.save(args.out)
        step_best = best.num_timesteps
        print(f">> Melhor checkpoint de validação ({step_best} passos) salvo em {args.out}")
    else:
        shutil.copy2(last_out, args.out)
        step_best = int(model.num_timesteps)
        print(">> Validação não produziu checkpoint; modelo final salvo")

    rewards = list(train_env.episode_returns)
    lengths = list(train_env.episode_lengths)
    csv_out = args.results_dir / f"treino_dqn_res_{csv_profile}_seed{args.seed}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "passos", "recompensa_total"])
        for i, (r, l) in enumerate(zip(rewards, lengths), start=1):
            w.writerow([i, int(l), float(r)])
    if rewards:
        n = min(100, len(rewards))
        print(f">> Recompensa média (últimos {n} eps): {sum(rewards[-n:]) / n:.1f}")
    eval_env.close()
    print(f">> Treino concluído ({dt_train / 60:.1f} min, "
          f"{args.timesteps / dt_train:.0f} passos/s); log: {csv_out}")


if __name__ == "__main__":
    main()
