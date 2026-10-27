# -*- coding: utf-8 -*-
"""
Treino do agente RL (DQN, PPO ou QR-DQN) sobre DroneFleetEnv.

Uso:
  python3 train_rl.py --algo dqn    --timesteps 500000
  python3 train_rl.py --algo ppo    --timesteps 500000
  python3 train_rl.py --algo qrdqn --timesteps 500000

O melhor checkpoint de validação é salvo em rl/modelos/<algo>_drone_fleet_v2.zip
e os logs são gravados em rl/rl_resultados/.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from drone_fleet_env import DroneFleetEnv, load_params

HERE = Path(__file__).resolve().parent
MODELOS = HERE / "modelos"
RESULTADOS = HERE / "rl_resultados"


def main():
    ap = argparse.ArgumentParser(description="Treino RL do dimensionamento de frota")
    ap.add_argument("--algo", choices=["dqn", "ppo", "qrdqn"], default="dqn")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-seed", type=int, default=None,
                    help="seed base da validação (padrão: seed de treino + 100000)")
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--fleet-min", type=int, default=None)
    ap.add_argument("--fleet-max", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTADOS,
                    help="diretório exclusivo para logs de treino/validação")
    args = ap.parse_args()
    validation_seed = (args.validation_seed if args.validation_seed is not None
                       else args.seed + 100_000)
    if validation_seed == args.seed:
        ap.error("a seed de validação deve ser diferente da seed de treino")

    MODELOS.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    P = load_params(fleet_min=args.fleet_min, fleet_max=args.fleet_max)
    raw_env = DroneFleetEnv(P, seed=args.seed)
    env = Monitor(raw_env)  # Monitor acumula episode_returns/episode_lengths sem limite
    eval_params = load_params(fleet_min=args.fleet_min, fleet_max=args.fleet_max)
    eval_env = Monitor(DroneFleetEnv(eval_params, seed=validation_seed))

    out = args.out or (MODELOS / f"{args.algo}_drone_fleet_v2.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    callback_dir = results_dir / "validacao"
    callback_dir.mkdir(parents=True, exist_ok=True)
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

    if args.algo == "dqn":
        model_cls = DQN
        model = DQN(
            "MlpPolicy", env,
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
    elif args.algo == "ppo":
        model_cls = PPO
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            seed=args.seed,
        )
    else:
        try:
            from sb3_contrib import QRDQN
        except ImportError as exc:
            raise SystemExit("QR-DQN requer sb3-contrib. Instale com: pip install -r requirements.txt") from exc
        model_cls = QRDQN
        model = QRDQN(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=128,
            learning_starts=10_000,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=2_000,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            gamma=0.995,
            policy_kwargs={"net_arch": [128, 128], "n_quantiles": 50},
            verbose=1,
            seed=args.seed,
        )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    dt = time.perf_counter() - t0

    last_out = out.with_name(f"{out.stem}_last{out.suffix or '.zip'}")
    model.save(last_out)
    best_model_path = callback_dir / "best_model.zip"
    if best_model_path.exists():
        best_model = model_cls.load(best_model_path, env=env)
        best_model.save(out)
        print(f">> Melhor checkpoint de validação salvo em {out}; último checkpoint em {last_out}")
    else:
        shutil.copy2(last_out, out)
        print(f">> Validação não produziu checkpoint; modelo final salvo em {out}")
    eval_env.close()
    env.close()
    print(f">> Treino concluído ({dt/60:.1f} min, "
          f"{args.timesteps/dt:.0f} passos/s de ambiente)")

    # extrai recompensa por episódio direto do Monitor (não sofre truncamento)
    csv_out = results_dir / f"treino_{args.algo}_seed{args.seed}.csv"
    rewards = list(env.episode_returns)
    lengths = list(env.episode_lengths)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "passos", "recompensa_total"])
        for i, (r, l) in enumerate(zip(rewards, lengths), start=1):
            w.writerow([i, int(l), float(r)])
    print(f">> Log de episódios ({len(rewards)}): {csv_out}")

    if rewards:
        n_ult = min(100, len(rewards))
        print(f">> Recompensa média (últimos {n_ult} eps): "
              f"{float(sum(rewards[-n_ult:])) / n_ult:.1f}")


if __name__ == "__main__":
    main()
