# -*- coding: utf-8 -*-
"""Executa treino multi-seed e avaliação em uma bateria comum de teste.

Exemplo:
  python3 experiment_rl.py --algo dqn --timesteps 500000 \
      --train-seeds 42 43 44 45 46 --eval-seeds 1 2 ... 30
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
MODELOS = HERE / "modelos"
RESULTADOS = HERE / "rl_resultados" / "experimentos"


def main():
    ap = argparse.ArgumentParser(description="Experimento RL reproduzível multi-seed")
    ap.add_argument("--algo", choices=["dqn", "ppo", "qrdqn", "marl", "dt", "all"], default="dqn")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--name", default=None, help="nome da pasta do experimento")
    args = ap.parse_args()

    if args.algo == "all":
        name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithms = ["dqn", "ppo", "qrdqn", "marl", "dt"]
        summaries = []
        for algorithm in algorithms:
            command = [sys.executable, str(HERE / "experiment_rl.py"),
                       "--algo", algorithm, "--timesteps", str(args.timesteps),
                       "--train-seeds", *map(str, args.train_seeds),
                       "--eval-seeds", *map(str, args.eval_seeds),
                       "--eval-freq", str(args.eval_freq),
                       "--eval-episodes", str(args.eval_episodes),
                       "--name", f"{name}_{algorithm}"]
            if args.skip_static:
                command.append("--skip-static")
            try:
                subprocess.run(command, cwd=HERE, check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"Execução interrompida no algoritmo {algorithm} "
                    f"(código {exc.returncode}). Os resultados anteriores foram preservados; "
                    "corrija a etapa indicada e reinicie com outro --name."
                ) from None
            summary_path = RESULTADOS / algorithm / f"{name}_{algorithm}" / "resumo_por_modelo.csv"
            if not summary_path.is_file():
                raise SystemExit(
                    f"O algoritmo {algorithm} terminou sem resumo esperado: {summary_path}"
                )
            frame = pd.read_csv(summary_path, sep=";", decimal=",")
            frame.insert(1, "Algoritmo", algorithm)
            summaries.append(frame)
        comparison_dir = RESULTADOS / "comparacao_all" / name
        comparison_dir.mkdir(parents=True, exist_ok=False)
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(comparison_dir / "resumo_por_modelo.csv", sep=";", decimal=",", index=False)
        metrics = combined.select_dtypes(include="number").columns.difference(["Seed_treino"])
        comparison = combined.groupby(["Algoritmo", "Politica"])[list(metrics)].agg(["mean", "std"])
        comparison.to_csv(comparison_dir / "resumo_comparativo.csv", sep=";", decimal=",")
        print(f"\nComparação completa salva em: {comparison_dir}")
        print(comparison.round(3).to_string())
        return

    validation_seeds = {seed + 100_000 for seed in args.train_seeds}
    if validation_seeds.intersection(args.eval_seeds):
        ap.error("as seeds de avaliação devem ser diferentes das seeds base de validação (treino + 100000)")
    if len(set(args.train_seeds)) != len(args.train_seeds):
        ap.error("--train-seeds não pode conter valores repetidos")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        ap.error("--eval-seeds não pode conter valores repetidos")

    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTADOS / args.algo / name
    model_dir = run_dir / "modelos"
    model_dir.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    seed_frames = []

    for train_seed in args.train_seeds:
        model_suffix = ".pt" if args.algo in ("marl", "dt") else ".zip"
        model_path = model_dir / f"{args.algo}_seed{train_seed}{model_suffix}"
        train_dir = run_dir / f"treino_seed_{train_seed}" / "treino"
        if args.algo == "marl":
            train_cmd = [
                sys.executable, str(HERE / "marl_train.py"),
                "--timesteps", str(args.timesteps),
                "--seed", str(train_seed),
                "--validation-seed", str(train_seed + 100_000),
                "--eval-freq", str(args.eval_freq),
                "--eval-episodes", str(args.eval_episodes),
                "--out", str(model_path),
                "--results-dir", str(train_dir),
            ]
        elif args.algo == "dt":
            train_cmd = [
                sys.executable, str(HERE / "train_decision_transformer.py"),
                "--seed", str(train_seed),
                "--out", str(model_path),
                "--dataset-out", str(train_dir / "dataset_offline.npz"),
            ]
        else:
            train_cmd = [
                sys.executable, str(HERE / "train_rl.py"),
                "--algo", args.algo,
                "--timesteps", str(args.timesteps),
                "--seed", str(train_seed),
                "--validation-seed", str(train_seed + 100_000),
                "--eval-freq", str(args.eval_freq),
                "--eval-episodes", str(args.eval_episodes),
                "--out", str(model_path),
                "--results-dir", str(train_dir),
            ]
        subprocess.run(train_cmd, cwd=HERE, check=True)
        if not model_path.is_file():
            raise SystemExit(
                f"Treino de {args.algo} seed {train_seed} terminou sem salvar checkpoint: "
                f"{model_path}. A avaliação foi cancelada."
            )

        eval_dir = run_dir / f"treino_seed_{train_seed}" / "teste"
        cmd = [
            sys.executable, str(HERE / "evaluate_rl.py"),
            "--algo", args.algo,
            "--model", str(model_path),
            "--seeds", *[str(s) for s in args.eval_seeds],
            "--out-dir", str(eval_dir),
        ]
        if args.skip_static:
            cmd.append("--skip-static")
        subprocess.run(cmd, cwd=HERE, check=True)

        summary = pd.read_csv(eval_dir / "resumo_comparacao_rl.csv", sep=";", decimal=",")
        summary.insert(0, "Seed_treino", train_seed)
        summary_rows.append(summary)
        per_seed = pd.read_csv(eval_dir / "resultados_por_seed.csv", sep=";", decimal=",")
        per_seed.insert(0, "Seed_treino", train_seed)
        seed_frames.append(per_seed)

    by_model = pd.concat(summary_rows, ignore_index=True)
    by_model.to_csv(run_dir / "resumo_por_modelo.csv", sep=";", decimal=",", index=False)
    per_seed_all = pd.concat(seed_frames, ignore_index=True)
    per_seed_all.to_csv(run_dir / "resultados_por_seed_treino.csv", sep=";", decimal=",", index=False)

    numeric = by_model.select_dtypes(include="number").columns.difference(["Seed_treino"])
    between = by_model.groupby("Politica")[list(numeric)].agg(["mean", "std"])
    between.to_csv(run_dir / "resumo_entre_treinos.csv", sep=";", decimal=",")

    print(f"\nExperimento completo: {run_dir}")
    print("Resumo entre execuções de treino (média e desvio padrão dos modelos):")
    print(between.round(3).to_string())


if __name__ == "__main__":
    main()
