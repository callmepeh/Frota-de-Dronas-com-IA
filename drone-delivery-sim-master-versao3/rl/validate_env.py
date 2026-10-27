# -*- coding: utf-8 -*-
"""
Validação do ambiente Python contra os CSVs já gerados pelo simulador Java.

Roda o ambiente com a frota TRAVADA em 60 drones (equivalente ao modo STATIC
usado nas execuções que produziram ad_mmt_log.csv / system_state_log.csv) e
compara as estatísticas resultantes (MMT por entrega, fila, frota) com os
valores reais dos CSVs. O Java não é executado: apenas seus resultados são
reutilizados, como pedido.

Uso:
  python3 validate_env.py            # compara com ambos os CSVs
  python3 validate_env.py --seeds 10
Saída: rl_resultados/validacao_ambiente.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor

from drone_fleet_env import DroneFleetEnv, load_params

HERE = Path(__file__).resolve().parent
RESULTADOS = HERE / "rl_resultados"
PROJECT = HERE.parent


def run_static_episode(P, seed):
    """Episódio com frota fixa em 60 (ações não alteram a frota)."""
    env = Monitor(DroneFleetEnv(P, seed=seed, fleet_min=60, fleet_max=60))
    obs, _ = env.reset(seed=seed)
    done = False
    deliveries = []
    gen = drop = 0
    fila_max = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(3)  # ação 0
        done = terminated or truncated
        deliveries.extend(info["deliveries_list"])
        gen = info["generated"]
        drop = info["dropped"]
        fila_max = max(fila_max, info["queue"])
    m = np.array([t / 60_000 for _, t in deliveries])
    return m, gen, drop, fila_max


def est(m: np.ndarray, gen: int, drop: int, fila_max: int) -> dict:
    return {
        "Entregas": int(len(m)),
        "Gerados": gen,
        "Dropados": drop,
        "MMT_med(min)": round(float(np.mean(m)), 1) if len(m) else math.nan,
        "MMT_mediano(min)": round(float(np.median(m)), 1) if len(m) else math.nan,
        "MMT_P95(min)": round(float(np.percentile(m, 95)), 1) if len(m) else math.nan,
        "MMT_max(min)": round(float(np.max(m)), 1) if len(m) else math.nan,
        "SLA_ok(%)": round(100 * float(np.mean(m <= 80)), 1) if len(m) else math.nan,
        "Fila_max": fila_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 5])
    args = ap.parse_args()

    RESULTADOS.mkdir(exist_ok=True)
    P = load_params(simulation_time_ms=72_000_000)

    ambientes = []
    for seed in args.seeds:
        ambientes.append(("Ambiente(frota=60)", est(*run_static_episode(P, seed))))

    # estatísticas reais dos CSVs do projeto
    refs = []
    for rotulo, f in [("Java", "ad_mmt_log.csv"),
                      ("Java-adaptativo", "ad_mmt_log_dynamic_run.csv")]:
        p = PROJECT / f
        if not p.is_file():
            continue
        rows = [r for r in csv.DictReader(open(p, encoding="utf-8"), delimiter=";")
                if r.get("MeanMissionTime")]
        m = np.array([float(r["MeanMissionTime"]) / 60_000 for r in rows])
        state = PROJECT / ("system_state_log.csv" if "dynamic" not in f
                           else "system_state_log_dynamic_run.csv")
        fila_max = 0
        state_rows = (csv.DictReader(open(state, encoding="utf-8"), delimiter=";")
                      if state.is_file() else [])
        state_rows = list(state_rows)
        for r in state_rows:
            try:
                fila_max = max(fila_max, int(float(r["QueueSize"])))
            except (ValueError, KeyError):
                pass
        gen = len(rows) + (int(float(state_rows[-1]["QueueSize"])) if state_rows else 0)
        refs.append((rotulo, est(m, gen, 0, fila_max)))

    linhas = refs + ambientes
    colunas = list(linhas[0][1].keys())
    with open(RESULTADOS / "validacao_ambiente.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Origem"] + colunas)
        for rotulo, st in linhas:
            w.writerow([rotulo] + [st[c] for c in colunas])

    print("== Validação do ambiente Python × resultados reais do simulador Java ==\n")
    hdr = f"{'Origem':22s}" + "".join(f"{c:>16s}" for c in colunas)
    print(hdr)
    for rotulo, st in linhas:
        print(f"{rotulo:22s}" + "".join(f"{st[c]!s:>16s}" for c in colunas))
    print(f"\n>> CSV salvo em {RESULTADOS / 'validacao_ambiente.csv'}")


if __name__ == "__main__":
    main()
