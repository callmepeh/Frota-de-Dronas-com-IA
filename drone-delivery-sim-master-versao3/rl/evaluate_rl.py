# -*- coding: utf-8 -*-
"""
Avaliação comparativa: política DQN/PPO/QR-DQN vs Busca Binária analítica (DT)
vs frota estática de 60 drones, nos mesmos cenários estocásticos (mesmas seeds).

Métricas por política: entregas, MMT médio/mediano/P95 (min), SLA (80 min) %,
fila média/máx, frota média ponderada no tempo, minutos de voo alugados,
energia (Wh), deliveries/Wh, nº de aluguéis/devoluções e tempo de CPU médio
gasto para decisão (robô: inferência da rede vs ~6 avaliações da busca binária).

Uso:
  python3 evaluate_rl.py --algo qrdqn --model modelos/qrdqn_drone_fleet_v2.zip
  python3 evaluate_rl.py --algo ppo
Saída: rl_resultados/resumo_comparacao_rl.csv, serie_<politica>_seed<k>.csv
e grafico_comparacao_rl.png
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from drone_fleet_env import DroneFleetEnv, load_params
from dt_baseline import BinarySearchDT
from stable_baselines3 import DQN, PPO

HERE = Path(__file__).resolve().parent
RESULTADOS = HERE / "rl_resultados"
SLA_MIN = 80.0
METRICAS_CI = ("MMT_med(min)", "MMT_P95(min)", "SLA_ok(%)", "Frota_media",
               "Q_fila_media", "Entregas_por_Wh", "Decisao_CPU_med(ms)")


def t_critical_95(n: int) -> float:
    """Critical value for a two-sided 95% Student-t CI (Cornish-Fisher)."""
    if n < 2:
        return float("nan")
    df = n - 1
    z = 1.959963984540054
    z2 = z * z
    return (z + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def decide_rl(model, obs, core, env):
    action, _ = model.predict(obs, deterministic=True)
    return int(action), None


def decide_marl(model, obs, core, env):
    return model.predict(env.multiagent_observation(), deterministic=True)[0], None


def decide_transformer(model, obs, core, env):
    return model.predict(obs, deterministic=True)[0], None


def decide_dt(obs, core, env):
    dt = env._dt_baseline  # instalado pelo runner
    dt.observe(core.proc_t)
    desired = dt.decide(core, core.proc_t)
    committed = core.committed()
    delta = desired - committed
    # converte para a ação discreta mais próxima (mesmo espaço de ações do RL)
    deltas = (-5, -3, -1, 0, +1, +3, +5)
    action = int(np.argmin([abs(d - delta) for d in deltas]) if abs(delta) <= 5
                 else deltas.index(0))
    if abs(delta) > 5:
        action = deltas.index(5) if desired > committed else deltas.index(-5)
    return action, desired


def decide_static60(obs, core, env):
    committed = core.committed()
    delta = env.P.initial_fleet - committed
    deltas = (-5, -3, -1, 0, +1, +3, +5)
    if abs(delta) <= 5:
        return deltas.index(delta), committed + delta
    action = deltas.index(5) if delta > 0 else deltas.index(-5)
    return action, None


def run_episode(policy_name, decide_fn, P, seed, model=None, policy_object=None):
    env = DroneFleetEnv(P, seed=seed)
    obs, info = env.reset(seed=seed)
    if policy_name == "BuscaBinariaDT":
        env._dt_baseline = BinarySearchDT(P)
    if policy_object is not None:
        policy_object.reset_episode()

    decision_ms_total = 0.0
    steps = 0
    rows = []
    done = False
    deliveries = []
    while not done:
        t0 = time.perf_counter()
        action, hint = decide_fn(obs, env.core, env)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        decision_ms_total += dt_ms
        if policy_name == "MARL":
            obs, reward, terminated, truncated, info = env.step_multiagent(*action)
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        if policy_object is not None:
            policy_object.observe_transition(reward)
        steps += 1
        done = terminated or truncated
        deliveries.extend(info["deliveries_list"])
        rows.append({
            "Time(ms)": info["t_ms"],
            "FrotaOp": info["fleet_op"],
            "Ociosos": info["idle"],
            "Ativos": info["active"],
            "AlugadosOpTw": info["rented_op"],
            "Fila": info["queue"],
            "MMTEwma(min)": round(info["mmt_ewma_min"], 3),
            "ViolSLA_Janela(%)": "" if info["window_sla_violation"] is None
                                  else round(100 * info["window_sla_violation"], 2),
            "Alugueis": info["rentals"],
            "Devolucoes": info["returns"],
            "Recompensa": round(info["reward"] if info["reward"] is not None else 0.0, 3),
            "Decisao_CPU(ms)": round(dt_ms, 4),
        })

    e_info = env._info()
    dur_h = e_info["t_ms"] / 3_600_000
    m_list = np.array([m / 60_000 for _, m in deliveries]) if deliveries else np.array([np.nan])
    dentro = np.sum(m_list <= SLA_MIN) / max(1, len(m_list))
    wh = e_info["voo_wh"] + e_info["pouso_wh"]
    resumo = {
        "Politica": policy_name,
        "Seed": seed,
        "Entregas": len(deliveries),
        "MMT_med(min)": float(np.mean(m_list)),
        "MMT_mediano(min)": float(np.median(m_list)),
        "MMT_P95(min)": float(np.percentile(m_list, 95)),
        "SLA_ok(%)": round(100 * dentro, 2),
        "Violacoes": int(np.sum(m_list > SLA_MIN)),
        "Q_fila_media": float(np.mean([r["Fila"] for r in rows])),
        "Q_fila_max": int(max(r["Fila"] for r in rows)),
        "Frota_media": round(e_info["fleet_tw"], 2),
        "Voo_drone_h": round(e_info["active_tw"] * dur_h, 1),
        "Aluguel_drone_h": round(e_info["rented_tw"] * dur_h, 1),
        "E_Consu(Wh)": int(wh),
        "Entregas_por_Wh": round(len(deliveries) / wh, 4) if wh > 0 else np.nan,
        "Alugueis": e_info["rentals"],
        "Devolucoes": e_info["returns"],
        "Dropados": e_info["dropped"],
        "Recompensa_media": round(float(np.mean([r["Recompensa"] for r in rows])), 2),
        "Decisao_CPU_med(ms)": round(decision_ms_total / steps, 4),
    }
    return resumo, rows


def grafico(policies_data: dict[str, list[dict]], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                         'figure.figsize': (11, 9)})

    fig, axes = plt.subplots(3, 1, sharex=True)
    cores = {"RL": "tab:green", "MARL": "tab:purple",
             "DecisionTransformer": "tab:orange",
             "BuscaBinariaDT": "tab:blue", "Estatico60": "tab:red"}
    rotulos = {"RL": "Política RL", "MARL": "Política multiagente",
               "DecisionTransformer": "Decision Transformer",
               "BuscaBinariaDT": "Busca Binária (DT)",
               "Estatico60": "Frota Estática 60"}

    # (a) frota
    for name, seeds_rows in policies_data.items():
        frotas = [np.array([r["FrotaOp"] for r in rows]) for rows in seeds_rows]
        n = min(len(f) for f in frotas)
        curva = np.mean([f[:n] for f in frotas], axis=0)
        t = np.arange(n) * 5 / 60  # passos de 5 min → horas
        axes[0].plot(t, curva, label=rotulos[name], color=cores.get(name), lw=1.5)
    axes[0].set_ylabel("Frota operacional")
    axes[0].axhline(60, color="gray", ls=":", lw=1)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Comparação RL × Busca Binária (cenários idênticos por seed)")

    # (b) MMT (EWMA por passo de decisão, média entre seeds)
    for name, seeds_rows in policies_data.items():
        mmts = [np.array([r["MMTEwma(min)"] for r in rows]) for rows in seeds_rows]
        n = min(len(m) for m in mmts)
        curva = np.mean([m[:n] for m in mmts], axis=0)
        axes[1].plot(np.arange(n) * 5 / 60, curva, label=rotulos[name],
                     color=cores.get(name), lw=1.5)
    axes[1].axhline(40, color="green", ls=":", lw=1, label="Alvo DT (40 min)")
    axes[1].axhline(80, color="red", ls=":", lw=1, label="SLA (80 min)")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("MMT EWMA (min)")

    # (c) fila
    for name, seeds_rows in policies_data.items():
        filas = [np.array([r["Fila"] for r in rows]) for rows in seeds_rows]
        n = min(len(f) for f in filas)
        curva = np.mean([f[:n] for f in filas], axis=0)
        axes[2].plot(np.arange(n) * 5 / 60, curva, label=rotulos[name],
                     color=cores.get(name), lw=1.5)
    axes[2].set_ylabel("Fila de pedidos")
    axes[2].set_xlabel("Tempo de simulação (h)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f">> Gráfico salvo em {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Avaliação comparativa RL vs DT")
    ap.add_argument("--model", type=Path, default=None,
                    help="caminho do checkpoint compatível; se omitido, procura modelos *_v2 em modelos/")
    ap.add_argument("--algo", choices=["dqn", "ppo", "qrdqn", "marl", "dt"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="diretório exclusivo (padrão: rl_resultados/avaliacao_<modelo>)")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        cand = [HERE / "modelos" / "qrdqn_drone_fleet_v2.zip",
                HERE / "modelos" / "marl_coop_drone_fleet_v1.pt",
                HERE / "modelos" / "decision_transformer_drone_fleet_v1.pt",
                HERE / "modelos" / "dqn_drone_fleet_v2.zip",
                HERE / "modelos" / "ppo_drone_fleet_v2.zip"]
        found = [c for c in cand if c.exists()]
        if not found:
            raise SystemExit("Checkpoint v2 não encontrado. Rode train_rl.py ou informe --model; "
                             "checkpoints antigos usam estado incompatível de 9 dimensões.")
        model_path = found[0]
    out_dir = args.out_dir or (RESULTADOS / f"avaliacao_{model_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    algo = args.algo or ("marl" if "marl" in model_path.name.lower()
                         else "dt" if "transformer" in model_path.name.lower() or model_path.name.lower().startswith("dt_")
                         else "qrdqn" if "qrdqn" in model_path.name.lower()
                         else "dqn" if "dqn" in model_path.name.lower() else "ppo")
    policy_object = None
    if algo == "marl":
        from marl_train import CooperativeFleetPolicy
        model = CooperativeFleetPolicy.load(model_path)
        model_cls = None
    elif algo == "dt":
        from decision_transformer import DecisionTransformerPolicy
        model = DecisionTransformerPolicy.load(model_path)
        model_cls = None
        policy_object = model
    elif algo == "qrdqn":
        try:
            from sb3_contrib import QRDQN
        except ImportError as exc:
            raise SystemExit("Avaliar QR-DQN requer sb3-contrib: pip install -r requirements.txt") from exc
        model_cls = QRDQN
    else:
        model_cls = DQN if algo == "dqn" else PPO
    if model_cls is not None:
        model = model_cls.load(model_path)

    P = load_params()
    rl_name = "MARL" if algo == "marl" else "DecisionTransformer" if algo == "dt" else "RL"
    rl_decider = decide_marl if algo == "marl" else decide_transformer if algo == "dt" else decide_rl
    policies = [(rl_name, lambda obs, core, env: rl_decider(model, obs, core, env)),
                ("BuscaBinariaDT", decide_dt)]
    if not args.skip_static:
        policies.append(("Estatico60", decide_static60))

    resumos = []
    policies_data: dict[str, list] = {}
    for name, fn in policies:
        rows_all = []
        for seed in args.seeds:
            resumo, rows = run_episode(name, fn, P, seed,
                                       policy_object=policy_object if name == rl_name else None)
            if name == rl_name:
                resumo["Politica"] = rl_name
            resumos.append(resumo)
            rows_all.append(rows)
            # série CSV
            serie = out_dir / f"serie_{name}_seed{seed}.csv"
            with open(serie, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  [{name} seed={seed}] entregas={resumo['Entregas']} "
                  f"MMT={resumo['MMT_med(min)']:.1f}min SLA={resumo['SLA_ok(%)']}% "
                  f"frota={resumo['Frota_media']} cpu={resumo['Decisao_CPU_med(ms)']}ms")
        policies_data[name] = rows_all

    import pandas as pd
    df = pd.DataFrame(resumos)
    resumo_agreg = df.groupby("Politica").mean(numeric_only=True).drop(columns=["Seed"])
    resumo_agreg.to_csv(out_dir / "resumo_comparacao_rl.csv", sep=";", decimal=",")
    df.to_csv(out_dir / "resultados_por_seed.csv", sep=";", decimal=",", index=False)

    # Student-t CIs summarize seed uncertainty; paired contrasts use the same
    # evaluation seeds (common random numbers).
    stat_rows = []
    for policy, group in df.groupby("Politica"):
        for metric in METRICAS_CI:
            vals = group[metric].dropna().astype(float)
            n = len(vals)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if n > 1 else 0.0
            half = t_critical_95(n) * sd / np.sqrt(n) if n > 1 else float("nan")
            stat_rows.append({"Politica": policy, "Metrica": metric, "n": n,
                              "Media": mean, "DP": sd,
                              "IC95_inf_aprox": mean - half,
                              "IC95_sup_aprox": mean + half})
    import pandas as pd
    pd.DataFrame(stat_rows).to_csv(out_dir / "estatisticas_ic95.csv", sep=";", decimal=",", index=False)

    paired_rows = []
    for metric in ("MMT_med(min)", "MMT_P95(min)", "SLA_ok(%)", "Frota_media", "Q_fila_media"):
        wide = df.pivot(index="Seed", columns="Politica", values=metric)
        for baseline in ("BuscaBinariaDT", "Estatico60"):
            if rl_name not in wide or baseline not in wide:
                continue
            delta = (wide[rl_name] - wide[baseline]).dropna().astype(float)
            n = len(delta)
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1)) if n > 1 else 0.0
            half = t_critical_95(n) * sd / np.sqrt(n) if n > 1 else float("nan")
            paired_rows.append({"Metrica": metric, "Comparacao": f"{rl_name} - {baseline}",
                                "n_seeds_pareadas": n, "Diferenca_media": mean,
                                "DP_diferencas": sd, "IC95_inf_aprox": mean - half,
                                "IC95_sup_aprox": mean + half})
    pd.DataFrame(paired_rows).to_csv(out_dir / "diferencas_pareadas.csv", sep=";", decimal=",", index=False)
    print("\n== Resumo agregado (média das seeds) ==")
    print(resumo_agreg.round(2).to_string())
    print(f"\n>> Resultados salvos em {out_dir}")

    grafico(policies_data, out_dir / "grafico_comparacao_rl.png")


if __name__ == "__main__":
    main()
