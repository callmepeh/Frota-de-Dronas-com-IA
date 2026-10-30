# -*- coding: utf-8 -*-
"""
Avaliação da política residual DQN × Busca Binária DT × Estático 60,
nos mesmos cenários (mesmas seeds, mesma regra de chegada, mesmo ambiente
DroneFleetEnv v3+residual). Todos os DT/estático aplicam o alvo exato via
step_target (fidelidade à runDigitalTwin; no protocolo antigo a ação era
arredondada para o delta discreto — anotado na análise).

Saída em --out-dir: resultados_por_seed.csv, resumo_agregado.csv,
estatisticas_ic95.csv, diferencas_pareadas.csv, veredito_digital_twin.json,
série por seed e gráfico comparativo.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from drone_fleet_env import DroneFleetEnv, load_params
from dt_baseline import BinarySearchDT
from evaluate_rl import t_critical_95

HERE = Path(__file__).resolve().parent
SLA_MIN = 80.0
METRICAS = ("MMT_med(min)", "MMT_mediano(min)", "MMT_P95(min)", "SLA_ok(%)",
            "SLA_janelas(%)", "Q_fila_media", "Q_fila_max", "Idade_max_fila(min)",
            "Frota_media", "Frota_max", "Aluguel_drone_h", "Alugueis",
            "Devolucoes", "E_Consu(Wh)", "Entregas_por_Wh",
            "Decisao_CPU_med(ms)", "Dropados", "Mudancas_frota",
            "Delta_abs_medio")


def run_episode(name, decider, P, seed, model=None):
    """decider(self_env) chamado a cada decisão; retorna (resumo, rows)."""
    env = DroneFleetEnv(P, seed=seed, obs_version="v3", action_mode="residual")
    env.reset(seed=seed)
    dt_baseline = BinarySearchDT(P) if name == "BuscaBinariaDT" else None
    t_start = time.perf_counter()

    decision_ms = 0.0
    steps = 0
    rows = []
    deliveries = []
    fila_max = 0
    idade_max = 0.0
    frotas = []
    mudancas = 0
    delta_abs = 0.0
    prev_committed = env.core.committed()
    done = False
    while not done:
        t0 = time.perf_counter()
        if name == "ResidualDQN":
            action, _ = model.predict(env._build_obs(), deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
        elif name == "BuscaBinariaDT":
            dt_baseline.observe(env.core.proc_t)
            desired = dt_baseline.decide(env.core, env.core.proc_t)
            obs, r, term, trunc, info = env.step_target(desired)
        else:  # Estatico60
            obs, r, term, trunc, info = env.step_target(P.initial_fleet)
        decision_ms += (time.perf_counter() - t0) * 1000.0

        if info["committed"] != prev_committed:
            mudancas += 1
            delta_abs += abs(info["committed"] - prev_committed)
        prev_committed = info["committed"]
        frotas.append(info["fleet_op"])
        fila_max = max(fila_max, info["queue"])
        if env.core.queue:
            idade_max = max(idade_max, (env.core.proc_t - min(env.core.queue)) / 60_000.0)
        deliveries.extend(info["deliveries_list"])
        rows.append({"Time(ms)": info["t_ms"], "FrotaOp": info["fleet_op"],
                     "Fila": info["queue"], "MMTEwma(min)": round(info["mmt_ewma_min"], 3),
                     "Recompensa": round(r, 3), "Alugueis": info["rentals"]})
        steps += 1
        done = term or trunc

    e_info = env._info()
    dur_h = e_info["t_ms"] / 3_600_000
    m = np.array([mm / 60_000 for _, mm in deliveries]) if deliveries else np.array([np.nan])
    dentro = float(np.mean(m <= SLA_MIN)) if len(m) else 0.0
    # SLA por janela de decisão (5 min)
    viol_janelas = e_info["window_sla_violation"]
    wh = e_info["voo_wh"] + e_info["pouso_wh"]
    resumo = {
        "Politica": name,
        "Seed": seed,
        "Entregas": len(deliveries),
        "MMT_med(min)": float(np.mean(m)),
        "MMT_mediano(min)": float(np.median(m)),
        "MMT_P95(min)": float(np.percentile(m, 95)),
        "SLA_ok(%)": round(100 * dentro, 2),
        "SLA_janelas(%)": round(100 * (1 - (viol_janelas or 0.0)), 2)
                          if viol_janelas is not None else np.nan,
        "Violacoes": int(np.sum(m > SLA_MIN)),
        "Q_fila_media": float(np.mean([r["Fila"] for r in rows])),
        "Q_fila_max": int(fila_max),
        "Idade_max_fila(min)": round(idade_max, 1),
        "Frota_media": round(e_info["fleet_tw"], 2),
        "Frota_max": int(max(frotas)),
        "Voo_drone_h": round(e_info["active_tw"] * dur_h, 1),
        "Aluguel_drone_h": round(e_info["rented_tw"] * dur_h, 1),
        "Alugueis": e_info["rentals"],
        "Devolucoes": e_info["returns"],
        "Dropados": e_info["dropped"],
        "E_Consu(Wh)": int(wh),
        "Entregas_por_Wh": round(len(deliveries) / wh, 5) if wh > 0 else np.nan,
        "Recompensa_media": round(float(np.mean([r["Recompensa"] for r in rows])), 2),
        "Decisao_CPU_med(ms)": round(decision_ms / steps, 4),
        "Mudancas_frota": mudancas,
        "Delta_abs_medio": round(delta_abs / steps, 3),
        "Duracao_decisao_total(s)": round(time.perf_counter() - t_start, 1),
    }
    return resumo, rows


def grafico(data: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cores = {"ResidualDQN": "tab:green", "BuscaBinariaDT": "tab:blue",
             "Estatico60": "tab:red"}
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for name, rows_all in data.items():
        n = min(len(rows) for rows in rows_all)
        frota = np.mean([np.array([r["FrotaOp"] for r in rows])[:n] for rows in rows_all], axis=0)
        fila = np.mean([np.array([r["Fila"] for r in rows])[:n] for rows in rows_all], axis=0)
        t = np.arange(n) * 5 / 60
        axes[0].plot(t, frota, label=name, color=cores.get(name), lw=1.4)
        axes[1].plot(t, fila, label=name, color=cores.get(name), lw=1.4)
    axes[0].set_ylabel("Frota operacional")
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Fila de pedidos")
    axes[1].set_xlabel("Tempo de simulação (h)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def veredito(paired: list[dict], out_path: Path, n_seeds: int):
    """Critério da pesquisa: IA só supera DT se SLA >=, MMT <=, P95 <=,
    fila não significativamente pior e menos frota/aluguel/energia."""
    d = {(r["Metrica"]): r for r in paired if r["Baseline"] == "BuscaBinariaDT"}
    sla_ok = d["SLA_ok(%)"]["Diferenca_media"] >= 0
    mmt_ok = d["MMT_med(min)"]["Diferenca_media"] <= 0
    p95_ok = d["MMT_P95(min)"]["Diferenca_media"] <= 0
    fila_ic = d["Q_fila_media"]
    fila_nao_pior = fila_ic["IC95_sup_aprox"] <= 0
    recurso_ok = any(d[k]["Diferenca_media"] < 0
                     for k in ("Frota_media", "Aluguel_drone_h", "E_Consu(Wh)")
                     if k in d)
    se_superou = sla_ok and mmt_ok and p95_ok and fila_nao_pior and recurso_ok
    out = {
        "criterio": "SLA>=DT e MMT<=DT e P95<=DT e fila não significativamente pior (IC pareado) e menor frota/aluguel/energia",
        "n_seeds_pareadas": n_seeds,
        "sla_ok": bool(sla_ok), "mmt_ok": bool(mmt_ok), "p95_ok": bool(p95_ok),
        "fila_nao_pior": bool(fila_nao_pior), "reducao_recurso": bool(recurso_ok),
        "ia_superou_dt": bool(se_superou),
        "ic_detalhes": {k: [r["Diferenca_media"], r["IC95_inf_aprox"], r["IC95_sup_aprox"]]
                        for k, r in d.items()},
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n== Veredito contra Digital Twin ==")
    print(json.dumps({k: out[k] for k in ("n_seeds_pareadas", "sla_ok", "mmt_ok",
                                          "p95_ok", "fila_nao_pior", "reducao_recurso",
                                          "ia_superou_dt")}, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="Avaliação política residual")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from stable_baselines3 import DQN
    model = DQN.load(args.model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    P = load_params()
    resumos, dados = [], {}
    for name in ("ResidualDQN", "BuscaBinariaDT", "Estatico60"):
        rows_all = []
        for seed in args.seeds:
            resumo, rows = run_episode(name, None, P, seed, model=model)
            resumos.append(resumo)
            rows_all.append(rows)
            print(f"  [{name} seed={seed}] MMT={resumo['MMT_med(min)']:.1f} "
                  f"SLA={resumo['SLA_ok(%)']}% frota={resumo['Frota_media']} "
                  f"aluguel={resumo['Aluguel_drone_h']}")
        dados[name] = rows_all

    import pandas as pd
    df = pd.DataFrame(resumos)
    df.to_csv(args.out_dir / "resultados_por_seed.csv", sep=";", decimal=",", index=False)
    agreg = df.groupby("Politica").mean(numeric_only=True).drop(columns=["Seed"])
    agreg.to_csv(args.out_dir / "resumo_agregado.csv", sep=";", decimal=",")

    stat_rows = []
    for policy, group in df.groupby("Politica"):
        for metric in METRICAS:
            vals = group[metric].dropna().astype(float)
            n = len(vals)
            mean, sd = float(vals.mean()), (float(vals.std(ddof=1)) if n > 1 else 0.0)
            half = t_critical_95(n) * sd / np.sqrt(n) if n > 1 else float("nan")
            stat_rows.append({"Politica": policy, "Metrica": metric, "n": n,
                              "Media": mean, "DP": sd,
                              "IC95_inf_aprox": mean - half, "IC95_sup_aprox": mean + half})
    pd.DataFrame(stat_rows).to_csv(args.out_dir / "estatisticas_ic95.csv",
                                   sep=";", decimal=",", index=False)

    paired_rows = []
    wide_tables = {}
    for metric in ("MMT_med(min)", "MMT_mediano(min)", "MMT_P95(min)", "SLA_ok(%)",
                   "Q_fila_media", "Frota_media", "Aluguel_drone_h", "E_Consu(Wh)"):
        wide = df.pivot(index="Seed", columns="Politica", values=metric)
        wide_tables[metric] = wide
        if "ResidualDQN" not in wide or "BuscaBinariaDT" not in wide:
            continue
        delta = (wide["ResidualDQN"] - wide["BuscaBinariaDT"]).dropna().astype(float)
        n = len(delta)
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1)) if n > 1 else 0.0
        half = t_critical_95(n) * sd / np.sqrt(n) if n > 1 else float("nan")
        paired_rows.append({"Metrica": metric, "Baseline": "BuscaBinariaDT",
                            "Comparacao": "IA - DT", "n_seeds_pareadas": n,
                            "Diferenca_media": mean, "DP_diferencas": sd,
                            "IC95_inf_aprox": mean - half, "IC95_sup_aprox": mean + half})
    pd.DataFrame(paired_rows).to_csv(args.out_dir / "diferencas_pareadas.csv",
                                     sep=";", decimal=",", index=False)
    veredito(paired_rows, args.out_dir / "veredito_digital_twin.json", len(args.seeds))

    # séries por seed da IA
    for j, rows in enumerate(dados["ResidualDQN"], 1):
        serie = args.out_dir / f"serie_ResidualDQN_seed{args.seeds[j-1]}.csv"
        with open(serie, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    grafico(dados, args.out_dir / "grafico_comparacao_residual.png")
    print("\n== Resumo agregado (média das seeds) ==")
    print(agreg.round(2).to_string())
    print(f"\n>> Resultados salvos em {args.out_dir}")


if __name__ == "__main__":
    main()
