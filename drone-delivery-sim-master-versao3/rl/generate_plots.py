# -*- coding: utf-8 -*-
"""
Gera os gráficos de TREINAMENTO e de RESULTADO DO MODELO a partir dos CSVs
produzidos pela skill RL (rl_resultados/).

Gráficos salvos:
  rl_resultados/grafico_treino_dqn.png       — curva de aprendizado (recompensa por episódio)
  rl_resultados/grafico_resultado_modelo.png — painéis comparativos das políticas avaliadas

Uso: python3 generate_plots.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "rl_resultados"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (11, 6),
})

ROTULOS = {"RL": "Política RL (DQN)", "BuscaBinariaDT": "Busca Binária (DT)",
           "Estatico60": "Frota Estática 60"}
CORES = {"RL": "tab:green", "BuscaBinariaDT": "tab:blue", "Estatico60": "tab:red"}


def grafico_treino(algo: str = "dqn"):
    """Curva de aprendizado: recompensa por episódio + média móvel."""
    src = RES / f"treino_{algo}.csv"
    df = pd.read_csv(src)
    ep = df["recompensa_total"].dropna().reset_index(drop=True)
    n = len(ep)
    janela = max(10, min(50, max(2, n // 10)))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(np.arange(1, n + 1), ep, color="steelblue", alpha=.45, lw=1,
            label=f"recompensa por episódio ({n} episódios)")
    media = ep.rolling(janela, min_periods=1).mean()
    ax.plot(np.arange(1, n + 1), media, color="tab:blue", lw=2.4,
            label=f"média móvel ({janela} eps)")
    ax.axhline(float(media.iloc[-1]), color="tab:green", ls=":", lw=1.6,
               label=f"nível final ≈ {media.iloc[-1]:.0f}")
    ax.axhline(0, color="gray", lw=.8, ls="--")

    # marca de exploração -> exploração reduzida (epsilon-greedy do DQN: 30% do treino)
    ax.annotate("exploração\ngreedy", xy=(n * .15, float(ep.min())),
                fontsize=8.5, color="dimgray", ha="center",
                xytext=(n * .15, float(ep.min()) + (float(ep.max()) - float(ep.min())) * .18),
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=1))

    ax.set_xlabel("Episódio de treinamento (1 episódio = 20 h simuladas = 240 decisões)")
    ax.set_ylabel("Recompensa acumulada por episódio")
    ax.set_title(f"Aprendizado do agente {algo.upper()} — controle do tamanho da frota")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = RES / f"grafico_treino_{algo}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f">> {out}")
    return out


def grafico_resultados():
    """Painéis de barras comparando RL × Busca Binária × Estático (média das seeds)."""
    src = RES / "resumo_comparacao_rl.csv"
    df = pd.read_csv(src, sep=";", decimal=",").set_index("Politica")
    df = df.loc[[k for k in ("RL", "BuscaBinariaDT", "Estatico60") if k in df.index]]

    paineis = [
        ("SLA_ok(%)", "Entregas dentro do SLA (80 min) (%)", "{:.1f}%"),
        ("MMT_med(min)", "MMT médio por entrega (min)", "{:.1f}"),
        ("Frota_media", "Frota média ponderada no tempo", "{:.1f}"),
        ("Entregas", "Entregas concluídas (20 h)", "{:.0f}"),
        ("Recompensa_media", "Recompensa média por decisão", "{:.1f}"),
        ("Decisao_CPU_med(ms)", "CPU por decisão (ms)", "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (col, titulo, fmt) in zip(axes.flat, paineis):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].dropna()
        barras = ax.bar([ROTULOS.get(i, i) for i in vals.index], vals.values,
                        color=[CORES.get(i, "tab:gray") for i in vals.index],
                        edgecolor="black", linewidth=.6, width=.62)
        melhor = vals.idxmax() if col in ("SLA_ok(%)", "Entregas", "Recompensa_media") \
            else vals.idxmin()
        cores = {"RL": "lime", "BuscaBinariaDT": "gold", "Estatico60": "tomato"}
        for barra, idx in zip(barras, vals.index):
            if idx == melhor:
                barra.set_edgecolor(cores.get(idx, "black"))
                barra.set_linewidth(2.2)
        ax.bar_label(barras, fmt=fmt, padding=2, fontsize=10)
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=.3, ls="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9, rotation=8)

    fig.suptitle("Resultado do modelo treinado — RL × Busca Binária × Estático "
                 "(média de 5 cenários idênticos)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = RES / "grafico_resultado_modelo.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f">> {out}")
    return out


if __name__ == "__main__":
    grafico_treino("dqn")
    grafico_resultados()
