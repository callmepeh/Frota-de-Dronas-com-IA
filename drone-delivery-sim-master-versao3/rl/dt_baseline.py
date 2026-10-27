# -*- coding: utf-8 -*-
"""
Baseline: réplica em Python da Busca Binária analítica do Digital Twin
(SimScheduler.runDigitalTwin), que a IA de RL substitui.

O DT Java avalia, para cada candidato n em [min, max], o MMT estacionário do
modelo SPN (runScenario) e escolhe a menor frota com MMT <= DT_TARGET_MMT.
Aqui o MMT estacionário é estimado com um modelo M/M/n (Erlang-C) calibrado
com os tempos de voo reais do simulador (FLIGHT_MEAN_MS), preservando a mesma
semântica: busca binária sobre a frota + regra de devolução quando a fila está
vazia (idle > idleReserve).
"""
from __future__ import annotations

import math

from drone_fleet_env import DroneSimCore, FLIGHT_MEAN_MS, RLParams, load_params


def erlang_c_probability(a: float, n: int) -> float:
    """P(espera) de M/M/n; requer a < n. Série em espaço seguro (a, n reais)."""
    if n <= 0:
        return 1.0
    term = 1.0          # a^0 / 0!
    series = 1.0        # soma dos termos k < n
    for k in range(1, n):
        term *= a / k
        series += term
    # último termo k = n: a^n/n!
    term *= a / n
    denom_top = series + term / (1.0 - a / n)
    return term / denom_top


def stationary_mmt(n: int, lam_per_ms: float, target=None) -> float:
    """MMT estacionário estimado (ms) para frota n e taxa de chegada lam (pac/ms).

    Espelha runScenario: serviço = missão completa (média FLIGHT_MEAN_MS,
    hypoexponencial aproximada por exponencial), fila M/M/n.
    Retorna inf quando o sistema é instável (lam >= n * mu).
    """
    mu = 1.0 / FLIGHT_MEAN_MS
    a = lam_per_ms / mu
    if a >= n:
        return float("inf")
    wait_prob = erlang_c_probability(a, n)
    wait_q = wait_prob / (n * mu - lam_per_ms)
    return wait_q + FLIGHT_MEAN_MS


class BinarySearchDT:
    """Réplica da decisão de frota por busca binária do DT Java."""

    def __init__(self, P: RLParams | None = None):
        self.P = P or load_params()
        self.ad_ewma = self.P.ad_min  # aproxima o AD médio monitorado pelo DatabaseManager

    def observe(self, t_ms: float, ad_ms: float | None = None):
        """Atualiza estimativa suavizada da demanda (alpha ~ janela do monitor)."""
        cur = ad_ms if ad_ms is not None else self._instant_ad(t_ms)
        self.ad_ewma += 0.30 * (cur - self.ad_ewma)

    def _instant_ad(self, t_ms: float) -> float:
        # ad dinâmico instantâneo (mesma fórmula de DiscreteSimMain.getDynamicAD)
        P = self.P
        if not P.dynamic_ad:
            return P.ad_fixed
        cycle = P.up_time + P.high_time + P.down_time + P.low_time
        t = t_ms % cycle
        if t < P.up_time:
            return P.ad_min + (P.ad_max - P.ad_min) * (t / P.up_time)
        if t < P.up_time + P.high_time:
            return P.ad_max
        if t < P.up_time + P.high_time + P.down_time:
            elapsed = t - (P.up_time + P.high_time)
            return P.ad_max - (P.ad_max - P.ad_min) * (elapsed / P.down_time)
        return P.ad_min

    def decide(self, core: DroneSimCore, t_ms: float) -> int:
        """Recomenda a frota alvo — espelha runDigitalTwin."""
        P = self.P
        lam = 1.0 / max(self.ad_ewma, 1e-6)  # pacotes/ms
        lo, hi, best = P.fleet_min, P.fleet_max, P.fleet_max
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if stationary_mmt(mid, lam) <= P.target_mmt_ms:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        desired = best
        # Regra do Java: fila vazia e ociosos acima da reserva -> devolve o excedente
        if len(core.queue) == 0 and core.counts[0] > P.idle_reserve:
            desired = max(P.fleet_min, core.operational() - (core.counts[0] - P.idle_reserve))
        return int(desired)


def evaluate_fleet_grid(P: RLParams, lam_per_ms: float, n_grid=None):
    """Utilitário de diagnóstico: MMT estacionário por tamanho de frota."""
    if n_grid is None:
        n_grid = range(P.fleet_min, P.fleet_max + 1)
    return [(n, stationary_mmt(n, lam_per_ms)) for n in n_grid]
