# -*- coding: utf-8 -*-
"""
Ambiente Gymnasium (MDP) para dimensionamento de frota de drones por RL.

Este módulo EMULA a dinâmica de eventos discretos do simulador Java
(DiscreteSimMain / SimScheduler / Drone) com alta fidelidade, sem executá-lo:

| Mecânica do Java                  | Réplica aqui                                    |
|-----------------------------------|-------------------------------------------------|
| Drone.missionSteps                | estágios exponenciais 30s/20min/30s/30s/20min/30s|
| Utils.generateExponential         | -ln(1-U)*mean (numpy Generator)                  |
| chegadas `AD` exponenciais        | idem, com AD dinâmico do config.properties       |
| DT_DECISION_INTERVAL (5 min)      | um passo = 1 intervalo de decisão                |
| SimScheduler.runDigitalTwin       | substituído pela ação do agente RL               |
| requestRentals/requestReturns     | aluguel/devolução com trânsito de 15 s           |
| RECHARGE_COMPLETE / pendings      | recarga de 15 s + fila de devoluções             |
| fila máxima, drop                 | idem (maxQueueSize)                              |

Os CSVs existentes do projeto são usados apenas para VALIDAÇÃO/CALIBRAÇÃO
(validate_env.py), não para treino direto.

Espaço de estados |S| (vetor observável):
  1. fila/maxQueue, 2. intensidade de demanda (adMin/AD), 3. frota operacional,
  4. ociosos, 5. em recarga, 6. em trânsito (aluguel+devolução),
  7. EWMA de MMT, 8-9. fase do ciclo (sin/cos),
  10. MMT recente e 11. proporção recente de violações do SLA

Espaço de ações |A|: {-5,-3,-1,0,+1,+3,+5} drones (delta no alvo de frota)
Recompensa R: sinal gradual de MMT/SLA recente, frota operacional e fila
"""
from __future__ import annotations

import heapq
import json
import math
import re
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Estágios de missão de Drone.java: (duracao_media_ms, tipo)
MISSION_STAGES = (
    (30_000, "pouso"), (1_200_000, "voo"),
    (30_000, "pouso"), (30_000, "pouso"),
    (1_200_000, "voo"), (30_000, "pouso"),
)
FLIGHT_MEAN_MS = sum(d for d, _ in MISSION_STAGES)  # 2_490_000 ms (~41,5 min)

ACTION_DELTAS = (-5, -3, -1, 0, +1, +3, +5)
RESERVE_LEVELS = (0, 2, 4, 6)
# Ações da política residual sobre o DT: correção discreta em drones
RESIDUAL_DELTAS = (-4.0, -2.0, 0.0, 2.0, 4.0)


@dataclass
class RLParams:
    """Parâmetros carregados do config.properties + overrides de RL."""
    # Simulação
    simulation_time_ms: float = 72_000_000.0
    decision_interval_ms: float = 300_000.0
    max_queue: int = 1000
    recharge_ms: float = 15_000.0
    transit_ms: float = 15_000.0
    target_mmt_ms: float = 2_400_000.0
    sla_mmt_ms: float = 4_800_000.0  # SLA de 80 min (notebook de análise)
    idle_reserve: int = 2
    # Demanda (AD dinâmico em ms; dt = tempo médio de interchegada)
    ad_fixed: float = 30_000.0
    dynamic_ad: bool = True
    ad_min: float = 30_000.0
    ad_max: float = 120_000.0
    up_time: float = 9_000_000.0
    high_time: float = 0.0
    down_time: float = 9_000_000.0
    low_time: float = 9_000_000.0
    # Frota — intervalo de decisão da IA (expande os limites de min=max=60 do
    # projeto para que o RL/DT tenham margem real de otimização)
    fleet_min: int = 30
    fleet_max: int = 100
    initial_fleet: int = 60
    # Recompensa gradual baseada em qualidade, SLA, frota e fila.
    r_quality: float = 2.0
    r_sla: float = 3.0
    r_per_drone: float = -0.03
    r_queue: float = -0.5
    # Energia (proxy p/ deliveries/Wh)
    wh_per_min_voo: float = 2.0
    wh_per_min_pouso: float = 1.0


@dataclass
class RewardProfile:
    """Recompensa configurável e registrável (nenhum peso implícito).

    Termos por passo de decisão (janela = entregas do intervalo):
      + w_sla * (2*sla_frac - 1)                     SLA da janela (−1..+1)
      + w_mmt * (1 - min(mmt/mmt_limit, 2))          MMT médio da janela
      + w_p95 * (1 - min(p95/p95_limit, 2))          P95 das entregas recentes
      + w_queue * fila/maxQueue                      pressão de fila
      + w_over80 * excesso_médio_por_entrega         entrega acima do SLA
      + w_fleet * frota_operacional                  custo de frota própria
      + w_rental * drone_h_alugados_no_passo         custo de aluguel
      + w_change * |Δfrota|/fmax                     estabilidade das decisões
      + w_sla_short * (alvo − sla_frac) se abaixo    restrição de SLA (alvo)

    Alvo/limites explícitos: sla_target (fração), mmt_limit_ms, p95_limit_ms,
    sla_mmt_ms (limite de 80 min para violação individual).
    """
    name: str = "equilibrado"
    sla_target: float = 0.92
    mmt_limit_ms: float = 2_400_000.0      # 40 min (alvo do DT)
    p95_limit_ms: float = 4_800_000.0      # 80 min
    sla_mmt_ms: float = 4_800_000.0        # SLA individual de 80 min
    w_sla: float = 3.0
    w_mmt: float = 2.0
    w_p95: float = 1.0
    w_queue: float = -2.0
    w_over80: float = -1.0                 # por minuto médio de excesso/entrega
    w_fleet: float = -0.03                 # por drone operacional por passo
    w_rental: float = -1.0                 # por drone-h alugado no passo
    w_change: float = -1.0                 # por |Δfrota| normalizada
    w_sla_short: float = -3.0              # restrição: alvo de SLA não atingido


# Perfis obrigatórios da pesquisa (pesos registrados em cada experimento).
REWARD_PROFILES: dict[str, RewardProfile] = {
    "servico": RewardProfile(
        name="servico", sla_target=0.95,
        w_sla=4.0, w_mmt=3.0, w_p95=2.0, w_queue=-3.0,
        w_over80=-2.0, w_fleet=-0.01, w_rental=-0.5,
        w_change=-0.5, w_sla_short=-6.0,
    ),
    "equilibrado": RewardProfile(
        name="equilibrado", sla_target=0.92,
        w_sla=3.0, w_mmt=2.0, w_p95=1.0, w_queue=-2.0,
        w_over80=-1.0, w_fleet=-0.03, w_rental=-1.0,
        w_change=-1.0, w_sla_short=-3.0,
    ),
    "custo": RewardProfile(
        name="custo", sla_target=0.90,
        w_sla=2.0, w_mmt=1.0, w_p95=0.5, w_queue=-1.0,
        w_over80=-0.5, w_fleet=-0.08, w_rental=-2.5,
        w_change=-1.5, w_sla_short=-4.0,
    ),
}


def load_reward_profile(source) -> RewardProfile:
    """Perfil por nome de preset ou por caminho de JSON com pesos explícitos."""
    if isinstance(source, RewardProfile):
        return source
    key = str(source)
    if key in REWARD_PROFILES:
        return REWARD_PROFILES[key]
    path = Path(key)
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {f.name: data[f.name] for f in fields(RewardProfile) if f.name in data}
    return RewardProfile(**fields)


def load_params(path: str | Path | None = None, **overrides) -> RLParams:
    """Carrega config.properties do projeto como base e aplica overrides."""
    p = RLParams()
    path = Path(path) if path else PROJECT_ROOT / "config.properties"
    if path.is_file():
        props = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
        g = lambda k, t, d=None: t(props[k]) if k in props else d
        p.recharge_ms = g("RECARGA", float, p.recharge_ms)
        p.transit_ms = g("DESLOCAMENTO", float, p.transit_ms)
        p.target_mmt_ms = g("DT_TARGET_MMT", float, p.target_mmt_ms)
        p.idle_reserve = g("DT_IDLE_RESERVE", int, p.idle_reserve)
        p.decision_interval_ms = g("DT_DECISION_INTERVAL", float, p.decision_interval_ms)
        p.max_queue = g("maxQueueSize", int, p.max_queue)
        p.ad_fixed = g("ad", float, p.ad_fixed)
        if "adMin" in props:
            p.dynamic_ad = True
            p.ad_min = g("adMin", float, p.ad_min)
            p.ad_max = g("adMax", float, p.ad_max)
            p.up_time = g("upTime", float, p.up_time)
            p.high_time = g("highTime", float, 0.0)
            p.down_time = g("downTime", float, p.down_time)
            p.low_time = g("lowTime", float, p.low_time)
        else:
            p.dynamic_ad = False
    for k, v in overrides.items():
        if v is not None:
            if k == "fleet_min":
                p.fleet_min = int(v)
            elif k == "fleet_max":
                p.fleet_max = int(v)
            else:
                setattr(p, k, v)
    return p


class DroneSimCore:
    """Núcleo de eventos discretos — réplica fiel do simulador Java."""

    AVAILABLE, ACTIVE, RECHARGING, RENT_TRANSIT, RETURN_TRANSIT, RETURNED = range(6)

    def __init__(self, P: RLParams, seed: int):
        self.P = P
        self.rng_arr = np.random.default_rng(seed)            # stream de chegadas (independente)
        self.rng_ops = np.random.default_rng(seed + 1_000_003)  # missões/recargas/trânsitos
        self.now = 0.0
        self.proc_t = 0.0
        self.events: list[tuple] = []
        self.seq = 0
        self.queue: deque = deque()
        self.drones: list[dict] = []
        self.counts = [0] * 6
        self.rented_op = 0          # drones alugados operacionais (custo)
        self.pending_returns = 0
        self.next_drone_id = P.initial_fleet
        for _ in range(P.initial_fleet):
            self._add_drone(rented=False, state=self.AVAILABLE)
        # métricas
        self.deliveries: list[tuple[float, float]] = []  # (t, tempo_missao_ms)
        self.generated = 0
        self.dropped = 0
        self.total_voo_ms = 0.0
        self.total_pouso_ms = 0.0
        self.rentals_total = 0
        self.returns_total = 0
        # integrais temporais
        self.I_fleet = self.I_active = self.I_idle =        self.I_rented = 0.0
        self._schedule_first_arrival()

    # ---------- utilidades ----------
    def _exp(self, rng, mean: float) -> float:
        u = rng.random()
        while u <= 0.0 or u >= 1.0:
            u = rng.random()
        return -math.log(1.0 - u) * mean

    def _schedule(self, delay: float, kind: str, payload=None):
        self.seq += 1
        heapq.heappush(self.events, (self.now + max(0.0, delay), self.seq, kind, payload))

    def _add_drone(self, rented: bool, state: int):
        d = {"id": self.next_drone_id, "rented": rented, "state": state}
        self.next_drone_id += 1
        self.drones.append(d)
        self.counts[state] += 1
        if rented and state in (self.AVAILABLE, self.ACTIVE, self.RECHARGING):
            self.rented_op += 1
        return d

    def _set_state(self, d: dict, new_state: int):
        old = d["state"]
        self.counts[old] -= 1
        self.counts[new_state] += 1
        if d["rented"]:
            old_op = old in (self.AVAILABLE, self.ACTIVE, self.RECHARGING)
            new_op = new_state in (self.AVAILABLE, self.ACTIVE, self.RECHARGING)
            self.rented_op += int(new_op) - int(old_op)
        d["state"] = new_state

    def ad_of(self, t: float) -> float:
        """AD(t) — réplica de DiscreteSimMain.getDynamicAD (ms de interchegada)."""
        P = self.P
        if not P.dynamic_ad:
            return P.ad_fixed
        cycle = P.up_time + P.high_time + P.down_time + P.low_time
        t = t % cycle
        if t < P.up_time:
            return P.ad_min + (P.ad_max - P.ad_min) * (t / P.up_time)
        if t < P.up_time + P.high_time:
            return P.ad_max
        if t < P.up_time + P.high_time + P.down_time:
            elapsed = t - (P.up_time + P.high_time)
            return P.ad_max - (P.ad_max - P.ad_min) * (elapsed / P.down_time)
        return P.ad_min

    # ---------- consultas ----------
    def operational(self) -> int:
        return self.counts[0] + self.counts[1] + self.counts[2]

    def committed(self) -> int:
        return self.operational() + self.counts[3]

    # ---------- gestão de frota (réplica de requestRentals/requestReturns) ----------
    def apply_fleet_target(self, desired: int) -> int:
        P = self.P
        desired = int(max(P.fleet_min, min(P.fleet_max, desired)))
        if desired > self.committed():
            self.rent(desired - self.committed())
        elif desired < self.operational():
            already = self.counts[4] + self.pending_returns
            self.request_returns(max(0, self.operational() - desired - already))
        return desired

    def rent(self, n: int):
        for _ in range(n):
            self._add_drone(rented=True, state=self.RENT_TRANSIT)
            self.rentals_total += 1
            self._schedule(self.P.transit_ms, "rent_arrive", self.drones[-1]["id"])

    def request_returns(self, n: int):
        left = n
        for d in list(self.drones):
            if left == 0:
                break
            if d["rented"] and d["state"] == self.AVAILABLE:
                self._begin_return(d)
                left -= 1
        self.pending_returns += left

    def _begin_return(self, d: dict):
        self._set_state(d, self.RETURN_TRANSIT)
        self.returns_total += 1
        self._schedule(self.P.transit_ms, "return_arrive", d["id"])

    # ---------- loop de eventos ----------
    def _integrate(self, t1: float):
        dt = t1 - self.proc_t
        if dt > 0:
            c = self.counts
            self.I_fleet += dt * (c[0] + c[1] + c[2])
            self.I_active += dt * c[1]
            self.I_idle += dt * c[0]
            self.I_rented += dt * self.rented_op
        self.proc_t = t1
        self.now = t1

    def run_until(self, until: float):
        until = max(until, self.proc_t)
        while True:
            nxt = min(self.events[0][0] if self.events else until, until)
            self._integrate(nxt)
            if nxt >= until:
                break
            _, _, kind, payload = heapq.heappop(self.events)
            if kind == "arrival":
                self._handle_arrival()
            elif kind == "stage":
                self._handle_stage(payload)
            elif kind == "recharge_done":
                self._handle_recharge_done(payload)
            elif kind == "rent_arrive":
                d = self._drone(payload)
                if d and d["state"] == self.RENT_TRANSIT:
                    self._set_state(d, self.AVAILABLE)
                    self._dispatch()
            elif kind == "return_arrive":
                d = self._drone(payload)
                if d and d["state"] == self.RETURN_TRANSIT:
                    self._set_state(d, self.RETURNED)
            elif kind == "end":
                break

    def _drone(self, did: int):
        for d in self.drones:
            if d["id"] == did:
                return d
        return None

    def _schedule_first_arrival(self):
        self.now = 0.0
        self._schedule(self._exp(self.rng_arr, self.ad_of(0.0)), "arrival")

    def _reschedule_arrival(self, at_t: float):
        self.now = at_t
        self._schedule(self._exp(self.rng_arr, self.ad_of(at_t)), "arrival")

    def _handle_arrival(self):
        t = self.now
        self.generated += 1
        self._reschedule_arrival(t)  # agenda próxima chegada
        if len(self.queue) < self.P.max_queue:
            self.queue.append(t)
        else:
            self.dropped += 1
        self._dispatch()

    def _dispatch(self):
        for d in list(self.drones):
            if d["state"] != self.AVAILABLE:
                continue
            if not self.queue:
                return
            self._start_mission(d, self.queue.popleft())

    def _start_mission(self, d: dict, arrival_t: float):
        # Como no Java: missionTime = queueTime + flightTime, com queueTime
        # contado desde a chegada do pacote à fila.
        d["pickup_t"] = arrival_t
        d["stages"] = list(MISSION_STAGES)
        self._set_state(d, self.ACTIVE)
        self._next_stage(d)

    def _next_stage(self, d: dict):
        dur, typ = d["stages"][0]
        dur = self._exp(self.rng_ops, dur)
        d["stage_typ"] = typ
        self._schedule(dur, "stage", (d["id"], dur, typ))

    def _handle_stage(self, payload):
        did, dur, typ = payload
        d = self._drone(did)
        if d is None or d["state"] != self.ACTIVE:
            return
        if typ == "voo":
            self.total_voo_ms += dur
        else:
            self.total_pouso_ms += dur
        d["stages"].pop(0)
        if d["stages"]:
            self._next_stage(d)
        else:
            self.deliveries.append((self.now, self.now - d["pickup_t"]))
            self._set_state(d, self.RECHARGING)
            self._schedule(self._exp(self.rng_ops, self.P.recharge_ms), "recharge_done", did)

    def _handle_recharge_done(self, did: int):
        d = self._drone(did)
        if d is None or d["state"] != self.RECHARGING:
            return
        if d["rented"] and self.pending_returns > 0:
            self.pending_returns -= 1
            self._begin_return(d)
        else:
            self._set_state(d, self.AVAILABLE)
            self._dispatch()


class DroneFleetEnv(gym.Env):
    """Ambiente Gymnasium: ações manipulam a frota a cada intervalo de decisão."""

    metadata = {"render_modes": []}

    def __init__(self, params: RLParams | None = None, seed: int | None = None,
                 fleet_min: int | None = None, fleet_max: int | None = None,
                 obs_version: str | None = None, action_mode: str | None = None,
                 reward_profile=None):
        """obs_version: 'v2' (padrão histórico, 11 dims) ou 'v3' (estado aprimorado).
        action_mode: 'legacy' (Δ frota {-5..+5}) ou 'residual' (Δ frota + correção {−4..+4}
        sobre a recomendação corrente do Digital Twin). reward_profile: preset
        ('servico'/'equilibrado'/'custo'), caminho de JSON ou RewardProfile.
        """
        super().__init__()
        self.P = params or load_params()
        if fleet_min is not None:
            self.P.fleet_min = int(fleet_min)
        if fleet_max is not None:
            self.P.fleet_max = int(fleet_max)
        self.fmax = max(self.P.fleet_max, self.P.initial_fleet)
        self.obs_version = str(obs_version or "v2")
        self.action_mode = str(action_mode or "legacy")
        self.reward_profile = (load_reward_profile(reward_profile)
                               if reward_profile is not None else None)
        self.actions = tuple(RESIDUAL_DELTAS) if self.action_mode == "residual" else tuple(ACTION_DELTAS)
        obs_dim = 11 if self.obs_version == "v2" else 30
        low_v2 = np.array([0.0] * 7 + [-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_v2 = np.array([1.0] * 6 + [3.0, 1.0, 1.0, 3.0, 1.0], dtype=np.float32)
        if self.obs_version == "v2":
            low, high = low_v2, high_v2
        else:
            # extras: 19 dimensões v3 (ver _build_obs_v3)
            low = np.concatenate((low_v2,
                                  np.array([0.0] * 5 + [-1.0] + [0.0] * 12 + [-1.0],
                                           dtype=np.float32)))
            high = np.concatenate((high_v2,
                                   np.array([1.0] * 5 + [1.0] + [1.0] * 7 + [3.0, 3.0, 3.0,
                                            3.0, 1.0, 1.0], dtype=np.float32)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.default_seed = seed
        self._has_reset = False

    # ----- observação -----
    def _build_obs(self) -> np.ndarray:
        core, P = self.core, self.P
        c = core.counts
        mmt_norm = min(float(self.mmt_ewma) / P.sla_mmt_ms, 3.0) if self.n_delivered > 0 else 0.0
        window_mmt = float(np.mean([m for _, m in self._last_window])) if self._last_window else self.mmt_ewma
        window_mmt_norm = min(window_mmt / max(P.sla_mmt_ms, 1.0), 3.0) if self.n_delivered > 0 else 0.0
        window_viol = (sum(m > P.sla_mmt_ms for _, m in self._last_window) / len(self._last_window)
                       if self._last_window else 0.0)
        lam_rel = min(P.ad_min / max(core.ad_of(core.proc_t), 1.0), 1.0)
        cycle = P.up_time + P.high_time + P.down_time + P.low_time
        phase = 2 * math.pi * (core.proc_t % cycle) / cycle
        base = np.array([
            len(core.queue) / P.max_queue,
            lam_rel,
            core.operational() / self.fmax,
            c[0] / self.fmax,                      # ociosos
            c[2] / self.fmax,                      # recarga
            (c[3] + c[4]) / self.fmax,             # trânsito
            mmt_norm,
            math.sin(phase), math.cos(phase), window_mmt_norm, window_viol,
        ], dtype=np.float32)
        if self.obs_version == "v2":
            return base
        return self._build_obs_v3(base)

    def _build_obs_v3(self, base: np.ndarray) -> np.ndarray:
        """Estado aprimorado (v3): tudo disponível causalmente no instante da decisão.

        Estimativas de demanda usam apenas EWMA do AD instantâneo (mesma fonte
        monitorada pelo controlador DT) — nada do futuro estocástico é revelado;
        as "previsões" de +5/+10/+20 min assumem o ciclo de demanda conhecido
        (hipótese documentada: cronograma de demanda é parâmetro de operação).
        """
        core, P, fmax = self.core, self.P, self.fmax
        c = core.counts
        t = core.proc_t
        # --- demanda: EWMA causal do AD instantâneo + histórico/tendência ---
        cur_ad = core.ad_of(t)
        prev_ewma = self._demand_ewma
        self._demand_ewma += 0.30 * (cur_ad - self._demand_ewma)
        self._demand_hist.append(self._demand_ewma)
        lam_ewma_rel = min(P.ad_min / max(self._demand_ewma, 1.0), 1.0)
        trend = float(np.clip((self._demand_ewma - prev_ewma) / max(prev_ewma, 1.0), -1.0, 1.0))
        hist = list(self._demand_hist)[-4:] if len(self._demand_hist) >= 4 else list(self._demand_hist)
        # --- previsões sob o ciclo de demanda conhecido (5/10/20 min) ---
        forecasts = np.array([
            min(P.ad_min / max(core.ad_of(t + dt_ms), 1.0), 1.0)
            for dt_ms in (300_000.0, 600_000.0, 1_200_000.0)])
        # --- fila: idade média/máxima dos pedidos (causal: timestamp de chegada) ---
        if core.queue:
            ages = np.array([t - a for a in core.queue])
            q_age_mean = min(float(ages.mean()) / max(P.sla_mmt_ms, 1.0), 3.0)
            q_age_max = min(float(ages.max()) / max(P.sla_mmt_ms, 1.0), 3.0)
        else:
            q_age_mean = q_age_max = 0.0
        # --- entregas recentes (~últimas 200): P95 e violações em contagem ---
        recent = list(self._recent_deliveries)
        if recent:
            m_arr = np.array([m for _, m in recent])
            recent_p95 = min(float(np.percentile(m_arr, 95)) / max(P.sla_mmt_ms, 1.0), 3.0)
            viol_count = min(float(np.sum(m_arr > P.sla_mmt_ms)) / 50.0, 3.0)
        else:
            recent_p95 = viol_count = 0.0
        # --- recomendado do Digital Twin (controlador em operação) e gap residual ---
        dt_rec = self._dt_recommendation()
        gap = (core.committed() - dt_rec) / 5.0
        rented_idle = sum(d["rented"] and d["state"] == DroneSimCore.AVAILABLE
                          for d in core.drones)
        return_pending = (c[4] + core.pending_returns) / fmax
        return np.concatenate(
            (base, np.array([
                rented_idle / fmax,                # alugados ociosos
                return_pending,                    # devem retornar nos próximos intervalos
                core.committed() / fmax,           # frota total comprometida
                core.rented_op / fmax,             # frota alugada operacional
                lam_ewma_rel,                      # demanda atual estimada
                trend,                             # tendência da demanda
                # demanda estimada das últimas decisões (histórico rel.)
                *[min(P.ad_min / max(h, 1.0), 1.0) for h in hist],
                *forecasts,                        # previsão +5/+10/+20 min
                q_age_mean, q_age_max,             # idade média/máxima da fila
                recent_p95, viol_count,            # P95 e nº de violações recentes
                dt_rec / fmax,                     # recomendação corrente do Digital Twin
                float(np.clip(gap, -1.0, 1.0)),    # frota atual − recomendação
            ], dtype=np.float32)))

    def _current_dt_rec(self) -> int:
        """Recomendação do DT cacheada no instante do obs atual (evita dupla
        atualização da EWMA dentro do mesmo intervalo de decisão)."""
        if getattr(self, "_dt_rec_t", None) != self.core.proc_t:
            self._dt_recommendation()
            self._dt_rec_t = self.core.proc_t
        return self._last_dt_rec

    def _dt_recommendation(self) -> int:
        """Recomendação corrente do controlador Busca Binária DT (causal).

        Espelha o uso em evaluate_rl.py: observe(t) com EWMA 0,30 e decide()
        por busca binária; inclui a regra de devolução de ociosos (idle_reserve).
        """
        if self._dt is None:
            from dt_baseline import BinarySearchDT
            self._dt = BinarySearchDT(self.P)
        self._dt.observe(self.core.proc_t)
        self._last_dt_rec = int(self._dt.decide(self.core, self.core.proc_t))
        return self._last_dt_rec

    def multiagent_observation(self) -> np.ndarray:
        """Global view for cooperative agents, with rented idle capacity."""
        rented_idle = sum(d["rented"] and d["state"] == DroneSimCore.AVAILABLE
                          for d in self.core.drones)
        return np.concatenate((self._build_obs(),
                               np.array([rented_idle / self.fmax], dtype=np.float32)))

    def _advance_to_decision(self):
        core, P = self.core, self.P
        t0 = core.proc_t
        until = min(self.t_decision, P.simulation_time_ms)
        prev_deliveries = len(core.deliveries)
        core.run_until(until)
        self._last_window = core.deliveries[prev_deliveries:]

    def _update_mmt_ewma(self):
        for _, m in self._last_window:
            alpha = 1.0 / min(self.n_delivered + 1, 100)
            self.mmt_ewma += alpha * (m - self.mmt_ewma)
            self.n_delivered += 1
            self._recent_deliveries.append((self.core.proc_t, m))
        self._last_committed = self.core.committed()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        first_default_seed = seed is None and self.default_seed is not None and not self._has_reset
        gym_seed = int(self.default_seed) if first_default_seed else seed
        # Seed the Gymnasium RNG on the first reset too. Subsequent episode
        # seeds are drawn from this reproducible stream.
        super().reset(seed=gym_seed)
        if seed is not None:
            seed_value = int(seed)
        elif first_default_seed:
            seed_value = int(self.default_seed)
        else:
            seed_value = int(self.np_random.integers(0, 2**31 - 1))
        self._has_reset = True
        self.P.simulation_time_ms = max(self.P.simulation_time_ms, self.P.decision_interval_ms * 4)
        self.core = DroneSimCore(self.P, seed_value)
        self.mmt_ewma = 0.0
        self.n_delivered = 0
        self.t_decision = self.P.decision_interval_ms
        self._last_window = []
        # Estado v3: estimativa causal de demanda, histórico (deque fixo espilhado
        # com 4 amostras) e entregas recentes; DT embutido para o modo residual.
        self._demand_ewma = self.core.ad_of(0.0)
        self._demand_hist = deque([self._demand_ewma] * 4, maxlen=4)
        self._recent_deliveries = deque(maxlen=200)
        self._dt = None
        self._last_dt_rec = 0
        self._last_committed = self.core.committed()
        self._advance_to_decision()
        self._update_mmt_ewma()
        return self._build_obs(), self._info()

    def step(self, action: int):
        core, P = self.core, self.P
        if self.action_mode == "residual":
            correction = RESIDUAL_DELTAS[int(action)]
            desired = clamp_residual(core.committed(), self._current_dt_rec()
                                     + correction, P)
        else:
            delta = ACTION_DELTAS[int(action)]
            desired = clamp_fleet_delta(delta, core)
        applied = core.apply_fleet_target(desired)
        return self._finish_step(applied)

    def step_target(self, target_fleet: int):
        """Aplica um alvo de frota direto (para baselines DT/estático no modo v3
        ou residual, sem passar pelo espaço de ações do agente)."""
        applied = self.core.apply_fleet_target(int(target_fleet))
        return self._finish_step(applied)

    def step_multiagent(self, fleet_action: int, reserve_action: int):
        """Joint team step: capacity agent + idle-rental reserve agent."""
        core, P = self.core, self.P
        delta = ACTION_DELTAS[int(fleet_action)]
        desired = clamp_fleet_delta(delta, core)
        applied = core.apply_fleet_target(desired)

        reserve = RESERVE_LEVELS[int(reserve_action)]
        if not core.queue:
            rented_idle = sum(d["rented"] and d["state"] == DroneSimCore.AVAILABLE
                              for d in core.drones)
            fleet_margin = max(0, core.committed() - P.fleet_min)
            n_return = min(max(0, rented_idle - reserve), fleet_margin)
            if n_return:
                core.request_returns(n_return)
        return self._finish_step(applied)

    def _finish_step(self, applied: int):
        core, P = self.core, self.P
        self._advance_to_decision()
        self._update_mmt_ewma()
        reward = self._reward()
        self.t_decision += P.decision_interval_ms
        truncated = self.t_decision > P.simulation_time_ms
        terminated = False
        info = self._info(applied=applied, reward=reward)
        return self._build_obs(), float(reward), terminated, truncated, info

    def _reward(self) -> float:
        core, P = self.core, self.P
        if self.reward_profile is not None:
            return float(self._reward_profile(core, P))
        window = self._last_window
        if window:
            mean_mmt = float(np.mean([m for _, m in window]))
            quality = 1.0 - min(mean_mmt / max(P.sla_mmt_ms, 1.0), 2.0)
            sla_ok = sum(m <= P.sla_mmt_ms for _, m in window) / len(window)
            service = P.r_quality * quality + P.r_sla * (2.0 * sla_ok - 1.0)
        else:
            service = 0.0
        queue_pressure = len(core.queue) / max(P.max_queue, 1)
        return service + P.r_per_drone * core.operational() + P.r_queue * queue_pressure

    def _reward_profile(self, core, P) -> float:
        """Termos registrados em RewardProfile; nenhum peso implícito."""
        R, fmax = self.reward_profile, self.fmax
        window = list(self._last_window)
        recent = list(self._recent_deliveries)
        drone_h_rented = core.rented_op * (P.decision_interval_ms / 3_600_000.0)
        d_fleet = abs(core.committed() - self._last_committed) / fmax
        if window:
            m = np.array([mm for _, mm in window])
            src = np.array([mm for _, mm in recent]) if recent else m
            mean_mmt = float(m.mean())
            p95 = float(np.percentile(src, 95))
            sla_frac = float(np.mean(m <= R.sla_mmt_ms))
            over80 = float(np.mean(np.maximum(m - R.sla_mmt_ms, 0.0))) / 60_000.0
        else:
            mean_mmt = p95 = sla_frac = over80 = 0.0
        r = (R.w_sla * (2.0 * sla_frac - 1.0)
             + R.w_mmt * (1.0 - min(mean_mmt / max(R.mmt_limit_ms, 1.0), 2.0))
             + R.w_p95 * (1.0 - min(p95 / max(R.p95_limit_ms, 1.0), 2.0))
             + R.w_queue * (len(core.queue) / max(P.max_queue, 1))
             + R.w_over80 * over80
             + R.w_fleet * core.operational()
             + R.w_rental * drone_h_rented
             + R.w_change * d_fleet)
        if sla_frac < R.sla_target:
            r += R.w_sla_short * (R.sla_target - sla_frac)
        return r

    def _info(self, applied: int | None = None, reward: float | None = None) -> dict:
        core, P = self.core, self.P
        dur = max(core.proc_t, 1e-9)
        window = self._last_window
        viol = sum(1 for _, m in window if m > P.sla_mmt_ms) / len(window) if window else None
        return {
            "t_ms": core.proc_t,
            "queue": len(core.queue),
            "fleet_op": core.operational(),
            "idle": core.counts[0],
            "active": core.counts[1],
            "recharge": core.counts[2],
            "rent_transit": core.counts[3],
            "return_transit": core.counts[4],
            "rented_op": core.rented_op,
            "deliveries": len(core.deliveries),
            "generated": core.generated,
            "dropped": core.dropped,
            "mmt_ewma_min": self.mmt_ewma / 60_000,
            "window_sla_violation": viol,
            "fleet_tw": core.I_fleet / dur,
            "active_tw": core.I_active / dur,
            "rented_tw": core.I_rented / dur,
            "voo_wh": core.total_voo_ms / 60_000 * P.wh_per_min_voo,
            "pouso_wh": core.total_pouso_ms / 60_000 * P.wh_per_min_pouso,
            "rentals": core.rentals_total,
            "returns": core.returns_total,
            "dt_rec": self._last_dt_rec,
            "committed": core.committed(),
            "deliveries_list": list(self._last_window),
            "applied_fleet": applied,
            "reward": reward,
        }

    def render(self):  # pragma: no cover
        i = self._info()
        print(f"[t={i['t_ms']/3600000:.2f}h] fila={i['queue']} frota={i['fleet_op']} "
              f"mmt_ewma={i['mmt_ewma_min']:.1f}min entregas={i['deliveries']}")


def clamp_fleet_delta(delta: int, core: "DroneSimCore") -> float:
    return core.committed() + delta


def clamp_residual(committed: int, target: float, P: "RLParams") -> int:
    """Alvo final = recomendação DT + correção da IA, com limites de segurança:
    frota mínima/máxima declaradas nos parâmetros (aluguel/devolução e fila
    continuam sujeitos a apply_fleet_target, igual ao controlador DT)."""
    return int(max(P.fleet_min, min(P.fleet_max, int(round(target)))))
