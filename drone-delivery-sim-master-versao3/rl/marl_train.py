# -*- coding: utf-8 -*-
"""Cooperative independent Q-learning for fleet capacity and return reserve."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drone_fleet_env import DroneFleetEnv, load_params

HERE = Path(__file__).resolve().parent
MODELOS = HERE / "modelos"
RESULTADOS = HERE / "rl_resultados"
OBS_DIM = 12
FLEET_ACTIONS = 7
RESERVE_ACTIONS = 4


class QNetwork(nn.Module):
    def __init__(self, action_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, x):
        return self.layers(x)


class IndependentQLearner:
    """One independent Q learner; agents share observation and team reward."""
    def __init__(self, action_count: int, seed: int, device: torch.device):
        torch.manual_seed(seed)
        self.device = device
        self.online = QNetwork(action_count).to(device)
        self.target = QNetwork(action_count).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.replay_capacity = 100_000
        self.replay = [None] * self.replay_capacity
        self.replay_size = 0
        self.replay_position = 0

    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.online.layers[-1].out_features))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def remember(self, obs, action, reward, next_obs, done):
        self.replay[self.replay_position] = (
            obs.copy(), int(action), float(reward), next_obs.copy(), float(done))
        self.replay_position = (self.replay_position + 1) % self.replay_capacity
        self.replay_size = min(self.replay_size + 1, self.replay_capacity)

    def update(self, batch_size: int, gamma: float):
        indices = random.sample(range(self.replay_size), batch_size)
        batch = [self.replay[i] for i in indices]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        q = self.online(obs_t).gather(1, action_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_t).max(dim=1).values
            target = reward_t + gamma * (1.0 - done_t) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()


class CooperativeFleetPolicy:
    """Inference adapter exposing both agents' joint action."""
    def __init__(self, fleet_net: QNetwork, reserve_net: QNetwork, device: torch.device):
        self.fleet_net = fleet_net.eval()
        self.reserve_net = reserve_net.eval()
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        device_obj = torch.device(device)
        data = torch.load(path, map_location=device_obj, weights_only=True)
        if data.get("obs_dim") != OBS_DIM:
            raise ValueError(f"Checkpoint MARL espera {data.get('obs_dim')} entradas; esperado {OBS_DIM}.")
        fleet_net, reserve_net = QNetwork(FLEET_ACTIONS), QNetwork(RESERVE_ACTIONS)
        fleet_net.load_state_dict(data["fleet_state_dict"])
        reserve_net.load_state_dict(data["reserve_state_dict"])
        return cls(fleet_net.to(device_obj), reserve_net.to(device_obj), device_obj)

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            fleet_action = int(self.fleet_net(x).argmax(dim=1).item())
            reserve_action = int(self.reserve_net(x).argmax(dim=1).item())
        return (fleet_action, reserve_action), None


def save_policy(path: Path, fleet_agent: IndependentQLearner,
                reserve_agent: IndependentQLearner, step: int,
                validation_reward: float, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "obs_dim": OBS_DIM,
        "fleet_state_dict": fleet_agent.online.state_dict(),
        "reserve_state_dict": reserve_agent.online.state_dict(),
        "step": int(step),
        "validation_reward": float(validation_reward),
        "seed": int(seed),
        "action_deltas": [-5, -3, -1, 0, 1, 3, 5],
        "idle_reserve_levels": [0, 2, 4, 6],
    }, path)


def evaluate_team(fleet_agent, reserve_agent, params, seed, episodes, device):
    env = DroneFleetEnv(load_params(
        fleet_min=params.fleet_min, fleet_max=params.fleet_max), seed=seed)
    scores = []
    for _ in range(episodes):
        env.reset()
        obs = env.multiagent_observation()
        done = False
        total = 0.0
        while not done:
            fa = fleet_agent.act(obs, 0.0, np.random.default_rng(0))
            ra = reserve_agent.act(obs, 0.0, np.random.default_rng(1))
            obs, reward, terminated, truncated, _ = env.step_multiagent(fa, ra)
            total += reward
            done = terminated or truncated
            obs = env.multiagent_observation()
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description="MARL cooperativo: capacidade + reserva de alugados")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-seed", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--fleet-min", type=int, default=None)
    ap.add_argument("--fleet-max", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()

    validation_seed = (args.validation_seed if args.validation_seed is not None
                       else args.seed + 100_000)
    if validation_seed == args.seed:
        ap.error("a seed de validação deve ser diferente da seed de treino")
    if args.timesteps <= 0 or args.eval_freq <= 0 or args.eval_episodes <= 0:
        ap.error("timesteps, eval-freq e eval-episodes devem ser positivos")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_params(fleet_min=args.fleet_min, fleet_max=args.fleet_max)
    env = DroneFleetEnv(params, seed=args.seed)
    rng = np.random.default_rng(args.seed + 987_653)
    fleet_agent = IndependentQLearner(FLEET_ACTIONS, args.seed + 17, device)
    reserve_agent = IndependentQLearner(RESERVE_ACTIONS, args.seed + 31, device)
    # Separate module initialization; seed both learners from the experiment seed.
    random.seed(args.seed + 71)

    out = args.out or (MODELOS / "marl_coop_drone_fleet_v1.pt")
    results_dir = args.results_dir or (RESULTADOS / "marl_coop")
    results_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    validation_path = results_dir / "best_validation.pt"
    validation_env_params = load_params(fleet_min=params.fleet_min, fleet_max=params.fleet_max)

    obs, _ = env.reset()
    obs = env.multiagent_observation()
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0
    reward_log = []
    best_validation = -float("inf")
    batch_size, learning_starts, train_freq = 128, 10_000, 4
    gamma, target_update = 0.995, 2_000

    for step in range(1, args.timesteps + 1):
        progress = min(step / max(int(args.timesteps * 0.35), 1), 1.0)
        epsilon = 1.0 + progress * (0.05 - 1.0)
        fleet_action = fleet_agent.act(obs, epsilon, rng)
        reserve_action = reserve_agent.act(obs, epsilon, rng)
        next_base, reward, terminated, truncated, _ = env.step_multiagent(
            fleet_action, reserve_action)
        done = terminated or truncated
        next_obs = env.multiagent_observation()
        fleet_agent.remember(obs, fleet_action, reward, next_obs, done)
        reserve_agent.remember(obs, reserve_action, reward, next_obs, done)
        episode_reward += reward
        episode_steps += 1
        obs = next_obs

        if step >= learning_starts and step % train_freq == 0:
            if fleet_agent.replay_size >= batch_size:
                fleet_agent.update(batch_size, gamma)
                reserve_agent.update(batch_size, gamma)
        if step % target_update == 0:
            fleet_agent.target.load_state_dict(fleet_agent.online.state_dict())
            reserve_agent.target.load_state_dict(reserve_agent.online.state_dict())

        if done:
            episode_idx += 1
            reward_log.append((episode_idx, episode_steps, episode_reward))
            obs, _ = env.reset()
            obs = env.multiagent_observation()
            episode_reward, episode_steps = 0.0, 0

        if step % args.eval_freq == 0 or step == args.timesteps:
            val_score = evaluate_team(
                fleet_agent, reserve_agent, validation_env_params,
                validation_seed, args.eval_episodes, device)
            print(f"[step {step}] epsilon={epsilon:.3f} validação={val_score:.2f}")
            if val_score > best_validation:
                best_validation = val_score
                save_policy(validation_path, fleet_agent, reserve_agent,
                            step, val_score, args.seed)

    save_policy(out.with_name(f"{out.stem}_last{out.suffix}"),
                fleet_agent, reserve_agent, args.timesteps,
                best_validation, args.seed)
    best_data = torch.load(validation_path, map_location="cpu", weights_only=True)
    torch.save(best_data, out)
    log_path = results_dir / f"treino_marl_seed{args.seed}.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "passos", "recompensa_total"])
        writer.writerows(reward_log)
    env.close()
    print(f">> Melhor equipe salva em {out} (step={best_data['step']}, "
          f"validação={best_data['validation_reward']:.2f}); log={log_path}")


if __name__ == "__main__":
    main()
