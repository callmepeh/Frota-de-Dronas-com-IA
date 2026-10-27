# -*- coding: utf-8 -*-
"""Collect offline trajectories and train a return-conditioned sequence policy."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from decision_transformer import DecisionTransformer, OBS_DIM, N_ACTIONS
from drone_fleet_env import ACTION_DELTAS, DroneFleetEnv, load_params
from dt_baseline import BinarySearchDT

HERE = Path(__file__).resolve().parent


def collect_episodes(seed, count, exploration):
    params = load_params()
    episodes = []
    rng = np.random.default_rng(seed)
    for episode_id in range(count):
        env = DroneFleetEnv(load_params(), seed=seed + episode_id)
        obs, _ = env.reset(seed=seed + episode_id)
        baseline = BinarySearchDT(params)
        states, actions, rewards = [], [], []
        done = False
        while not done:
            baseline.observe(env.core.proc_t)
            desired = baseline.decide(env.core, env.core.proc_t)
            delta = desired - env.core.committed()
            action = (int(np.argmin([abs(x - delta) for x in ACTION_DELTAS]))
                      if abs(delta) <= 5 else ACTION_DELTAS.index(5 if delta > 0 else -5))
            if rng.random() < exploration:
                action = int(rng.integers(N_ACTIONS))
            states.append(obs.astype(np.float32))
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated
        episodes.append({"states": np.asarray(states), "actions": np.asarray(actions),
                         "rewards": np.asarray(rewards)})
        env.close()
    return episodes


def trajectory_rtg(rewards):
    return np.concatenate((np.cumsum(rewards[::-1])[::-1], np.zeros(1))).astype(np.float32)


def sample_batch(episodes, batch_size, context, rtg_scale, device):
    states = np.zeros((batch_size, context, OBS_DIM), np.float32)
    returns = np.zeros((batch_size, context, 1), np.float32)
    actions = np.full((batch_size, context), N_ACTIONS, np.int64)
    previous = np.full((batch_size, context), N_ACTIONS, np.int64)
    targets = np.zeros((batch_size, context), np.int64)
    padding = np.ones((batch_size, context), bool)
    for b in range(batch_size):
        ep = episodes[random.randrange(len(episodes))]
        end = random.randrange(1, len(ep["actions"]) + 1)
        start = max(0, end - context)
        length = end - start
        # Right padding keeps every query position's causal attention row valid.
        # With left padding, padded query tokens can have all keys masked, which
        # produces NaNs in Transformer attention on recent PyTorch versions.
        states[b, :length] = ep["states"][start:end]
        returns[b, :length, 0] = trajectory_rtg(ep["rewards"])[start:end] / rtg_scale
        targets[b, :length] = ep["actions"][start:end]
        if start == 0:
            previous[b, 1:length] = ep["actions"][:end - 1]
        else:
            previous[b, :length] = ep["actions"][start - 1:end - 1]
        padding[b, :length] = False
    return (torch.tensor(states, device=device), torch.tensor(returns, device=device),
            torch.tensor(previous, device=device), torch.tensor(targets, device=device),
            torch.tensor(padding, device=device))


def main():
    ap = argparse.ArgumentParser(description="Treina Decision Transformer offline para frota")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--exploration", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--updates-per-epoch", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--context", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-out", type=Path, default=None)
    args = ap.parse_args()
    if args.episodes < 2 or args.epochs < 1 or args.context < 2:
        ap.error("episodes >= 2, epochs >= 1 e context >= 2 são necessários")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = collect_episodes(args.seed + 200_000, args.episodes, args.exploration)
    if args.dataset_out:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dataset_out, episodes=np.array(episodes, dtype=object))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    val_count = max(1, len(episodes) // 10)
    val_eps = [episodes[i] for i in order[:val_count]]
    train_eps = [episodes[i] for i in order[val_count:]]
    returns = np.array([ep["rewards"].sum() for ep in train_eps])
    target_return = float(np.quantile(returns, 0.75))
    rtg_scale = max(float(np.quantile(np.abs(returns), 0.75)), 1.0)

    model = DecisionTransformer(context=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.updates_per_epoch):
            s, r, prev, target, mask = sample_batch(
                train_eps, args.batch_size, args.context, rtg_scale, device)
            logits = model(s, r, prev, padding_mask=mask)
            loss = nn.functional.cross_entropy(logits[~mask], target[~mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(max(20, args.updates_per_epoch // 10)):
                s, r, prev, target, mask = sample_batch(
                    val_eps, args.batch_size, args.context, rtg_scale, device)
                val_losses.append(float(nn.functional.cross_entropy(
                    model(s, r, prev, padding_mask=mask)[~mask], target[~mask]).item()))
        score = float(np.mean(val_losses))
        if not np.isfinite(score):
            raise RuntimeError(
                f"Perda de validação não finita na época {epoch + 1}; "
                "checkpoint não será tratado como válido."
            )
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={np.mean(losses):.4f} val={score:.4f}")
        if score < best_val:
            best_val = score
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"obs_dim": OBS_DIM, "context": args.context, "hidden": 128,
                        "layers": 3, "heads": 4, "rtg_scale": rtg_scale,
                        "target_return": target_return,
                        "state_dict": model.state_dict(), "seed": args.seed,
                        "dataset_episodes": args.episodes,
                        "behavior_exploration": args.exploration,
                        "validation_loss": best_val}, args.out)
    if not args.out.is_file():
        raise RuntimeError("O treino terminou sem produzir checkpoint Decision Transformer.")
    print(f"Checkpoint DT salvo em {args.out}; alvo de retorno={target_return:.2f}; val={best_val:.4f}")


if __name__ == "__main__":
    main()
