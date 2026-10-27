# -*- coding: utf-8 -*-
"""Compact return-conditioned Decision Transformer for the fleet environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

OBS_DIM = 11
N_ACTIONS = 7


class DecisionTransformer(nn.Module):
    """Causal sequence policy conditioned on return-to-go, states and actions."""
    def __init__(self, context: int = 20, hidden: int = 128, layers: int = 3,
                 heads: int = 4):
        super().__init__()
        self.context = context
        self.state_embed = nn.Linear(OBS_DIM, hidden)
        self.return_embed = nn.Linear(1, hidden)
        self.action_embed = nn.Embedding(N_ACTIONS + 1, hidden)  # extra=start token
        self.position = nn.Embedding(context, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.action_head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, states, returns_to_go, previous_actions, padding_mask=None):
        batch, length, _ = states.shape
        pos = torch.arange(length, device=states.device).unsqueeze(0)
        x = (self.state_embed(states) + self.return_embed(returns_to_go)
             + self.action_embed(previous_actions) + self.position(pos))
        causal_mask = torch.triu(torch.ones(length, length, device=states.device,
                                           dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.action_head(x)


class DecisionTransformerPolicy:
    """Evaluator adapter with episode-local context and return-to-go tracking."""
    def __init__(self, model, device, context, rtg_scale, target_return):
        self.model, self.device = model.eval(), device
        self.context, self.rtg_scale = int(context), float(rtg_scale)
        self.target_return = float(target_return)
        self.reset_episode()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu"):
        device = torch.device(device)
        data = torch.load(path, map_location=device, weights_only=True)
        if data.get("obs_dim") != OBS_DIM:
            raise ValueError(f"Checkpoint DT espera {data.get('obs_dim')} entradas; esperado {OBS_DIM}.")
        model = DecisionTransformer(context=data["context"], hidden=data["hidden"],
                                    layers=data["layers"], heads=data["heads"])
        model.load_state_dict(data["state_dict"])
        return cls(model.to(device), device, data["context"], data["rtg_scale"],
                   data["target_return"])

    def reset_episode(self):
        self.states, self.actions, self.rtgs = [], [], []
        self.current_return = self.target_return

    def predict(self, observation, deterministic=True):
        state = np.asarray(observation, dtype=np.float32).reshape(OBS_DIM)
        self.states.append(state)
        self.rtgs.append(self.current_return)
        start = max(0, len(self.states) - self.context)
        states = np.asarray(self.states[start:], dtype=np.float32)
        # The state at t receives the action taken at t-1 (or start token at t=0).
        prev_actions = ([N_ACTIONS] + self.actions[:len(states) - 1]
                        if start == 0 else self.actions[start - 1:len(self.states) - 1])
        returns = np.asarray(self.rtgs[start:], dtype=np.float32) / self.rtg_scale
        s = torch.as_tensor(states, device=self.device).unsqueeze(0)
        r = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(1, -1, 1)
        a = torch.as_tensor(prev_actions, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = int(self.model(s, r, a)[0, -1].argmax().item())
        self.actions.append(action)
        return action, None

    def observe_transition(self, reward: float):
        self.current_return -= float(reward)

