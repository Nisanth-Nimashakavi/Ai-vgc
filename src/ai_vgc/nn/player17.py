"""Player agent and evaluation harness for the 17-token Transformer (Wolfe_Alt6767 blueprint).

Supports:
- Raw policy play (fast inference).
- 1-turn simultaneous-move search play (matrix regret matching over candidate rollouts).
- 90-plan team preview selection.
"""

from __future__ import annotations

import json
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

import vgc_engine
from ai_vgc.nn.model17 import Transformer17
from ai_vgc.nn.search17 import search_simultaneous_turn


class Player17:
    """Agent that drives battle decisions using Transformer17 with optional 1-turn simultaneous search."""

    def __init__(
        self,
        model: Transformer17,
        use_search: bool = True,
        k: int = 6,
        m: int = 5,
        blend_alpha: float = 0.5,
        device: torch.device | str = "cpu",
        search_depth: int = 2,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.use_search = use_search
        self.k = k
        self.m = m
        self.blend_alpha = blend_alpha
        self.device = device
        self.search_depth = search_depth

    @torch.no_grad()
    def choose_preview(self, battle: vgc_engine.Battle, side: int = 0) -> list[int]:
        """Select 4 Pokemon via 90-plan team preview classification head."""
        enc = battle.encode_17(side)
        cat = torch.from_numpy(np.frombuffer(enc["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()).to(self.device)
        num = torch.from_numpy(np.frombuffer(enc["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()).to(self.device)

        out = self.model(cat, num)
        plan_id = int(out["preview_plan"].argmax(dim=-1).item())
        picks_bytes = vgc_engine.plan_to_picks(plan_id)
        return list(picks_bytes)

    @torch.no_grad()
    def choose_turn(self, battle: vgc_engine.Battle, side: int = 0) -> list[Any]:
        """Select joint action for Slot 0 and Slot 1."""
        if self.use_search:
            res = search_simultaneous_turn(
                self.model,
                battle,
                our_side=side,
                k=self.k,
                m=self.m,
                blend_alpha=self.blend_alpha,
                device=self.device,
                depth=self.search_depth,
            )
            return res["choices"]

        # Raw policy fallback (argmax without search)
        mask_0 = battle.legal_action_mask_17(side, 0)
        mask_1 = battle.legal_action_mask_17(side, 1)

        enc = battle.encode_17(side)
        cat = torch.from_numpy(np.frombuffer(enc["tok_cat"], dtype=np.uint32).reshape(1, 17, 4).copy()).to(self.device)
        num = torch.from_numpy(np.frombuffer(enc["tok_num"], dtype=np.float32).reshape(1, 17, 16).copy()).to(self.device)

        out = self.model(cat, num)
        logits_a = out["policy_a"][0]
        mask_a_t = torch.tensor(mask_0, dtype=torch.bool, device=self.device)
        logits_a[~mask_a_t] = -1e9
        act_a = int(logits_a.argmax().item())

        prev_a = torch.tensor([act_a], device=self.device)
        out_b = self.model(cat, num, action_a=prev_a)
        logits_b = out_b["policy_b"][0]
        mask_b_t = torch.tensor(mask_1, dtype=torch.bool, device=self.device)
        if act_a in (16, 17) and act_a < len(mask_b_t):
            mask_b_t[act_a] = False
        if mask_b_t.sum() == 0:
            mask_b_t[18] = True
        logits_b[~mask_b_t] = -1e9
        act_b = int(logits_b.argmax().item())

        ch0, ch1 = battle.actions_17_to_choices(side, act_a, act_b)
        return [ch0, ch1]
