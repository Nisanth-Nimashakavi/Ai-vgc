"""Fresh battle policy plus full-six CTS preview policy and value function."""
from __future__ import annotations

import torch
from torch import nn

from ai_vgc.nn.model17 import Transformer17, mlp
from ai_vgc.nn.preview18 import NUM_PREVIEW_PLANS


class Transformer18(Transformer17):
    def __init__(self, *, preview_layers=2, aux_extended=True, **kwargs):
        super().__init__(**kwargs)
        width = self.d_model
        self.preview_head = mlp(width, width, NUM_PREVIEW_PLANS)
        self.preview_mon_proj = mlp(64 + 32 + 32 + 64 + 6, width, width)
        self.preview_pos = nn.Embedding(12, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=kwargs.get("num_heads", 8),
            dim_feedforward=kwargs.get("dim_feedforward", 1024),
            dropout=kwargs.get("dropout", 0.0), activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.preview_transformer = nn.TransformerEncoder(layer, preview_layers, enable_nested_tensor=False)
        self.preview_norm = nn.LayerNorm(width)
        self.preview_value_head = mlp(width, 128, 1)
        self.action_emb = nn.Embedding(36, width)
        self.mega_delta = mlp(3 * width, width, 1)
        self.history_flags_proj = nn.Linear(6, width, bias=False)
        self.history_context_proj = mlp(width, width, width)

        self.aux_extended = aux_extended
        if aux_extended:
            # Dense auxiliary supervision heads:
            # 1. Resulting weather (next turn): 5 classes (0: None, 1: Rain, 2: Sun, 3: Sand, 4: Snow)
            self.aux_weather_head = mlp(width, width, 5)
            # 2. Resulting terrain (next turn): 5 classes (0: None, 1: Electric, 2: Grassy, 3: Psychic, 4: Misty)
            self.aux_terrain_head = mlp(width, width, 5)
            # 3. Who survives: 4 active slots (Our A, Our B, Foe A, Foe B) binary classification
            self.aux_survival_head = mlp(width, 64, 1)
            # 4. Resulting HP fraction: 4 active slots in [0, 1]
            self.aux_hp_head = mlp(width, 64, 1)
            # 5. Damage dealt (HP delta): 4 active slots in [0, 1]
            self.aux_damage_head = mlp(width, 64, 1)

    def encode_roster(self, cat, num, moves, flags=None):
        def embed(table, ids):
            ids = ids.long()
            ids = torch.where((ids >= 0) & (ids < table.num_embeddings), ids, 0)
            return table(ids)

        features = torch.cat((
            embed(self.species_emb, cat[..., 0]),
            embed(self.item_emb, cat[..., 1]),
            embed(self.ability_emb, cat[..., 2]),
            embed(self.move_emb, moves).mean(dim=-2), num.float(),
        ), dim=-1)
        tokens = self.preview_mon_proj(features)
        tokens = tokens + self.preview_pos(torch.arange(12, device=cat.device))
        if flags is not None:
            tokens = tokens + self.history_flags_proj(flags.float())
        hidden = self.preview_norm(self.preview_transformer(tokens)).mean(dim=1)
        return hidden

    def preview(self, cat, num, moves):
        hidden = self.encode_roster(cat, num, moves)
        return {"preview_plan": self.preview_head(hidden),
                "preview_value": self.preview_value_head(hidden).tanh().squeeze(-1)}

    def score_slot_b(self, hidden: torch.Tensor, action_a: torch.Tensor) -> torch.Tensor:
        """Fast autoregressive evaluation of Slot B directly from cached hidden tokens.
        Avoids re-running the 8-layer transformer backbone during rollout action sampling."""
        h = hidden
        B = h.shape[0]
        act_a_idx = action_a.long().clamp(0, self.action_emb.num_embeddings - 2)
        act_a_emb = self.action_emb(act_a_idx)
        qb = self.query_b(torch.cat([h[:, 1], act_a_emb], dim=-1))
        targets_b = torch.stack([h[:, 4], h[:, 5], h[:, 0], h[:, 16]], dim=1)
        moves_b = h[:, 12:16]

        W_mv = self.move_score_mlp[0].weight
        b_mv = self.move_score_mlp[0].bias
        d = self.d_model
        h_qb = torch.matmul(qb, W_mv[:, :d].t())[:, None, None, :]
        h_mb = torch.matmul(moves_b, W_mv[:, d:2*d].t())[:, :, None, :]
        h_tb = torch.matmul(targets_b, W_mv[:, 2*d:].t())[:, None, :, :]
        move_b_scores = self.move_score_mlp[2](self.move_score_mlp[1](h_qb + h_mb + h_tb + b_mv)).view(B, 16)

        bench_mons = torch.stack([h[:, 2], h[:, 3]], dim=1)
        qb_bench_exp = qb[:, None, :].expand(B, 2, -1)
        switch_b_in = torch.cat([qb_bench_exp, bench_mons], dim=-1)
        switch_b_scores = self.switch_score_mlp(switch_b_in).view(B, 2)

        pass_b_score = self.pass_score_mlp(qb)
        base = torch.cat([move_b_scores, switch_b_scores, pass_b_score], dim=-1)

        mW = self.mega_delta[0].weight
        mb = self.mega_delta[0].bias
        mh_q = torch.matmul(qb, mW[:, :d].t())[:, None, None, :]
        mh_m = torch.matmul(moves_b, mW[:, d:2*d].t())[:, :, None, :]
        mh_t = torch.matmul(targets_b, mW[:, 2*d:].t())[:, None, :, :]
        mega = base[:, :16] + self.mega_delta[2](self.mega_delta[1](mh_q + mh_m + mh_t + mb)).reshape(B, 16)
        return torch.cat([base, mega], dim=-1)

    def forward(self, tok_cat=None, tok_num=None, action_a=None, *,
                preview_cat=None, preview_num=None, preview_moves=None, preview_only=False,
                history_cat=None, history_num=None, history_moves=None, history_flags=None,
                extra_token_context=None, return_hidden=False):
        if preview_only:
            return self.preview(preview_cat, preview_num, preview_moves)
        context = None if history_cat is None else self.history_context_proj(
            self.encode_roster(history_cat, history_num, history_moves, history_flags))
        if extra_token_context is not None:
            context = extra_token_context if context is None else context.unsqueeze(1) + extra_token_context
        out = super().forward(tok_cat, tok_num, action_a=action_a,
                              token_context=context, return_hidden=True)
        hidden, qa, qb = (out.pop(k) for k in ("_hidden", "_query_a", "_query_b"))
        d = self.d_model
        mW = self.mega_delta[0].weight
        mb = self.mega_delta[0].bias
        for slot, query in enumerate((qa, qb)):
            moves = hidden[:, 8 + 4 * slot:12 + 4 * slot]
            targets = torch.stack((hidden[:, 4], hidden[:, 5], hidden[:, 1 - slot], hidden[:, 16]), dim=1)
            mh_q = torch.matmul(query, mW[:, :d].t())[:, None, None, :]
            mh_m = torch.matmul(moves, mW[:, d:2*d].t())[:, :, None, :]
            mh_t = torch.matmul(targets, mW[:, 2*d:].t())[:, None, :, :]
            key = "policy_a" if slot == 0 else "policy_b"
            base = out[key]
            mega = base[:, :16] + self.mega_delta[2](self.mega_delta[1](mh_q + mh_m + mh_t + mb)).reshape(len(base), 16)
            out[key] = torch.cat((base, mega), dim=-1)
        # The inherited four-mon preview output is not a valid full-six input.
        # Legacy clients must use this model's separate full-six preview API.
        out.pop("preview_plan")
        if preview_cat is not None:
            out.update(self.preview(preview_cat, preview_num, preview_moves))

        if getattr(self, "aux_extended", False):
            field_tok = hidden[:, 16]
            out["aux_weather"] = self.aux_weather_head(field_tok)
            out["aux_terrain"] = self.aux_terrain_head(field_tok)
            # Active Pokémon tokens: Our A (0), Our B (1), Foe A (4), Foe B (5)
            active_toks = torch.stack([hidden[:, 0], hidden[:, 1], hidden[:, 4], hidden[:, 5]], dim=1)
            out["aux_survives"] = self.aux_survival_head(active_toks).squeeze(-1)
            out["aux_hp"] = torch.sigmoid(self.aux_hp_head(active_toks).squeeze(-1))
            out["aux_damage"] = torch.sigmoid(self.aux_damage_head(active_toks).squeeze(-1))

        if return_hidden:
            out["_hidden"] = hidden

        return out
