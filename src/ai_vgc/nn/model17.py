"""Wolfe_Alt6767 17-Token Transformer Model (~7.5M Parameters).

Architecture:
- 17 Tokens:
    0..3:  Our 4 Pokémon (0: Active A, 1: Active B, 2: Bench 1, 3: Bench 2)
    4..7:  Opposing 4 Pokémon (4: Foe A, 5: Foe B, 6: Foe Bench 1, 7: Foe Bench 2)
    8..11: Active Slot A's 4 Moves (Precomputed damage arithmetic & speed vs Foes A & B)
    12..15: Active Slot B's 4 Moves (Precomputed damage arithmetic & speed vs Foes A & B)
    16:    Field Token (Weather, Terrain, Trick Room, Screens, Turns)
- Backbone: 8 Transformer layers, width d=256, 8 heads, dim_feedforward=1024, GELU.
- Structured relational move-target bilinear attention heads for Slot A and Slot B (19 actions each).
- Autoregressive conditioning of Slot B on Slot A.
- 90-Plan Team Preview Head (15 combos of 4 * 6 lead pairs).
- Value Head (win probability / game outcome).
- Opponent Action Prediction Head (simultaneous move search prior).
- Auxiliary KO Prediction Head.
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

NEG_INF = -1e4


def mlp(in_dim: int, hidden_dim: int, out_dim: int | None = None, dropout: float = 0.0) -> nn.Sequential:
    out = out_dim if out_dim is not None else hidden_dim
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out))
    return nn.Sequential(*layers)


class Transformer17(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_species: int = 2048,
        max_items: int = 1024,
        max_abilities: int = 512,
        max_moves: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Categorical Embeddings
        self.species_emb = nn.Embedding(max_species, 64, padding_idx=0)
        self.item_emb = nn.Embedding(max_items, 32, padding_idx=0)
        self.ability_emb = nn.Embedding(max_abilities, 32, padding_idx=0)
        self.type_emb = nn.Embedding(32, 16, padding_idx=0)

        self.move_emb = nn.Embedding(max_moves, 64, padding_idx=0)
        self.move_cat_emb = nn.Embedding(8, 8, padding_idx=0)
        self.move_target_emb = nn.Embedding(16, 8, padding_idx=0)

        self.weather_emb = nn.Embedding(16, 16, padding_idx=0)
        self.terrain_emb = nn.Embedding(16, 16, padding_idx=0)

        # Token Input Projections to d_model
        # Pokémon tokens (0..7): [species(64) + item(32) + ability(32) + type(16) + tok_num(16)] = 160
        self.mon_proj = mlp(64 + 32 + 32 + 16 + 16, d_model, d_model)

        # Move tokens (8..15): [move(64) + cat(8) + type(16) + target(8) + tok_num(16)] = 112
        self.move_proj = mlp(64 + 8 + 16 + 8 + 16, d_model, d_model)

        # Field token (16): [weather(16) + terrain(16) + dummy(16) + tok_num(16)] = 64
        self.field_proj = mlp(16 + 16 + 16, d_model, d_model)

        # Positional / Role Embeddings for 17 tokens
        self.pos_emb = nn.Embedding(17, d_model)

        # Transformer Backbone (8 layers, width 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        # Action Embeddings for autoregressive conditioning (Slot B depends on Slot A)
        self.action_emb = nn.Embedding(20, d_model)  # 0..18 actions + 19 as unknown/start

        # Relational Move-Target Scoring
        # A move action evaluates (Actor Query, Move Token, Target Token)
        self.query_a = mlp(d_model, d_model, d_model)
        self.query_b = mlp(2 * d_model, d_model, d_model)

        # Scoring MLPs
        self.move_score_mlp = mlp(3 * d_model, d_model, 1)
        self.switch_score_mlp = mlp(2 * d_model, d_model, 1)
        self.pass_score_mlp = nn.Linear(d_model, 1)

        # 90-Plan Team Preview Head (15 combos * 6 lead pairs)
        self.preview_head = mlp(d_model, d_model, 90)

        # Value Head (Game win probability from Field Token)
        self.value_head = mlp(d_model, 128, 1)

        # Opponent Action Prediction Heads (for simultaneous search)
        self.opp_act_a = mlp(d_model, d_model, 19)
        self.opp_act_b = mlp(d_model, d_model, 19)

        # Auxiliary KO Prediction Head (predicts KO probability on Foes A & B)
        self.ko_head = mlp(2 * d_model, 64, 2)

    def embed_tokens(self, tok_cat: torch.Tensor, tok_num: torch.Tensor) -> torch.Tensor:
        """Embed and project 17 tokens.
        tok_cat: [B, 17, 4] (uint32 / long)
        tok_num: [B, 17, 16] (float32)
        Returns: [B, 17, d_model]
        """
        B = tok_cat.shape[0]
        device = tok_cat.device

        # 1. Mon tokens (0..7)
        mon_cat = tok_cat[:, :8].long()
        sp_id = torch.where((mon_cat[..., 0] >= self.species_emb.num_embeddings) | (mon_cat[..., 0] == 65535), 0, mon_cat[..., 0])
        it_id = torch.where((mon_cat[..., 1] >= self.item_emb.num_embeddings) | (mon_cat[..., 1] == 65535), 0, mon_cat[..., 1])
        ab_id = torch.where((mon_cat[..., 2] >= self.ability_emb.num_embeddings) | (mon_cat[..., 2] == 65535), 0, mon_cat[..., 2])
        ty_id = torch.where((mon_cat[..., 3] >= self.type_emb.num_embeddings) | (mon_cat[..., 3] == 65535), 0, mon_cat[..., 3])

        mon_sp = self.species_emb(sp_id)
        mon_it = self.item_emb(it_id)
        mon_ab = self.ability_emb(ab_id)
        mon_ty = self.type_emb(ty_id)
        mon_num = tok_num[:, :8]
        mon_in = torch.cat([mon_sp, mon_it, mon_ab, mon_ty, mon_num], dim=-1)
        mon_h = self.mon_proj(mon_in)

        # 2. Move tokens (8..15)
        mv_cat = tok_cat[:, 8:16].long()
        m_id = torch.where((mv_cat[..., 0] >= self.move_emb.num_embeddings) | (mv_cat[..., 0] == 65535), 0, mv_cat[..., 0])
        mv_id = self.move_emb(m_id)
        mv_c = self.move_cat_emb(mv_cat[..., 1].clamp(0, 7))
        mv_ty = self.type_emb(mv_cat[..., 2].clamp(0, 31))
        mv_tg = self.move_target_emb(mv_cat[..., 3].clamp(0, 15))
        mv_num = tok_num[:, 8:16]
        mv_in = torch.cat([mv_id, mv_c, mv_ty, mv_tg, mv_num], dim=-1)
        mv_h = self.move_proj(mv_in)

        # 3. Field token (16)
        f_cat = tok_cat[:, 16].long()
        f_we = self.weather_emb(f_cat[..., 0].clamp(0, 15))
        f_te = self.terrain_emb(f_cat[..., 1].clamp(0, 15))
        f_num = tok_num[:, 16]
        f_in = torch.cat([f_we, f_te, f_num], dim=-1)
        f_h = self.field_proj(f_in).unsqueeze(1)

        # Combine all 17 tokens
        tokens = torch.cat([mon_h, mv_h, f_h], dim=1)

        # Add positional embedding
        pos_ids = torch.arange(17, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_emb(pos_ids)
        return tokens

    def forward(
        self,
        tok_cat: torch.Tensor,
        tok_num: torch.Tensor,
        action_a: torch.Tensor | None = None,
        is_preview: bool = False,
        token_context: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        Args:
            tok_cat: [B, 17, 4]
            tok_num: [B, 17, 16]
            action_a: [B] (optional ground truth action for teacher forcing slot B)
            is_preview: whether this step is team preview
        Returns:
            dict containing:
                - 'policy_a': [B, 19]
                - 'policy_b': [B, 19]
                - 'preview_plan': [B, 90]
                - 'value': [B, 1]
                - 'opp_a': [B, 19]
                - 'opp_b': [B, 19]
                - 'ko_pred': [B, 2]
        """
        B = tok_cat.shape[0]
        x = self.embed_tokens(tok_cat, tok_num)
        if token_context is not None:
            x = x + (token_context.unsqueeze(1) if token_context.ndim == 2 else token_context)
        h = self.norm(self.transformer(x))  # [B, 17, d_model]

        field_tok = h[:, 16]  # [B, d]

        # 1. Team Preview Plan (90 classes)
        # Pool all mon tokens (0..7) + field token (16)
        preview_rep = h[:, :8].mean(dim=1) + field_tok
        preview_logits = self.preview_head(preview_rep)

        # 2. Value Head (Win probability from field token)
        value = torch.tanh(self.value_head(field_tok))

        # 3. Opponent Action Prediction
        opp_a_logits = self.opp_act_a(h[:, 4])  # Foe A token
        opp_b_logits = self.opp_act_b(h[:, 5])  # Foe B token

        # 4. Auxiliary KO Prediction on Foe A & B
        ko_logits = self.ko_head(torch.cat([h[:, 4], h[:, 5]], dim=-1))

        # 5. Relational Action Scoring for Slot A (Token 0)
        qa = self.query_a(h[:, 0])  # [B, d]

        # Candidate target tokens for Slot A attacks:
        # Target 0: Foe A (token 4)
        # Target 1: Foe B (token 5)
        # Target 2: Ally / Self (token 1: Active B)
        # Target 3: Spread / Field (token 16)
        targets_a = torch.stack([h[:, 4], h[:, 5], h[:, 1], h[:, 16]], dim=1)  # [B, 4, d]

        # Moves for Slot A: Tokens 8..11
        moves_a = h[:, 8:12]  # [B, 4, d]

        # Factorized bilinear move-target scoring: 4 moves x 4 targets = 16 logits
        W_mv = self.move_score_mlp[0].weight
        b_mv = self.move_score_mlp[0].bias
        d = self.d_model
        h_qa = torch.matmul(qa, W_mv[:, :d].t())[:, None, None, :]
        h_ma = torch.matmul(moves_a, W_mv[:, d:2*d].t())[:, :, None, :]
        h_ta = torch.matmul(targets_a, W_mv[:, 2*d:].t())[:, None, :, :]
        move_a_scores = self.move_score_mlp[2](self.move_score_mlp[1](h_qa + h_ma + h_ta + b_mv)).view(B, 16)

        # Switch scoring: Bench 1 (token 2) and Bench 2 (token 3)
        bench_mons = torch.stack([h[:, 2], h[:, 3]], dim=1)  # [B, 2, d]
        qa_bench_exp = qa[:, None, :].expand(B, 2, -1)
        switch_a_in = torch.cat([qa_bench_exp, bench_mons], dim=-1)
        switch_a_scores = self.switch_score_mlp(switch_a_in).view(B, 2)  # [B, 2]

        # Pass scoring
        pass_a_score = self.pass_score_mlp(qa)  # [B, 1]

        policy_a = torch.cat([move_a_scores, switch_a_scores, pass_a_score], dim=-1)  # [B, 19]

        # 6. Relational Action Scoring for Slot B (Token 1, autoregressively conditioned on Action A)
        if action_a is not None:
            act_a_idx = action_a.long().clamp(0, self.action_emb.num_embeddings - 2)
        else:
            # During inference without teacher forcing, take argmax of policy_a
            act_a_idx = policy_a.argmax(dim=-1)

        act_a_emb = self.action_emb(act_a_idx)
        qb = self.query_b(torch.cat([h[:, 1], act_a_emb], dim=-1))  # [B, d]

        # Candidate target tokens for Slot B attacks:
        # Target 0: Foe A (token 4)
        # Target 1: Foe B (token 5)
        # Target 2: Ally / Self (token 0: Active A)
        # Target 3: Spread / Field (token 16)
        targets_b = torch.stack([h[:, 4], h[:, 5], h[:, 0], h[:, 16]], dim=1)  # [B, 4, d]

        # Moves for Slot B: Tokens 12..15
        moves_b = h[:, 12:16]  # [B, 4, d]

        h_qb = torch.matmul(qb, W_mv[:, :d].t())[:, None, None, :]
        h_mb = torch.matmul(moves_b, W_mv[:, d:2*d].t())[:, :, None, :]
        h_tb = torch.matmul(targets_b, W_mv[:, 2*d:].t())[:, None, :, :]
        move_b_scores = self.move_score_mlp[2](self.move_score_mlp[1](h_qb + h_mb + h_tb + b_mv)).view(B, 16)

        qb_bench_exp = qb[:, None, :].expand(B, 2, -1)
        switch_b_in = torch.cat([qb_bench_exp, bench_mons], dim=-1)
        switch_b_scores = self.switch_score_mlp(switch_b_in).view(B, 2)  # [B, 2]

        pass_b_score = self.pass_score_mlp(qb)  # [B, 1]

        policy_b = torch.cat([move_b_scores, switch_b_scores, pass_b_score], dim=-1)  # [B, 19]

        output = {
            "policy_a": policy_a,
            "policy_b": policy_b,
            "preview_plan": preview_logits,
            "value": value,
            "opp_a": opp_a_logits,
            "opp_b": opp_b_logits,
            "ko_pred": ko_logits,
        }
        if return_hidden:
            output.update(_hidden=h, _query_a=qa, _query_b=qb)
        return output
