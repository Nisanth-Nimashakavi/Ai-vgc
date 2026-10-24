"""Policy/value network for doubles.

Architecture (vgc-bench's transformer-over-Pokemon idea, with these changes):

- Each move is its own vector (learned embedding + static features from
  `encode.move_table()` + PP/last-used + damage/KO/speed against both
  opposing actives), and a Pokemon token keeps all four move vectors instead
  of pooling them.
- A transformer runs over a global token (weather, field, rating...) plus the
  12 Pokemon tokens.
- Actions are *scored*, not read off a fixed output layer: a move action is
  scored from (slot query, that move's vector, the target's token, the damage
  of that move on that target, mega flag); a switch from (slot query, the
  switch-in's token). The network can then generalise across move slots and
  target positions instead of learning 107 unrelated outputs.
- Slot b is conditioned on slot a's action (autoregressive), so the two slots
  form a coherent joint action (e.g. not both attacking a Protecting foe).
- A value head predicts the probability of winning from the global token.
- With `series=True`, the Bo3 series context (`series.py`) is added to each move vector, each
  Pokemon token and the global token through zero-initialised layers, so a network fine-tuned from
  one without it starts out identical.
- With `aux=True`, an auxiliary head predicts what each opposing active does
  this turn: which of its four moves (or a switch) and whether it targets our
  slot a, slot b or something else. It is trained on the human games and
  gives the shared encoder a signal beyond win/loss; search can use it later.
  With `aux_mt=True` too, the target is predicted per move (Protect and a
  single-target attack get their own targets) instead of once per Pokemon.
- With `aux_extended=True`, additional auxiliary heads predict:
  - Future weather (next turn)
  - Future terrain (next turn)
  - Damage dealt by each active to each opposing active
  - Survival turns for each Pokemon
  - Game length (turns remaining)
  - Entropy collapse (how much hidden info revealed next turn)
  These dense supervision signals force the encoder to learn game mechanics
  (type chart, damage formula, ability interactions) rather than memorizing matchups.

Action ids are poke-env's DoublesEnv ids: 0 pass, 1-6 switch, 7-26 move k
target t (id = 7 + 5k + t + 2, t in -2..2), 27-46 the same plus mega. Ids
47-106 (z-move, dynamax, tera) are never legal in this format.
"""

from __future__ import annotations

import torch
from torch import nn

from ai_vgc.nn.encode import N_ACT, N_GLOB, N_MV, N_TOK, T
from ai_vgc.nn.series import N_SER_GLOB, N_SER_MV, N_SER_TOK

# FP16-safe negative infinity for masking logits (FP16 max ~65504)
NEG = -1e4


def mlp(i: int, h: int, o: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o if o is not None else h))


class Policy(nn.Module):
    def __init__(self, sizes: dict[str, int], move_table, d: int = 256, layers: int = 4,
                 heads: int = 8, dropout: float = 0.1, n_dmg: int = 8, aux: bool = False,
                 aux_mt: bool = False, aux_extended: bool = False, series: bool = False):
        # n_dmg: damage features per move x target this network reads. Checkpoints from
        # before the "blocked" flag have no n_dmg in their config and read the first 8.
        super().__init__()
        self.config = {"d": d, "layers": layers, "heads": heads, "dropout": dropout, "n_dmg": n_dmg, "aux": aux}
        if aux_mt:  # only when set, so older checkpoints' configs still compare equal
            self.config["aux_mt"] = True
        if aux_extended:
            self.config["aux_extended"] = True
        if series:
            self.config["series"] = True
        self.n_dmg = n_dmg
        de, dm = 64, 64
        self.species = nn.Embedding(sizes["species"], de)
        self.item = nn.Embedding(sizes["items"], 32)
        self.ability = nn.Embedding(sizes["abilities"], 32)
        self.move = nn.Embedding(sizes["moves"], de)
        self.register_buffer("move_table", torch.as_tensor(move_table, dtype=torch.float32))
        self.move_proj = mlp(de + N_MV + 2 + 2 * n_dmg, dm)
        self.tok_proj = mlp(de + 32 + 32 + N_TOK + 4 * dm, d)
        self.glob_proj = mlp(N_GLOB, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)
        self.none_tok = nn.Parameter(torch.zeros(d))
        self.slot_emb = nn.Embedding(2, d)
        self.prev_act = nn.Embedding(N_ACT + 1, d)  # index 0 = unknown / slot a
        self.query = mlp(3 * d, d)
        self.move_score = mlp(d + dm + d + n_dmg + 1, d, 1)
        self.switch_score = mlp(2 * d, d, 1)
        self.pass_score = nn.Linear(d, 1)
        self.value = mlp(d, d, 1)
        if aux:
            self.opp_query = mlp(2 * d, d)
            self.opp_move = mlp(d + dm, d, 1)
            self.opp_switch = nn.Linear(d, 1)
            self.opp_target = mlp(d + dm, d, 3) if aux_mt else nn.Linear(d, 3)
        if aux_extended:
            # Extended auxiliary heads (Nessie-style dense supervision)
            from ai_vgc.nn.encode import WEATHERS, FIELDS, TERRAINS
            n_weather = len(WEATHERS)
            n_terrain = len(TERRAINS)
            n_pokemon = 12  # 6 us + 6 them
            # Future weather (next turn) - from global token
            self.future_weather = mlp(d, d, n_weather)
            # Future terrain (next turn) - from global token
            self.future_terrain = mlp(d, d, n_terrain)
            # Damage prediction: each active (4) -> each opposing active (2) = 8 pairs
            # Predict damage bucket (0-5: none, 0-25%, 25-50%, 50-75%, 75-100%, 100%+)
            self.damage_pred = mlp(2 * d + dm, d, 6)  # attacker tok + defender tok + move vec
            # Survival prediction: turns until faint for each of 12 Pokemon (0-10, 10+)
            self.survival_pred = mlp(d, d, 11)
            # Game length: turns remaining (0-30, bucketed)
            self.game_length_pred = mlp(d, d, 16)
            # Entropy collapse: how much hidden info revealed next turn (0-1, bucketed)
            self.entropy_collapse_pred = mlp(d, d, 11)
        self.series = series
        if series:
            self.ser_mv = nn.Linear(N_SER_MV, dm)
            self.ser_tok = nn.Linear(N_SER_TOK, d)
            self.ser_glob = nn.Linear(N_SER_GLOB, d)
            for lin in (self.ser_mv, self.ser_tok, self.ser_glob):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    # ------------------------------------------------------------ encoder

    def encode(self, b: dict[str, torch.Tensor]):
        """Returns (cls [B,d], tokens [B,T,d], move vectors [B,T,4,dm])."""
        B = b["tok_cat"].shape[0]
        mv = b["mv_cat"].long()
        act = b["act_tok"].long()
        # Damage rows live per active slot; move them onto the active Pokemon's tokens.
        dmg = b["dmg"][..., :self.n_dmg]
        if dmg.shape[-1] < self.n_dmg:
            dmg = nn.functional.pad(dmg, (0, self.n_dmg - dmg.shape[-1]))
        tok_dmg = dmg.new_zeros((B, T, 4, 2 * self.n_dmg))
        rows = torch.arange(B, device=act.device)
        for s in range(4):
            ok = act[:, s] >= 0
            tok_dmg[rows[ok], act[ok, s]] = dmg[ok, s].flatten(-2)
        m = self.move_proj(torch.cat([
            self.move(mv), self.move_table[mv], b["mv_dyn"].float(), tok_dmg.float()], -1))
        series = self.series and "ser_tok" in b  # inputs saved before the series context: game 1
        if series:
            m = m + self.ser_mv(b["ser_mv"].float())
        cat = b["tok_cat"].long()
        x = self.tok_proj(torch.cat([
            self.species(cat[..., 0]), self.item(cat[..., 1]), self.ability(cat[..., 2]),
            b["tok_num"].float(), m.flatten(-2)], -1))
        g = self.glob_proj(b["glob"].float())
        if series:
            x = x + self.ser_tok(b["ser_tok"].float())
            g = g + self.ser_glob(b["ser_glob"].float())
        g = g.unsqueeze(1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                         b["tok_num"][..., 0] == 0], 1)
        h = self.norm(self.encoder(torch.cat([g, x], 1), src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:], m

    def _pick(self, toks: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """toks[b, idx[b]], or the learned 'none' vector where idx < 0."""
        out = toks.gather(1, idx.clamp(min=0)[:, None, None].expand(-1, 1, toks.shape[-1]))[:, 0]
        return torch.where((idx >= 0)[:, None], out, self.none_tok.expand_as(out))

    # ------------------------------------------------------------ heads

    def slot_logits(self, enc, b: dict[str, torch.Tensor], slot: int,
                    prev: torch.Tensor | None = None) -> torch.Tensor:
        """Unmasked logits [B, N_ACT] for one slot; `prev` = slot a's action (slot b only)."""
        cls, toks, m = enc
        B, d = cls.shape
        act = b["act_tok"].long()
        actor = self._pick(toks, act[:, slot])
        ctx = self.slot_emb.weight[slot].expand(B, d)
        if prev is not None:
            ctx = ctx + self.prev_act((prev.long() + 1).clamp(min=0))
        q = self.query(torch.cat([actor, cls, ctx], -1))

        # Moves: [B, 4 moves, 5 targets, 2 mega].
        mv = m.gather(1, act[:, slot].clamp(min=0)[:, None, None, None]
                      .expand(-1, 1, 4, m.shape[-1]))[:, 0]
        mv = mv * (act[:, slot] >= 0)[:, None, None]
        # Target order matches t = -2, -1, 0, 1, 2: our b, our a, none, their a, their b.
        tg = torch.stack([self._pick(toks, act[:, 1]), self._pick(toks, act[:, 0]),
                          self.none_tok.expand(B, d), self._pick(toks, act[:, 2]),
                          self._pick(toks, act[:, 3])], 1)
        dmg = b["dmg"][:, slot, ..., :self.n_dmg].float()  # [B, 4, 2, n_dmg]
        if dmg.shape[-1] < self.n_dmg:
            dmg = nn.functional.pad(dmg, (0, self.n_dmg - dmg.shape[-1]))
        dmg = torch.cat([dmg.new_zeros(B, 4, 3, self.n_dmg), dmg], 2)  # zeros for t = -2, -1, 0
        feats = torch.cat([
            q[:, None, None].expand(B, 4, 5, d), mv[:, :, None].expand(B, 4, 5, -1),
            tg[:, None].expand(B, 4, 5, d), dmg], -1)
        feats = torch.stack([feats, feats], 3)
        mega = feats.new_tensor([0.0, 1.0]).view(1, 1, 1, 2, 1).expand(B, 4, 5, 2, 1)
        s_move = self.move_score(torch.cat([feats, mega], -1))[..., 0]  # [B, 4, 5, 2]
        s_move = s_move.permute(0, 3, 1, 2).reshape(B, 40)  # [plain 20, mega 20], k-major

        s_switch = self.switch_score(torch.cat([q[:, None].expand(B, 6, d), toks[:, :6]], -1))[..., 0]
        s_pass = self.pass_score(q)
        rest = s_pass.new_full((B, N_ACT - 47), NEG)
        return torch.cat([s_pass, s_switch, s_move, rest], 1).float()

    def opp_logits(self, enc, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Opponent-action head: ([B, 2, 5] move 0-3 or switch, [B, 2, 4, 3] target our a / our b /
        other for each move; the same for all four moves without `aux_mt`)."""
        cls, toks, m = enc
        act = b["act_tok"].long()
        moves, targets = [], []
        for s in (2, 3):
            q = self.opp_query(torch.cat([self._pick(toks, act[:, s]), cls], -1))
            mv = m.gather(1, act[:, s].clamp(min=0)[:, None, None, None].expand(-1, 1, 4, m.shape[-1]))[:, 0]
            qm = torch.cat([q[:, None].expand(-1, 4, -1), mv], -1)
            moves.append(torch.cat([self.opp_move(qm)[..., 0], self.opp_switch(q)], -1))
            targets.append(self.opp_target(qm) if self.config.get("aux_mt")
                           else self.opp_target(q)[:, None].expand(-1, 4, -1))
        return torch.stack(moves, 1).float(), torch.stack(targets, 1).float()

    def extended_aux_logits(self, enc, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extended auxiliary heads (Nessie-style dense supervision).

        Returns dict with:
            - future_weather: [B, n_weather]
            - future_terrain: [B, n_terrain]
            - damage: [B, 4, 2, 6] (attacker slot, defender slot, damage bucket)
            - survival: [B, 12, 11] (pokemon, turns until faint bucket)
            - game_length: [B, 16] (turns remaining bucket)
            - entropy_collapse: [B, 11] (hidden info revealed next turn bucket)
        """
        if not self.config.get("aux_extended"):
            return {}
        cls, toks, m = enc
        B, d = cls.shape
        act = b["act_tok"].long()

        out = {}

        # Future weather and terrain from global token
        out["future_weather"] = self.future_weather(cls)
        out["future_terrain"] = self.future_terrain(cls)

        # Damage prediction: for each active attacker (our a, our b, their a, their b)
        # against each opposing active (their a, their b / our a, our b)
        # We use the attacker's token, defender's token, and the move vector
        damage_logits = []
        for att_slot in range(4):
            att_tok = self._pick(toks, act[:, att_slot])
            att_mv = m.gather(1, act[:, att_slot].clamp(min=0)[:, None, None, None]
                              .expand(-1, 1, 4, m.shape[-1]))[:, 0]  # [B, 4, dm]
            # Use first move for damage prediction (or could use max power move)
            att_mv_first = att_mv[:, 0]  # [B, dm]
            for def_slot in [2, 3] if att_slot < 2 else [0, 1]:
                def_tok = self._pick(toks, act[:, def_slot])
                feats = torch.cat([att_tok, def_tok, att_mv_first], -1)  # [B, 2d + dm]
                damage_logits.append(self.damage_pred(feats))
        # Shape: [B, 8, 6] -> [B, 4, 2, 6]
        damage_stack = torch.stack(damage_logits, 1)
        # Sanity check: should have 8 predictions (4 attackers × 2 defenders)
        if damage_stack.shape[1] != 8:
            # Fallback: reshape dynamically (e.g., if some slots are empty)
            damage_stack = damage_stack.view(B, 4, -1, 6)
        out["damage"] = damage_stack.view(B, 4, 2, 6)

        # Survival prediction for all 12 Pokemon
        survival_logits = []
        for i in range(12):
            tok = toks[:, i]
            survival_logits.append(self.survival_pred(tok))
        out["survival"] = torch.stack(survival_logits, 1)  # [B, 12, 11]

        # Game length from global token
        out["game_length"] = self.game_length_pred(cls)

        # Entropy collapse from global token
        out["entropy_collapse"] = self.entropy_collapse_pred(cls)

        return out

    def forward(self, b: dict[str, torch.Tensor], prev: torch.Tensor):
        """(logits slot a, logits slot b given slot a played `prev`, win logit)."""
        enc = self.encode(b)
        return (self.slot_logits(enc, b, 0), self.slot_logits(enc, b, 1, prev),
                self.value(enc[0])[:, 0].float())


def slot_b_mask(mask_b: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    """Torch version of `encode.joint_mask` over a batch."""
    m = mask_b.clone()
    rows = torch.arange(len(a0), device=a0.device)
    sw = (a0 >= 1) & (a0 <= 6)
    m[rows[sw], a0[sw].long()] = False
    mega = (a0 >= 27) & (a0 <= 46)
    m[mega, 27:47] = False
    tera = (a0 >= 87) & (a0 <= 106)
    m[tera, 87:107] = False
    # If slot a passed (a0 == 0) and slot b has any legal switch, slot b cannot also pass
    # during a forced replacement phase (disallow illegal double-pass).
    passed_a = (a0 == 0)
    has_switch_b = m[:, 1:7].any(-1)
    m[passed_a & has_switch_b, 0] = False
    return m


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A row with nothing legal (shouldn't happen) falls back to "pass" to avoid NaNs.
    mask = mask.clone()
    mask[~mask.any(1), 0] = True
    # Keep numerical faults inside the legal set.  Crucially, use the dtype's
    # floor—not a finite sentinel such as -1e4—for illegal logits: malformed
    # states can legitimately drive every legal logit below that sentinel.
    logits = torch.nan_to_num(logits, nan=0.0, neginf=0.0, posinf=0.0)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
