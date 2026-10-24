"""Model18 plus explicitly uncertain, public-evidence set beliefs."""
import torch
from torch import nn
from ai_vgc.nn.model18 import Transformer18


class Transformer19(Transformer18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.belief_projection = nn.Linear(16, self.d_model, bias=False)
        nn.init.zeros_(self.belief_projection.weight)

    def forward(self, *args, belief_features=None, **kwargs):
        context = None if belief_features is None else self.belief_projection(belief_features.float())
        return super().forward(*args, extra_token_context=context, **kwargs)


class Transformer19Belief(Transformer19):
    """Public-evidence policy with auxiliary opponent-set prediction heads.

    Inputs remain public. Hidden simulator sets are used only as loss targets.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opponent_belief = nn.ModuleDict({
            "item": nn.Linear(self.d_model, self.item_emb.num_embeddings + 1),
            "ability": nn.Linear(self.d_model, self.ability_emb.num_embeddings + 1),
            "move": nn.Linear(self.d_model, self.move_emb.num_embeddings),
        })

    def forward(self, *args, **kwargs):
        if kwargs.get("preview_only", False):
            return super().forward(*args, **kwargs)
        keep_hidden = kwargs.pop("return_hidden", False)
        output = super().forward(*args, return_hidden=True, **kwargs)
        hidden = output["_hidden"][:, 4:6]
        for key, head in self.opponent_belief.items():
            output[f"belief_{key}_logits"] = head(hidden)
        if not keep_hidden:
            output.pop("_hidden")
        return output


def architecture_for_state(state):
    return Transformer19Belief if any(k.removeprefix("module.").startswith("opponent_belief.") for k in state) else Transformer19


def initialize_from_model18(model, path):
    """Continue only the corrected model18 PPO lineage, preserving initial outputs."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("input_contract") != "public-history-18-v1; 35 explicit normal/Mega actions":
        raise ValueError("Warmstart requires the corrected model18 PPO public-history checkpoint")
    if checkpoint.get("model_type") != "Transformer18":
        raise ValueError("Only the current corrected Transformer18 lineage is accepted")
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    expected_missing = {"belief_projection.weight"}
    for k in model.state_dict():
        if k.startswith(("aux_", "opponent_belief.")):
            expected_missing.add(k)
    missing = set(result.missing_keys)
    if not missing.issubset(expected_missing) or result.unexpected_keys:
        raise ValueError(f"Unexpected warmstart differences: {result}")
    if torch.count_nonzero(model.belief_projection.weight):
        raise ValueError("New belief projection must initially preserve the existing policy")
    return checkpoint
