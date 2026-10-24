"""Build search-node inputs with the same CTS mask and roster memory as training."""
import json
import numpy as np
import torch

from ai_vgc.nn.history18 import apply_known_active_sets, known_history
from ai_vgc.nn.ppo18 import mask_closed_observation
from ai_vgc.nn.showdown_history19 import merge_branch_public_facts


def search_node_inputs(native, side, context, device):
    cat = np.empty((2, 1, 17, 4), dtype=np.int64)
    num = np.empty((2, 1, 17, 16), dtype=np.float32)
    for perspective, physical_side in enumerate((side, 1-side)):
        encoded = native.encode_17(physical_side)
        cat[perspective, 0] = np.frombuffer(encoded["tok_cat"], np.uint32).reshape(17, 4)
        num[perspective, 0] = np.frombuffer(encoded["tok_num"], np.float32).reshape(17, 16)
    mask_closed_observation(cat, num)
    facts = np.frombuffer(native.public_history_18(), "<u4").reshape(2, 1, 2, 6, 20).copy()
    if context.get('root_facts') is not None:
        facts = merge_branch_public_facts(facts, context['root_facts'], context['root_side'], None)
    sides = np.array([side], dtype=np.int64)
    apply_known_active_sets(cat, facts, sides)
    memory = known_history(context["pool"], context["full_teams"],
                           context["native_picks"], sides, facts)
    histories = [{name: torch.from_numpy(value[i]).to(device) for name, value in zip(
        ("history_cat", "history_num", "history_moves", "history_flags"), memory)}
        for i in (0, 1)]
    bank = context.get("belief_bank")
    if bank is not None:
        # Each hypothetical branch gets a fresh tracker, so it cannot modify the
        # real battle's posterior. It uses public events retained in the clone.
        for i, physical_side in enumerate((side, 1-side)):
            features = np.frombuffer(bank.tracker().features(native.public_view19(physical_side)),
                                     "<f4").reshape(1, 17, 16).copy()
            histories[i]["belief_features"] = torch.from_numpy(features).to(device)
            for actor in (0, 1):
                if features[0, actor, 15] != 0:
                    num[i, 0, actor, 7] = 0
    return [(torch.from_numpy(cat[i]).to(device), torch.from_numpy(num[i]).to(device),
             histories[i]) for i in (0, 1)]


def search_leaf_values(model, leaves, side, context, device):
    encoded = [search_node_inputs(leaf, side, context, device)[0] for leaf in leaves]
    cat = torch.cat([x[0] for x in encoded])
    num = torch.cat([x[1] for x in encoded])
    history = {key: torch.cat([x[2][key] for x in encoded]) for key in encoded[0][2]}
    with torch.no_grad():
        return model(cat, num, **history)["value"].reshape(-1).tolist()
