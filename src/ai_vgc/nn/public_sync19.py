"""Reconcile public Pokemon facts after a guessed shadow simulation."""

STATUS_NAMES = {"BRN": "Burn", "PAR": "Paralysis", "SLP": "Sleep", "FRZ": "Freeze",
                "PSN": "Poison", "TOX": "Toxic"}
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def sync_public_pokemon_state(native, pokemon):
    """Public announcements override simulated status and active stat stages."""
    changed = False
    status = STATUS_NAMES.get(getattr(pokemon.status, "name", None), "None")
    if native["status"] != status:
        native["status"] = status
        changed = True
    if pokemon.active and not pokemon.fainted:
        boosts = [int(pokemon.boosts.get(key, 0)) for key in BOOST_KEYS]
        if native["boosts"] != boosts:
            native["boosts"] = boosts
            changed = True
    return changed


def sync_own_pp(native, pokemon, move_ids):
    """The owner's request exposes PP; do not substitute shadow simulation PP."""
    changed = False
    for name, move in pokemon.moves.items():
        move_id = move_ids.get(name)
        if move_id is None or move_id not in native['moves']:
            continue
        slot = native['moves'].index(move_id)
        for key, value in (('pp', move.current_pp), ('max_pp_override', move.max_pp)):
            if native[key][slot] != value:
                native[key][slot] = value
                changed = True
    return changed
