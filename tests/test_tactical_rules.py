import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon, PokemonType, Field
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder, PassBattleOrder
from poke_env.data import GenData, to_id_str
import logging

from ai_vgc.nn.rules import _GUESS, move_fails, effective_priority

logger = logging.getLogger("test")

def filter_tactical_blunders(battle: DoubleBattle, orders: list, pairs: list, act0: Pokemon | None, act1: Pokemon | None, rules: bool) -> tuple[list, list]:
    if not rules or not orders:
        return orders, pairs

    token = _GUESS.set(True)
    try:
        living_foes = [m for m in battle.opponent_active_pokemon if m is not None and not m.fainted]
        living_opp_positions = [
            i + 1 for i, m in enumerate(battle.opponent_active_pokemon) if m is not None and not m.fainted
        ]

        PRIORITY_BLOCKING_SPECIES = {"farigiraf", "bruxish", "tsareena"}
        has_priority_blocker = any(
            (getattr(f, "species", "") in PRIORITY_BLOCKING_SPECIES)
            or (getattr(f, "ability", None) and to_id_str(f.ability) in ("armortail", "dazzling", "queenlymajesty"))
            for f in living_foes
        )
        has_psychic_terrain = Field.PSYCHIC_TERRAIN in getattr(battle, "fields", {})

        def resolve_target(tgt_code: int) -> int:
            if len(living_opp_positions) == 1:
                return living_opp_positions[0]
            if tgt_code not in living_opp_positions and living_opp_positions:
                return living_opp_positions[0]
            return tgt_code

        def is_priority_blunder(m_order, mon):
            if not m_order or not isinstance(getattr(m_order, "order", None), Move) or mon is None:
                return False
            move = m_order.order
            prio = effective_priority(battle, mon, move)
            if prio > 0:
                if has_priority_blocker:
                    return True
                if has_psychic_terrain:
                    tgt_pos = resolve_target(getattr(m_order, "move_target", 0))
                    tgt_mon = battle.opponent_active_pokemon[tgt_pos - 1] if 0 < tgt_pos <= len(battle.opponent_active_pokemon) else None
                    if tgt_mon is not None and getattr(battle, "is_grounded", lambda m: True)(tgt_mon):
                        return True
                    if any(getattr(battle, "is_grounded", lambda m: True)(f) for f in living_foes):
                        return True
            return False

        def is_charge_trap(m_order, mon):
            if not m_order or not isinstance(getattr(m_order, "order", None), Move):
                return False
            mid = getattr(m_order.order, "id", "")
            if mid == "electroshot":
                if getattr(mon, "item", "") == "powerherb":
                    return False
                weather = getattr(battle, "weather", {})
                if not any("rain" in str(w).lower() for w in weather):
                    return True
            return False

        def is_protect_spam(m_order, mon):
            if not m_order or not isinstance(getattr(m_order, "order", None), Move) or mon is None:
                return False
            mid = getattr(m_order.order, "id", "")
            if mid in ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap"):
                return getattr(mon, "protect_counter", 0) >= 1
            return False

        filtered_orders = []
        filtered_pairs = []

        for o, (a, b) in zip(orders, pairs):
            o1 = getattr(o, "first_order", None)
            o2 = getattr(o, "second_order", None)
            m1 = getattr(o1, "order", None) if o1 else None
            m2 = getattr(o2, "order", None) if o2 else None
            m1_is_move = isinstance(m1, Move) and act0 is not None
            m2_is_move = isinstance(m2, Move) and act1 is not None

            drop = False

            if m1_is_move and (is_charge_trap(o1, act0) or is_protect_spam(o1, act0)):
                drop = True
            if m2_is_move and (is_charge_trap(o2, act1) or is_protect_spam(o2, act1)):
                drop = True

            if not drop:
                if m1_is_move and is_priority_blunder(o1, act0):
                    drop = True
                if m2_is_move and is_priority_blunder(o2, act1):
                    drop = True

            if not drop and act0 is not None and act1 is not None and not act0.fainted and not act1.fainted:
                if m1_is_move and getattr(m1, "id", "") in ("earthquake", "magnitude"):
                    o2_is_protect = m2_is_move and getattr(m2, "id", "") in (
                        "protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "wideguard"
                    )
                    act1_immune = (
                        PokemonType.FLYING in getattr(act1, "types", ())
                        or getattr(act1, "ability", "") in ("levitate", "telepathy", "eartheater")
                        or getattr(act1, "item", "") == "airballoon"
                    )
                    if not o2_is_protect and not act1_immune:
                        all_foes_immune = all(
                            PokemonType.FLYING in getattr(f, "types", ())
                            or getattr(f, "ability", "") in ("levitate", "telepathy", "eartheater")
                            or getattr(f, "item", "") == "airballoon"
                            for f in living_foes
                        )
                        act1_mult = act1.damage_multiplier(m1) if hasattr(act1, "damage_multiplier") else 1.0
                        if all_foes_immune or act1_mult >= 1.5 or getattr(act1, "current_hp_fraction", 1.0) < 0.45:
                            drop = True

                if not drop and m2_is_move and getattr(m2, "id", "") in ("earthquake", "magnitude"):
                    o1_is_protect = m1_is_move and getattr(m1, "id", "") in (
                        "protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "wideguard"
                    )
                    act0_immune = (
                        PokemonType.FLYING in getattr(act0, "types", ())
                        or getattr(act0, "ability", "") in ("levitate", "telepathy", "eartheater")
                        or getattr(act0, "item", "") == "airballoon"
                    )
                    if not o1_is_protect and not act0_immune:
                        all_foes_immune = all(
                            PokemonType.FLYING in getattr(f, "types", ())
                            or getattr(f, "ability", "") in ("levitate", "telepathy", "eartheater")
                            or getattr(f, "item", "") == "airballoon"
                            for f in living_foes
                        )
                        act0_mult = act0.damage_multiplier(m2) if hasattr(act0, "damage_multiplier") else 1.0
                        if all_foes_immune or act0_mult >= 1.5 or getattr(act0, "current_hp_fraction", 1.0) < 0.45:
                            drop = True

            if not drop:
                if m1_is_move:
                    t1 = resolve_target(getattr(o1, "move_target", 0))
                    fail_reason = move_fails(battle, 0, m1, t1)
                    if fail_reason:
                        drop = True
                if m2_is_move:
                    t2 = resolve_target(getattr(o2, "move_target", 0))
                    fail_reason = move_fails(battle, 1, m2, t2)
                    if fail_reason:
                        drop = True

            if not drop:
                filtered_orders.append(o)
                filtered_pairs.append((a, b))

        if filtered_orders:
            return filtered_orders, filtered_pairs
        return orders, pairs
    finally:
        _GUESS.reset(token)

def test_fakeout_into_armor_tail():
    b = DoubleBattle("tag", "p2", logger, 9)
    b._player_role = "p2"
    p_rilla = Pokemon(gen=9, species="rillaboom")
    p_rilla._active = True
    p_sylv = Pokemon(gen=9, species="sylveon")
    p_sylv._active = True
    b._active_pokemon = {"p2a": p_rilla, "p2b": p_sylv}

    p_fari = Pokemon(gen=9, species="farigiraf")
    p_fari._active = True
    p_sneas = Pokemon(gen=9, species="sneasler")
    p_sneas._active = True
    b._opponent_active_pokemon = {"p1a": p_fari, "p1b": p_sneas}

    m_fo = Move("fakeout", gen=9)
    m_hyper = Move("hypervoice", gen=9)
    order_bad = DoubleBattleOrder(SingleBattleOrder(m_fo, move_target=1), SingleBattleOrder(m_hyper, move_target=0))
    m_wood = Move("woodhammer", gen=9)
    order_good = DoubleBattleOrder(SingleBattleOrder(m_wood, move_target=1), SingleBattleOrder(m_hyper, move_target=0))

    orders = [order_bad, order_good]
    pairs = [(0, 0), (1, 0)]
    f_orders, f_pairs = filter_tactical_blunders(b, orders, pairs, p_rilla, p_sylv, True)
    assert len(f_orders) == 1
    assert f_orders[0] == order_good
    print("test_fakeout_into_armor_tail PASSED!")

def test_earthquake_friendly_fire():
    b = DoubleBattle("tag", "p2", logger, 9)
    b._player_role = "p2"
    p_swamp = Pokemon(gen=9, species="swampert")
    p_swamp._active = True
    p_arch = Pokemon(gen=9, species="archaludon")
    p_arch._active = True
    b._active_pokemon = {"p2a": p_swamp, "p2b": p_arch}

    p_raichu = Pokemon(gen=9, species="raichu")
    p_raichu._active = True
    b._opponent_active_pokemon = {"p1a": p_raichu}

    m_eq = Move("earthquake", gen=9)
    m_flash = Move("flashcannon", gen=9)
    m_protect = Move("protect", gen=9)
    m_wave = Move("wavecrash", gen=9)

    # Bad: EQ while Archaludon attacks (Earthquake hits partner Archaludon!)
    order_bad = DoubleBattleOrder(SingleBattleOrder(m_eq, move_target=0), SingleBattleOrder(m_flash, move_target=1))
    # Good: EQ while Archaludon protects
    order_good_protect = DoubleBattleOrder(SingleBattleOrder(m_eq, move_target=0), SingleBattleOrder(m_protect, move_target=0))
    # Good: Wave Crash instead of EQ
    order_good_wave = DoubleBattleOrder(SingleBattleOrder(m_wave, move_target=1), SingleBattleOrder(m_flash, move_target=1))

    orders = [order_bad, order_good_protect, order_good_wave]
    pairs = [(0, 0), (0, 1), (1, 0)]
    f_orders, f_pairs = filter_tactical_blunders(b, orders, pairs, p_swamp, p_arch, True)
    assert order_bad not in f_orders
    assert order_good_protect in f_orders
    assert order_good_wave in f_orders
    print("test_earthquake_friendly_fire PASSED!")

def test_drillrun_into_fainted_slot_redirect_to_pelipper():
    b = DoubleBattle("tag", "p2", logger, 9)
    b._player_role = "p2"
    p_goli = Pokemon(gen=9, species="golisopod")
    p_goli._active = True
    b._active_pokemon = {"p2b": p_goli}

    p_peli = Pokemon(gen=9, species="pelipper")
    p_peli._active = True
    p_fainted = Pokemon(gen=9, species="archaludon")
    p_fainted._active = True
    p_fainted.faint()
    b._opponent_active_pokemon = {"p1a": p_peli, "p1b": p_fainted}

    m_drill = Move("drillrun", gen=9)
    m_leech = Move("leechlife", gen=9)
    # Target 2 was fainted slot, but redirects to Pelipper!
    order_bad = DoubleBattleOrder(PassBattleOrder(), SingleBattleOrder(m_drill, move_target=2))
    order_good = DoubleBattleOrder(PassBattleOrder(), SingleBattleOrder(m_leech, move_target=2))

    orders = [order_bad, order_good]
    pairs = [(18, 0), (18, 1)]
    f_orders, f_pairs = filter_tactical_blunders(b, orders, pairs, None, p_goli, True)
    assert len(f_orders) == 1
    assert f_orders[0] == order_good
    print("test_drillrun_into_fainted_slot_redirect_to_pelipper PASSED!")

if __name__ == "__main__":
    test_fakeout_into_armor_tail()
    test_earthquake_friendly_fire()
    test_drillrun_into_fainted_slot_redirect_to_pelipper()
    print("ALL TESTS PASSED!")
