"""Damage formula, ported from pokemon-showdown sim/battle-actions.ts (modifyDamage/getDamage)."""
import jax.numpy as jnp


def modify(value, mod4096):
    """Showdown battle.modify with a fixed-point modifier (4096 = 1.0), round half down."""
    return (value * mod4096 + 2047) // 4096


def damage(level, power, atk, dfn, roll, crit, stab4096, type_exp, burned, spread, weather4096, final4096):
    """One damage value. All args are int arrays (broadcastable), so this vmaps over battles.

    roll: 0..15 (Showdown's this.random(16)); type_exp: sum of +1/-1 per defending type, or
    -99 for an immunity; stab4096: 4096/6144/8192; weather4096: 4096/6144/2048.
    """
    base = ((2 * level // 5 + 2) * power * atk // dfn) // 50
    base = base + 2
    base = jnp.where(spread, modify(base, 3072), base)
    base = modify(base, weather4096)
    base = jnp.where(crit, base * 3 // 2, base)
    base = base * (100 - roll) // 100
    base = modify(base, stab4096)
    up = jnp.clip(type_exp, 0, 6)
    base = base * (2 ** up)
    down = jnp.clip(-type_exp, 0, 6)
    # each resist step truncates separately, as Showdown does
    for i in range(6):
        base = jnp.where(i < down, base // 2, base)
    base = jnp.where(burned, modify(base, 2048), base)
    base = modify(base, final4096)
    base = jnp.maximum(base, 1)
    return jnp.where(type_exp <= -99, 0, base)
