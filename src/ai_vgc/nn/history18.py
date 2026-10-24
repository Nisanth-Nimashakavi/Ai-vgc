"""Assemble full public roster memory from per-perspective native announcements.

Flags: seen/selected, current-item-known, current-item-present, ability-known,
fainted, active. Item embeddings retain the last announced item after loss;
the current-item flags distinguish consumption/removal from still holding it.
"""
import numpy as np


def known_history(pool, full_teams, native_picks, learner_sides, facts):
    count = len(full_teams)
    teams = full_teams.T.reshape(-1)
    opponents = full_teams[:, ::-1].T.reshape(-1)
    cat, num, moves = [a.reshape(2, count, 12, *a.shape[2:])
                       for a in pool.observations(teams, opponents)]
    flags = np.zeros((2, count, 12, 6), dtype=np.float32)
    flags[:, :, :6, 1] = 1
    flags[:, :, :6, 2] = pool.item_present[teams].reshape(2, count, 6)
    flags[:, :, :6, 3] = 1
    batch = np.arange(count)
    for perspective, us in enumerate((learner_sides, 1 - learner_sides)):
        for side in range(2):
            rows = facts[us, batch, side, :4]
            valid = rows[:, :, 0] != 0
            destination = native_picks[:, side] + (6 * (us != side))[:, None]
            bi = np.broadcast_to(batch[:, None], valid.shape)[valid]
            dest = destination[valid]
            r = rows[valid]
            cat[perspective, bi, dest, 0] = r[:, 1]
            cat[perspective, bi, dest, 1] = np.where(r[:, 4] == 65535, 0, r[:, 4])
            cat[perspective, bi, dest, 2] = np.where((r[:, 5] != 0) & (r[:, 6] != 65535), r[:, 6], 0)
            moves[perspective, bi, dest] = np.where(r[:, 9:13] == 65535, 0, r[:, 9:13])
            num[perspective, bi, dest] = r[:, 13:19] / 500.0
            flags[perspective, bi, dest] = np.stack((
                r[:, 0], r[:, 2], (r[:, 2] != 0) & (r[:, 3] != 65535),
                r[:, 5], r[:, 7], r[:, 8],
            ), axis=-1)
    return cat, num, moves, flags


def apply_known_active_sets(cat, facts, learner_sides):
    batch = np.arange(len(learner_sides))
    for perspective, us in enumerate((learner_sides, 1 - learner_sides)):
        rows = facts[us, batch, 1 - us, :4]
        valid = rows[:, :, 19] > 0
        bi = np.broadcast_to(batch[:, None], valid.shape)[valid]
        r = rows[valid]
        slots = 3 + r[:, 19]
        cat[perspective, bi, slots, 1] = np.where((r[:, 2] != 0) & (r[:, 3] != 65535), r[:, 3], 0)
        cat[perspective, bi, slots, 2] = np.where((r[:, 5] != 0) & (r[:, 6] != 65535), r[:, 6], 0)
