"""Every legal Reg M-C ability and item, with its Showdown hooks and whether the sim implements it.

An entry is DONE only when a differential test against Showdown covers it. `python -m ai_vgc.gpusim.ledger`
prints the table and exits non-zero while anything is TODO, so nothing can be skipped silently.
"""
import sys
from collections import Counter

from . import effects as E
from . import tables as T

# An id is done once it is in effects.py: tests/gpusim/test_game.py draws random teams from that list, fails if
# any listed id never appears (test_every_implemented_effect_is_exercised), and diffs every turn against Showdown.
TEST = "tests/gpusim/test_game.py"
DONE_ABILITIES: dict = dict.fromkeys(E.ABILITIES, TEST)
DONE_ITEMS: dict = dict.fromkeys(E.ITEMS, TEST)


def rows(t=None):
    t = t or T.load()
    for kind, done in (("ability", DONE_ABILITIES), ("item", DONE_ITEMS)):
        for x in t.raw["abilities" if kind == "ability" else "items"]:
            yield kind, x["id"], x["handlers"], done.get(x["id"])


def main():
    todo = Counter()
    by_hook = Counter()
    for kind, name, hooks, covered in rows():
        todo[kind, bool(covered)] += 1
        if not covered:
            by_hook[kind, hooks[0] if hooks else "(no hooks)"] += 1
    for kind in ("ability", "item"):
        n_done, n_todo = todo[kind, True], todo[kind, False]
        print(f"{kind}: {n_done} done, {n_todo} todo of {n_done + n_todo}")
        print("  todo by first hook:", ", ".join(f"{h}:{n}" for (k, h), n in by_hook.most_common() if k == kind))
    return 1 if any(not c for *_, c in rows()) else 0


if __name__ == "__main__":
    sys.exit(main())
