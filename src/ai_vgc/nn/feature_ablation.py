"""Offline, candidate-only feature ablation; leaves the evaluation opponent intact."""
from __future__ import annotations
import os
import sys
from ai_vgc.nn import encode as encoding


def main():
    if '--server' in sys.argv or '--ladder' in sys.argv:
        raise ValueError('Feature ablations are local evaluation only')
    variant = os.environ['CTS_ABLATION']
    if variant not in ('damage', 'history'):
        raise ValueError(variant)
    original_encode = encoding.encode

    def encode(battle, rating=0.0):
        row = original_encode(battle, rating)
        if battle.__dict__.get('_feature_ablation') == 'damage':
            row['dmg'][:] = 0
        elif battle.__dict__.get('_feature_ablation') == 'history':
            row['tok_num'][:, 51:76] = 0
            row['mv_dyn'][:, :, 1] = 0
            row['glob'][35:37] = 0
        return row

    encoding.encode = encode
    from ai_vgc.nn import player
    # Explicit assignment also covers imports before this module's main().
    player.encode = encode
    original_decide = player.NNPlayer._decide
    original_preview = player.NNPlayer.teampreview

    def mark(self, battle):
        battle._feature_ablation = None if self.username.startswith('nnopp') else variant

    async def decide(self, battle, mask, preview):
        mark(self, battle)
        return await original_decide(self, battle, mask, preview)

    async def preview(self, battle):
        mark(self, battle)
        return await original_preview(self, battle)

    player.NNPlayer._decide = decide
    player.NNPlayer.teampreview = preview
    print(f'Candidate-only feature ablation: {variant}; frozen opponent unchanged', flush=True)
    player.main()


if __name__ == '__main__':
    main()
