"""Controls for the new legality oracle, separate from measured matchups."""
import json

import vgc_engine

from scripts.validate_native_contracts import Oracle, FORMAT, state_differences


def team(first='Swampert', moves='- Tackle\n- Protect'):
    return (f'{first}\nAbility: Torrent\nLevel: 50\n{moves}\n\n'
            'Eevee\nAbility: Run Away\nLevel: 50\n- Protect\n\n'
            'Snorlax\nAbility: Immunity\nLevel: 50\n- Splash\n\n'
            'Pikachu\nAbility: Static\nLevel: 50\n- Splash').replace('Level: 50','Level: 50\nSerious Nature')


def test_neutral_legal_sets_and_seeded_python_transition_match():
    rpc=Oracle()
    try:
        init=rpc.call(op='init',format=FORMAT,seed=[1,2,3,4],p1team=team(),p2team=team())
        b=vgc_engine.Battle.from_teams(*[json.dumps(t) for t in init['sets']],format=FORMAT,
            stat_points=True,decision_phases=True,tera_allowed=False,ps_seed='1,2,3,4')
        assert not state_differences(b,init['state'])
        answer=rpc.call(op='legal')
        assert answer['original_state_unchanged']
        for side in range(2):
            for slot in range(2):
                assert set(map(tuple,answer['legal'][side][slot]))==set(map(tuple,b.legal_choices(side,slot)))
        choices=[[('move',0,0,1-side,0),('move',1,0,-1,-1)] for side in range(2)]
        result=b.step_move(*choices)
        after=rpc.call(op='step',choices=choices)
        assert not after['errors']
        assert not state_differences(b,after['state'],result)
        assert b.ps_seed==after['state']['seed']
    finally:rpc.close()


def test_helping_hand_requires_explicit_ally_in_ps_command_contract():
    rpc=Oracle()
    try:
        rpc.call(op='init',format=FORMAT,seed=[1,2,3,4],p1team=team(moves='- Helping Hand'),p2team=team())
        legal=rpc.call(op='legal')['legal'][0][0]
        assert ['move',0,0,0,1] in legal
        assert ['move',0,0,-1,-1] not in legal
    finally:rpc.close()


def test_fainted_replacements_are_switches_and_waiting_side_is_padding():
    rpc=Oracle()
    try:
        ours=team('Garchomp @ Choice Band','- Earthquake').replace('Ability: Torrent','Ability: Rough Skin')
        theirs=team('Pichu','- Splash').replace('Ability: Torrent','Ability: Static')
        rpc.call(op='init',format=FORMAT,seed=[1,2,3,4],p1team=ours,p2team=theirs)
        reply=rpc.call(op='step',choices=[
            [('move',0,0,-1,-1),('move',1,0,-1,-1)],
            [('move',0,0,-1,-1),('move',1,0,-1,-1)]])
        assert not reply['errors']
        # If only Pichu fainted, slot 1 must remain pass, not have slot 0's
        # switch candidates relabelled by getChoiceIndex auto-skipping.
        legal=rpc.call(op='legal')['legal']
        assert ['switch',0,2,-1,-1] in legal[1][0]
        assert legal[0]==[[['pass',0,0,-1,-1]],[['pass',1,0,-1,-1]]]
    finally:rpc.close()
