from utils import Params

from gym_ww import ww, vil
from gym_ww.envs.WwEnv import WwEnv

CONFIGS = dict(

    existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
    num_players=Params.num_player,
    penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=0,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
        # a penalty equal to index_of(agent2,targets)*penalty
        trg_accord=-1,

        # Penalty for a wasted voting round, i.e. the agents didn't reach the required threshold of X%
        wasted_round=-5,

    ),
    max_days=Params.max_days,

    # signal is used in the communication phase to signal other agents about intentions
    # the length concerns the dimension of the signal while the components is the range of values it can fall into
    # a range value of 2 is equal to binary variable
    signal_length=Params.signal_length,
    signal_range=Params.signal_range,

    # Multi communication round config
    num_rounds=Params.num_rounds,
    req_threshold=Params.req_threshold,

)
# CONFIGS['role2id'], CONFIGS['id2role'] = str_id_map(CONFIGS['existing_roles'])
