import math
import random

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

import gym_ww
from src.other.custom_utils import most_frequent

####################
# names for roles
####################
ww = gym_ww.ww
vil = gym_ww.vil


class WwEnv(MultiAgentEnv):

    def __init__(self, configs, ww_num=None):
        """

        :param ww_num: int, number of werewolves

        """

        # if config is dict
        if isinstance(configs, EnvContext) or isinstance(configs, dict):
            # get num player
            try:
                num_players = configs['num_players']
            except KeyError:
                raise AttributeError(f"Attribute 'num_players' should be present in the EnvContext")

        elif isinstance(configs, int):
            # used for back compatibility
            num_players = configs
        else:
            raise AttributeError(f"Type {type(configs)} is invalid for config")

        # number of player should be more than 5
        assert num_players >= 5, "Number of player should be >= 5"

        if ww_num is None:
            # number of wolves should be less than villagers
            num_wolves = math.floor(math.sqrt(num_players))
            num_villagers = num_players - num_wolves
            # random.shuffle(roles)

        else:
            assert ww_num < num_players, f"The number of werewolves  should be less than " \
                                         f"the number of players ({num_players}) "
            num_wolves = ww_num
            num_villagers = num_players - num_wolves

        roles = [ww] * num_wolves + [vil] * num_villagers

        self.num_players = num_players
        self.num_wolves = num_wolves
        self.roles = roles
        self.penalties = configs['penalties']
        self.max_days = configs['max_days']

        assert configs['signal_length'] <= num_players, "Signal length must be not greater than the number of players"

        self.signal_length = configs['signal_length']
        self.signal_range = configs['signal_range']

        # used for logging game
        self.ep_step = 0

        # define empty attributes, refer to initialize method for more info
        self.status_map = None
        self.shuffle_map = None
        self.unshuffle_map = None
        self.is_night = True
        self.is_comm = True
        self.day_count = 0
        self.phase = 0
        self.is_done = False
        self.custom_metrics = None
        self.role_map = None
        self.just_died = None
        self.initialize()

    #######################################
    #       INITIALIZATION
    #######################################

    def initialize(self):
        """
        Initialize attributes for new run
        :return:
        """

        self.role_map = {idx: self.roles[idx] for idx in range(self.num_players)}

        # map to shuffle player ids at the start of each game, check the readme under PolicyWw for more info
        sh = sorted(range(self.num_players), key=lambda k: random.random())
        self.shuffle_map = {idx: sh[idx] for idx in range(self.num_players)}
        self.unshuffle_map = {sh[idx]: idx for idx in range(self.num_players)}

        # list for agent status (dead=0, alive=1)
        self.status_map = [1 for _ in range(self.num_players)]

        # bool flag to keep track of turns
        self.is_night = True

        # first phase is communication night phase
        self.is_comm = True

        # reset is done
        self.is_done = False

        # reset day
        self.day_count = 0

        # rest just died
        self.just_died = None

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

            Returns:
                observation (object): the initial observation.
            """

        self.initialize()
        init_signal = {p: [-1] * self.signal_length for p in range(self.num_players)}
        obs = self.observe(phase=0, signal=init_signal, targets={k: -1 for k in range(self.num_players)})
        obs, _, _, _ = self.convert(obs, {}, {}, {}, 0)
        return obs

    #######################################
    #       MAIN CORE
    #######################################

    def day(self, actions, rewards):
        """
        Run the day phase, that is execute target based on votes and reward accordingly or the voting
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def execution(actions_exec, rewards_exec):
            """
            To be called when is execution phase
            :return:
            """

            # get the agent to be executed
            target = most_frequent(actions_exec)

            # penalize for non divergent target
            rewards_exec = self.target_accord(target, rewards_exec, actions_exec)

            # penalize target agent
            rewards_exec[target] += self.penalties.get("death")

            # kill him
            self.status_map[target] = 0
            self.just_died = target

            # update day
            self.day_count += 1

            return rewards_exec

        # call the appropriate method depending on the phase
        if self.is_comm:
            return rewards
        else:
            rewards = {id_: val + self.penalties.get('day') for id_, val in rewards.items()}
            return execution(actions, rewards)

    def night(self, actions, rewards):
        """
        Is night, time to perform actions!
        During this phase, villagers action are not considered
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: return updated rewards
        """

        if not self.is_comm:
            # execute wolf actions
            rewards = self.wolf_action(actions, rewards)

        return rewards

    def step(self, actions_dict):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            actions_dict (dict): a list of action provided by the agents

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # remove roles from ids
        actions_dict = {int(k.split("_")[1]): v for k, v in actions_dict.items()}

        signals, targets = self.split_target_signal(actions_dict)

        # rewards start from zero
        rewards = {id_: 0 for id_ in self.get_ids("all", alive=False)}

        # execute night action
        if self.is_night:
            rewards = self.night(targets, rewards)
        else:  # else go with day
            # apply action by day
            rewards = self.day(targets, rewards)

        # prepare for phase shifting
        is_night, is_comm, phase = self.update_phase()

        # get dones
        dones, rewards = self.check_done(rewards)
        # get observation
        obs = self.observe(phase, signals, targets)

        # initialize infos with dict
        infos = {idx: {'role': self.roles[idx]} for idx in self.get_ids("all", alive=False)}

        # convert to return in correct format, do not modify anything except for dones
        obs, rewards, dones, info = self.convert(obs, rewards, dones, infos, phase)

        # if game over reset
        if self.is_done:

            dones["__all__"] = True
            # normalize infos
        else:
            dones["__all__"] = False

        # shift phase
        self.is_night = is_night
        self.is_comm = is_comm

        return obs, rewards, dones, info

    def wolf_action(self, actions, rewards):
        """
        Perform wolf action, that is kill agent based on votes and reward
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def kill(actions_kill, rewards_kill):

            if not len(wolves_ids):
                raise Exception("Game not done but wolves are dead, have reset been called?")

            # get agent to be eaten
            target = most_frequent(actions_kill)

            # penalize for different ids
            rewards_kill = self.target_accord(target, rewards_kill, actions_kill)

            # kill agent and remember
            self.status_map[target] = 0
            self.just_died = target

            # penalize dead player
            rewards_kill[target] += self.penalties.get("death")

            return rewards_kill

        wolves_ids = self.get_ids(ww)
        # filter action to get only wolves
        actions = {k: v for k, v in actions.items() if k in wolves_ids}

        # call the appropriate method depending on the phase
        if self.is_comm:
            return rewards
        else:
            return kill(actions, rewards)

    #######################################
    #       UPDATER
    #######################################

    def update_phase(self):
        """
        Shift the phase to the next one, keep elif explicit for readability
        :return:
            night: bool, value of is_night
            comm: bool, value of is_com
            phase: int, value of phase
        """

        if self.is_night and self.is_comm:
            comm = False
            phase = 0
            night = True

        elif self.is_night and not self.is_comm:
            night = False
            comm = True
            phase = 1

        elif not self.is_night and self.is_comm:
            comm = False
            phase = 2
            night = False

        elif not self.is_night and not self.is_comm:
            night = True
            comm = True
            phase = 3

        else:
            raise ValueError("Something wrong when shifting phase")

        self.phase = phase
        return night, comm, phase

    #######################################
    #       UTILS
    #######################################

    def split_target_signal(self, actions_dict):
        """
        Split signal and target from the action dictionary
        :param actions_dict: dict[int->obj], map agent id to action
        :return: signals and target
        """

        # split signals from targets
        if self.signal_length > 0:
            signals = {k: v[1:] for k, v in actions_dict.items()}
            # remove signals from action dict
            targets = {k: v[0] for k, v in actions_dict.items()}
        else:
            signals = {}
            targets = actions_dict

        # apply unshuffle
        targets = {k: self.unshuffle_map[v] for k, v in targets.items()}

        return signals, targets

    def convert(self, obs, rewards, dones, info, phase):
        """
        Convert everything in correct format.
        1. Filter out vill when phase is 0
        2. Filter out dead agents if they did not just died (done in order to penalize just dead agents)
        3. add name to agents
        4. convert reward to floats
        """

        # remove villagers from night phase
        if phase == 0:
            rewards = {id_: val for id_, val in rewards.items() if id_ in self.get_ids(ww, alive=False)}
            obs = {id_: val for id_, val in obs.items() if id_ in self.get_ids(ww, alive=False)}
            info = {id_: val for id_, val in info.items() if id_ in self.get_ids(ww, alive=False)}

        # if the match is not done yet remove dead agents
        if not self.is_done:
            # filter out dead agents from rewards, not the one just died tho
            if phase in [1, 3]:
                rewards = {id_: val for id_, val in rewards.items() if
                           self.status_map[id_] or id_ == self.just_died}
                obs = {id_: val for id_, val in obs.items() if self.status_map[id_] or id_ == self.just_died}
                info = {id_: val for id_, val in info.items() if self.status_map[id_] or id_ == self.just_died}
            else:
                rewards = {id_: val for id_, val in rewards.items() if self.status_map[id_]}
                obs = {id_: val for id_, val in obs.items() if self.status_map[id_]}
                info = {id_: val for id_, val in info.items() if self.status_map[id_]}

        # add observation into info for policies
        for k, v in info.items():
            v.update(dict(obs=obs[k]))

        # add roles to ids for policy choosing
        rewards = {f"{self.roles[k]}_{k}": v for k, v in rewards.items()}
        obs = {f"{self.roles[k]}_{k}": v for k, v in obs.items()}
        dones = {f"{self.roles[k]}_{k}": v for k, v in dones.items()}
        info = {f"{self.roles[k]}_{k}": v for k, v in info.items()}

        # convert to float
        rewards = {k: float(v) for k, v in rewards.items()}

        return obs, rewards, dones, info

    def check_done(self, rewards):
        """
        Check if the game is over
        :param rewards: dict, maps agent id_ to curr reward
        :return:
            dones: list of bool statement
            rewards: update rewards
        """
        dones = {id_: False for id_ in rewards.keys()}

        # get list of alive agents
        alives = self.get_ids('all')

        # if there are more wolves than villagers than they won
        wolf_won = len(self.get_ids(ww)) >= len(self.get_ids(vil))
        # if there are no more wolves than the villager won
        village_won = all([role == vil for id_, role in self.role_map.items() if id_ in alives])

        if wolf_won:  # if wolves won
            # set flag to true (for reset)
            self.is_done = True
            # reward
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('lost')

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')

        if self.day_count >= self.max_days - 1:
            self.is_done = True

        return dones, rewards

    def get_ids(self, role, alive=True, include_just_died=False):
        """
        Return a list of ids given a role
        :param role: str, the role of the wanted ids
        :param alive: bool, if to get just alive players or everyone
        :param include_just_died: bool, if to include the player that just died, makes sense only if alive is True
        :return: list of ints
        """

        if role == "all":
            ids = list(self.role_map.keys())
        else:
            # get all the ids for a given role
            ids = [id_ for id_, rl in self.role_map.items() if rl == role]

        # filter out dead ones
        if alive:
            ids = [id_ for id_ in ids if self.status_map[id_]]
            if include_just_died:
                ids.append(self.just_died)

        return ids

    def target_accord(self, chosen_target, rewards, targets):
        """
        Reward/penalize agent based on the target chose for execution/kill depending on the choices it made.
        This kind of reward shaping is done in order for agents to output targets which are more likely to be chosen
        :param targets: dict[int->int], maps an agent to its target
        :param chosen_target: int, agent id_ chosen for execution/kill
        :param rewards: dict[int->int], map agent id_ to reward
        :return: updated rewards
        """

        for id_, vote in targets.items():
            # if the agent hasn't voted for the executed agent then it takes a penalty
            if vote != chosen_target:
                penalty = self.penalties["trg_accord"]
                rewards[id_] += penalty

        return rewards

    #######################################
    #       SPACES
    #######################################

    @property
    def action_space(self):
        """
        :return:
        """

        # the action space is made of two parts: the first element is the actual target they want to be executed
        # and the other ones are the signal space
        if self.signal_length > 0:
            space = gym.spaces.MultiDiscrete([self.num_players] * (1 + self.signal_length))
        else:
            space = gym.spaces.Discrete(self.num_players)
            space.nvec = [space.n]

        # high=[self.num_players]+[self.signal_range-1]*self.signal_length
        # low=[-1]+[0]*self.signal_length
        # space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int32)

        # should be a list of targets
        return space

    @property
    def observation_space(self):
        """
        Return observation space in gym box
        :return:
        """

        obs = dict(
            # number of days passed
            day=spaces.Discrete(self.max_days),
            # idx is agent id_, value is boll for agent alive
            status_map=spaces.MultiBinary(self.num_players),
            # number in range number of phases [com night, night, com day, day]
            phase=spaces.Discrete(4),
            # targets is now a vector, having an element outputted from each agent
            targets=gym.spaces.Box(low=-1, high=self.num_players, shape=(self.num_players,), dtype=np.int64),
            # own id
            own_id=gym.spaces.Discrete(self.num_players),

        )

        # add signal if the required
        if self.signal_length > 0:
            # signal is a matrix of dimension [num_player, signal_range]
            signal = dict(
                signal=gym.spaces.Box(low=-1, high=self.signal_range - 1, shape=(self.num_players, self.signal_length),
                                      dtype=np.int64))
            obs.update(signal)

        obs = gym.spaces.Dict(obs)

        return obs

    def observe(self, phase, signal, targets):
        """
        Return observation object
        :return:
        """

        def add_missing(signal_add, targets_add):
            """
            Add missing values (for dead agents) to targets and signal
            :param signal_add: ndarray, signal of size [num_player, signal_length]
            :param targets_add: dict[int->int], mapping agent ids to targets
            :return: tuple
                1: signal
                2: targets
            """

            # if the list of outputs is full then do nothing
            if len(targets_add) == self.num_players:
                return signal_add, targets_add

            # get indices to add
            to_add = set(range(self.num_players)) - set(targets_add.keys())

            # add a list of -1 of length signal_length to the signal_add
            sg_add = [-1] * self.signal_length

            # update dict with -1
            targets_add.update({elem: -1 for elem in to_add})

            if self.signal_length > 0:
                signal_add.update({elem: sg_add for elem in to_add})

            return signal_add, targets_add

        def shuffle_sort(dictionary, shuffle_map, value_too=True):
            """
            Shuffle a dictionary given a map
            @param dictionary: dict, dictionary to shuffle
            @param shuffle_map: dict, map
            @param value_too: bool, if to shuffle the value too
            @return: shuffled dictionary
            """

            new_dict = {}
            for k, v in dictionary.items():
                nk = shuffle_map[k]

                if value_too and v in shuffle_map.keys():
                    nv = shuffle_map[v]
                    new_dict[nk] = nv
                else:
                    new_dict[nk] = v

            new_dict = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[0])}

            return new_dict

        def get_targets_signal(signal_p, targets_p):
            """
            Given some initial values of signal_add and target perform:
            1. insertion of missing elements (dead players)
            2. shuffle of ids
            3. numpy stack
            """
            # add missing targets_add
            signal_p, targets_p = add_missing(signal_p, targets_p)

            # shuffle
            targets_p = shuffle_sort(targets_p, self.shuffle_map)
            signal_p = shuffle_sort(signal_p, self.shuffle_map, value_too=False)

            # stack observations
            # make matrix out of signals of size [num_player,signal_length]
            tg_p = np.asarray(list(targets_p.values()))
            if len(signal_p) > 0:
                sg_p = np.stack(list(signal_p.values()))
            else:
                sg_p = {}

            return tg_p, sg_p

        observations = {}

        tg, sg = get_targets_signal(signal, targets)

        # apply shuffle to status map
        st = [self.status_map[self.unshuffle_map[idx]] for idx in range(self.num_players)]

        # add observation for ww
        for idx in self.get_ids(ww, alive=False):
            # build obs dict
            obs = dict(
                day=self.day_count,  # day passed
                status_map=np.array(st),  # agent_id:alive?
                phase=phase,
                targets=tg,
                own_id=self.shuffle_map[idx],
            )

            if self.signal_length > 0:
                obs["signal_add"] = sg

            observations[idx] = obs

        # add observation for villagers
        # if the phase is 1 then the villagers are not allowed to see what the wolves voted so pad everything to -1

        if phase == 1:
            tg, sg = get_targets_signal({}, {})

        for idx in self.get_ids(vil, alive=False):
            # build obs dict
            obs = dict(
                day=self.day_count,  # day passed
                status_map=np.array(st),  # agent_id:alive?
                phase=phase,
                targets=tg,
                own_id=self.shuffle_map[idx],
            )

            if self.signal_length > 0:
                obs["signal_add"] = sg

            observations[idx] = obs

        return observations
