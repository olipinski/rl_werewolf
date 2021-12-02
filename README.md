# WereWolf Game

[Werewolf](https://en.wikipedia.org/wiki/Werewolf_social_deduction_game) is a simple deduction game that can be played
with at least 5 players. It is also knows as:

- Mafia (Mafia, International)
- Lupus in fabula (Wolf from fable, Latin)
- Pueblo duerme (Sleeping villagers, Spain)
- Los Hombres Lobo de Castronegro (The Werewolves of Castronegro, Spain)
- Μια Νύχτα στο Palermo (One night in Palermo, Greece)
- Městečko palermo (Town of Palermo, Czechia)
- 狼人殺 (Werewolf Kill, China)
- Libahunt (Werewolf , Estonia)
- Loup Garous (Werewolves, France)
- Werewölfe (Werewolves, Germany)
- Weerwolven (Werewolves, Netherlands)

In its most basic version there are __villagers__ (aka. vil) and __werewolves__ (aka. ww). Notice that the number of
wolves should always be less than the number of vil.

The game develops into tho phases, _night_ and _day_.

### Night

At _night_ time everyone closes their eyes, this prevents players to know which roles are assigned to other player.
Taking turns each non vil player open his eyes and choose an action. When only ww are present they open their eyes and
choose someone to eat.

### Day

During the day everyone open their eyes, assert the events of the night before (eaten players) and decide who is to be
executed. Here wolves have to be smart not to get catch and executed, to do so they lie.

### Game over

The game ends when either there are no more werewolves alive or there are more werewolves than villagers.

# Installation

Our Python version at the time of developing this code was 3.9. Any other Python version may not work correctly.

To install the dependencies for this code, run "pip install -r requirements.txt" in a preferred Python environment (
virtual or non-virtual)

Once installed, the training can be run with "python -m rlwolf.trainable.train {command line options}", where command
line options are explained in the train.py file, or additionally through the use of "--help".

## Helpful Links

### Custom gym env

- [basics](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)
- [examples](https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai)
- [Tutorial](https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/)

#### Multi agent

- [MA obs/action spaces utils](https://github.com/koulanurag/ma-gym/tree/master/ma_gym/envs/utils)
- [Discussion on ma openAi](https://github.com/openai/gym/issues/934)

##### Ray/RLlib

- [Ray Example](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)
- [multi-agent-and-hierarchical](https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
- [Docs](https://ray.readthedocs.io/en/latest/index.html)
- [Model configs](https://ray.readthedocs.io/en/latest/rllib-models.html#built-in-model-parameters)
- [Common config](https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters)
- [SelfPlay](https://github.com/ray-project/ray/issues/6669)
- [PPO Configs](https://github.com/ray-project/ray/blob/4633d81c390fd33d54aa62a5eb43fe104062bb41/rllib/agents/ppo/ppo.py#L19)
- [Understanding of ppo plots](https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2)

### RL frameworks

- [Comparison between rl framework](https://winderresearch.com/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/)
