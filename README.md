# Turing Test via Emergent Communication in the Game of Werewolf

This is the repo containing the code used in our paper titled "Turing Test via Emergent Communication in the Game of Werewolf" accepted at EmeCom 2022. It is based on the code and work by Brandizzi et al. available at this [link](https://github.com/nicofirst1/rl_werewolf).

For details of our modifications and code we refer to our paper on [OpenReview](https://openreview.net/forum?id=B4xM-Qb0mbq)

## Install

To install follow the instructions in the [Installation](Resources/MarkDowns/Installation.md) markdown.

## Run

To run you will have to move the script `train.sh` from the `scripts` dir into the root of this repo.

Parameters such as the number of rounds or the required threshold can be adjusted in [train_appo.py](https://github.com/olipinski/rl_werewolf/blob/multi-round/src/utils/Params.py)
