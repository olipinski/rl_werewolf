import argparse
import inspect
import json
import multiprocessing
import os
import shutil
import sys
import uuid

import tensorflow as tf
import termcolor


def join_paths(path1, path2):
    return os.path.join(path1, path2)


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


def initialize_dirs(dirs: list) -> None:
    for init_dir in dirs:
        if not os.path.exists(init_dir):
            print(f"Mkdir {init_dir}")
            os.makedirs(init_dir)


def empty_dirs(to_empty: list) -> None:
    for folder in to_empty:
        # Check if folder exists first
        if os.path.isdir(folder):
            # Proceed to clean
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)


#@singleton
class Params:
    ##########################
    # other
    ##########################

    unique_id = str(uuid.uuid1())[:8]

    ##########################
    #       Path params
    ##########################

    WORKING_DIR = os.getcwd().split("src")[0]
    SRC_DIR = join_paths(WORKING_DIR, "src")

    LOG_DIR = join_paths(WORKING_DIR, "log_dir")

    RAY_DIR = join_paths(LOG_DIR, f"ray_results_{unique_id}")
    GAME_LOG_DIR = join_paths(LOG_DIR, f"match_log_{unique_id}")
    EVAL_DIR = join_paths(LOG_DIR, f"eval_{unique_id}")

    episode_file = join_paths(EVAL_DIR, f"episode_{unique_id}.pkl")
    log_match_file = join_paths(GAME_LOG_DIR, f"matchlog_{unique_id}.log")
    params_file = join_paths(GAME_LOG_DIR, f"params_{unique_id}.log")

    ##########################
    # Performance stuff
    ##########################
    debug = False


    n_cpus = multiprocessing.cpu_count() if not debug else 1
    n_gpus = 1 if not debug and tf.test.is_gpu_available() else 0
    n_workers = 7 if not debug else 1

    ##########################
    # Evaluation params
    ##########################
    checkpoint_freq = 50
    log_step = 500
    max_checkpoint_keep = 10
    resume_training = False
    alternating=False

    ##########################
    # env params
    ##########################
    # check
    num_player = 9

    # maximum number of day before a match forcefully ends
    max_days = 10

    # signal is used in the communication phase to signal other agents about intentions
    # the length concerns the dimension of the signal while the components is the range of values it can fall into
    # a range value of 2 is equal to binary variable
    signal_length = 0
    signal_range = 9

    # Multi-round params
    # Number of rounds allowed before forcibly ending comm phase
    num_rounds = 3
    # Threshold required for the vote to pass
    req_threshold = 0.8

    ##########################
    #    METHODS
    ##########################

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        att = self.__get_attributes()

        """Create the parser to capture CLI arguments."""
        parser = argparse.ArgumentParser()

        # for every attribute add an arg instance
        for k, v in att.items():
            parser.add_argument(
                "--" + k.lower(), type=type(v), default=v,

            )

        for k, v in vars(parser.parse_args()).items():
            self.__setattr__(k, v)

    def __init__(self):
        print("Params class initialized")

        if not self.resume_training:
            empty_dirs([self.LOG_DIR])

        initialize_dirs([self.LOG_DIR, self.RAY_DIR, self.GAME_LOG_DIR, self.EVAL_DIR])

        # change values based on argparse
        self.__parse_args()

        if not self.resume_training:
            # log params to file and out
            with open(self.params_file, "w+") as f:
                self.__log_params([sys.stdout, f])

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def __log_params(self, out=sys.stdout):
        """
        Prints attributes as key value on given output
        :param out: the output for printing, default stdout
        :return:
        """

        # initializing print message
        hashes = f"\n{20 * '#'}\n"
        msg = f"{hashes} PARAMETER START {hashes}"

        # get the attributes ad dict
        attributes = self.__get_attributes()
        # dump using jason
        attributes = json.dumps(attributes, indent=4, sort_keys=True)

        msg += attributes
        msg += f"{hashes} PARAMETER END {hashes}"

        color = "yellow"
        msg = termcolor.colored(msg, color=color)

        if not isinstance(out, list):
            out = [out]

        # print them to given out
        for sts in out:
            print(msg, file=sts)
