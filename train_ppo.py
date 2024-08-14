'''
@Author: WANG Maonan
@Date: 2024-03-23 01:06:18
@Description: 不使用数据增强进行训练
@LastEditTime: 2024-03-26 22:09:48
'''
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
import sys
import gymnasium
from PPOBuild import PPO
from PPONetwork import FeedForwardNN
from model_structures.scnn import SCNN
from model_structures.eattention import EAttention
from sumo_env.make_tsc_env import make_env
from utils.lr_schedule import linear_schedule
from sumo_datasets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG  # 训练路网的信息
from eval_policy import eval_policy
logger.remove()
path_convert = get_abs_path(__file__)


# set_logger(path_convert('./'), log_level="INFO")

def create_env(params):
    env = make_env( **params)
    return env


def train(env, hyperparameters, actor_model, critic_model):
    
    
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """    
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
    
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, 1) #这里为了能使action为0，1，把输出直接改成了1

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)



def main():
    
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line
            没有使用args，直接训练

        Return:
            None
    """
    IS_DATA_AUG = False  # 是否使用数据增强

    log_path = path_convert('./log/')
    model_path = path_convert('./save_models/')
    tensorboard_path = path_convert('./tensorboard/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': True,
                'render_every_i': 10
              }

    # Define the parameters for the environment creation
    FOLDER_NAME = 'train_four_3'
    params = {
        'root_folder': path_convert(f"./sumo_datasets/"),
        'init_config': {
            'tls_id': SUMO_CONFIG[FOLDER_NAME]['tls_id'],
            'sumocfg': path_convert(f"./sumo_datasets/{FOLDER_NAME}/env/{SUMO_CONFIG[FOLDER_NAME]['sumocfg']}")
        },
        'env_dict': SUMO_CONFIG,
        'num_seconds': 3600,
        'use_gui': False,
        'log_file': log_path,
        'is_data_aug': IS_DATA_AUG,
        'env_index' : 0,
    }
    env = create_env(params)

    # Train or test, depending on the mode specified

    train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')


if __name__ == '__main__':
    main()