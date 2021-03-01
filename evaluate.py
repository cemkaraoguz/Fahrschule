import gym
from Models import ConvVAEWrapper, LiNetWrapper
from Utils import process_frame, parseArguments, getValueFromDict
import numpy as np
import random
import os
import sys
from Globals import *

def run_episode(env, encoder, model, render_mode=True, num_episode=5, seed=-1, num_frames_override=20):

  reward_list = []
  t_list = []
  crop = (IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX)
  size = (IM_HEIGHT, IM_WIDTH)
  max_episode_length = 1000

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):

    obs = env.reset()
    total_reward = 0.0
    for t in range(max_episode_length):

      if render_mode:
        env.render("human")
      
      if t<num_frames_override:
        action = ACTION_MAPPING[1,:]
      else:
        action = model.get_action(encoder, process_frame(obs, crop=crop, size=size))
      obs, reward, done, info = env.step(action)

      total_reward += reward

      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list
  
if __name__=="__main__":

  cmd_args = parseArguments(sys.argv[1:])
  num_episode = int(getValueFromDict(cmd_args, 'num_episode', 1))
  render_mode = str(getValueFromDict(cmd_args, 'render_mode', "True")).lower() in ["true", "1"]
  num_frames_override = int(getValueFromDict(cmd_args, 'num_frames_override', 20))
  model_file = str(getValueFromDict(cmd_args, 'model_file', ""))

  if model_file=="":
    print("model_file argument is necessary!")
    sys.exit(2)
  checkpoint_folder, checkpoint_file = os.path.split(model_file) 

  vae_model_file = str(getValueFromDict(cmd_args, 'vae_model_file', ""))
  if vae_model_file=="":
    print("vae_model_file argument is necessary!")
    sys.exit(2)   
  vae_checkpoint_folder, vae_checkpoint_file = os.path.split(vae_model_file) 

  seed = np.random.randint(10000)+10000 # Out of set of seeds used for data generation
  # Variational Autoencoder arguments
  args_vae = {'in_channels': IM_CHANNELS,
              'rows' : IM_HEIGHT,
              'cols' : IM_WIDTH,
              'num_hidden_features': [32, 64, 128, 256],
              'num_latent_features': 32,
              'strides': [2, 2, 2, 2],
              'do_use_cuda': True,
              'in_features': IM_HEIGHT*IM_WIDTH,
         }
  # Policy network arguments
  args_pn = {'num_epochs': 1,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'grad_clip': 0.1,
        'steps_per_epoch': None,
        'do_use_cuda': True,
        'num_classes': NUM_DISCRETE_ACTIONS,
        'in_channels' : args_vae['num_latent_features'],
        'num_channels': 64,
        'category_weights': None,
       }
       
  env_name = "CarRacing-v0"
  env = gym.make(env_name)
  
  policy_network = LiNetWrapper(args_pn)  
  policy_network.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)

  vae = ConvVAEWrapper(args_vae)
  vae.load_checkpoint(vae_checkpoint_folder, vae_checkpoint_file)
  
  reward_list, t_list = run_episode(env, vae, policy_network, render_mode=render_mode, 
    num_episode=num_episode, seed=seed, num_frames_override=num_frames_override)
  
  print(f"Average reward for {num_episode} evaluation(s) : {np.mean(reward_list)}")
  