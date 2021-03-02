import gym
from Models import ConvVAEWrapper, LiNetWrapper
from Utils import process_frame, parseArguments, getValueFromDict
import numpy as np
import random
import os
import sys
import pickle
from datetime import datetime
from Globals import *

def run_episodes(env, encoder, model, render_mode=True, num_episode=5, seed=-1, record_folder=''):

  reward_list = []
  t_list = []
  crop = (IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX)
  size = (IM_HEIGHT, IM_WIDTH)
  max_episode_length = MAX_EPISODE_LENGTH
  do_record = record_folder!=''

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
  
  samples = []
  for episode in range(num_episode):

    state = env.reset()
    total_reward = 0.0
    samples_episode = []
    for t in range(max_episode_length):

      if render_mode:
        env.render("human")
      
      action = model.get_action(encoder, process_frame(state, crop=crop, size=size))
      next_state, reward, done, info = env.step(action)
      
      if do_record:
        samples_episode.append((state, reward, action))
        
      total_reward += reward
      state = next_state
      
      if done:
        break

    print(f"Episode {episode}, total reward: {total_reward}")
    
    samples.extend(samples_episode)
    reward_list.append(total_reward)
    t_list.append(t)

  if do_record:
    filename = "eval_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
    if not os.path.exists(record_folder):
      os.mkdir(record_folder)
    with open(os.path.join(record_folder, filename), 'ab') as fp:
      pickle.dump(samples, fp)
    
  return reward_list, t_list
  
if __name__=="__main__":

  cmd_args = parseArguments(sys.argv[1:])
  num_episode = int(getValueFromDict(cmd_args, 'num_episode', 1))
  render_mode = str(getValueFromDict(cmd_args, 'render_mode', 'True')).lower() in ['true', '1']
  num_frames_override = int(getValueFromDict(cmd_args, 'num_frames_override', 20))
  model_file = str(getValueFromDict(cmd_args, 'model_file', ''))
  vae_model_file = str(getValueFromDict(cmd_args, 'vae_model_file', ''))
  epsilon = float(getValueFromDict(cmd_args, 'epsilon', 0.0))
  record_folder = str(getValueFromDict(cmd_args, 'data_folder', ''))

  if model_file=="":
    print("model_file argument is necessary!")
    sys.exit(2)
  checkpoint_folder, checkpoint_file = os.path.split(model_file) 

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
             }
             
  # Policy network arguments
  args_pn = {'num_epochs': 1,
             'do_use_cuda': True,
             'num_classes': NUM_DISCRETE_ACTIONS,
             'in_channels' : args_vae['num_latent_features'],
             'num_channels': 64,
             'epsilon': epsilon,
            }
       
  env_name = "CarRacing-v0"
  env = gym.make(env_name)
  
  policy_network = LiNetWrapper(args_pn)  
  policy_network.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)

  vae = ConvVAEWrapper(args_vae)
  vae.load_checkpoint(vae_checkpoint_folder, vae_checkpoint_file)
  
  reward_list, t_list = run_episodes(env, vae, policy_network, render_mode=render_mode, 
    num_episode=num_episode, seed=seed, record_folder=record_folder)
  
  print(f"Mean/std reward for {num_episode} evaluation(s) : {np.mean(reward_list)} {np.std(reward_list)}")
  