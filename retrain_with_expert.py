import gym
from Models import ConvVAEWrapper, LiNetWrapper
from Utils import process_frame, parseArguments, getValueFromDict
import numpy as np
import random
import os
import sys
from Globals import *
from pyglet.window import key
from torch.utils.data import DataLoader
from Datasets import CarRacingDataset
from datetime import datetime
import pickle

user_action = np.array([0.0, 0.0, 0.0])

def key_press(k, mod):
  global restart
  if k == 0xFF0D:
      restart = True
  if k == key.LEFT:
      user_action[0] = -GAIN_STEERING
  if k == key.RIGHT:
      user_action[0] = +GAIN_STEERING
  if k == key.UP:
      user_action[1] = +GAIN_THROTTLE
  if k == key.DOWN:
      user_action[2] = +GAIN_BRAKE  # set 1.0 for wheels to block to zero rotation

def key_release(k, mod):
  if k == key.LEFT and user_action[0] == -GAIN_STEERING:
      user_action[0] = 0
  if k == key.RIGHT and user_action[0] == +GAIN_STEERING:
      user_action[0] = 0
  if k == key.UP:
      user_action[1] = 0
      gas = 0
  if k == key.DOWN:
      user_action[2] = 0

def run_episode(env, encoder, model, render_mode=True, num_episode=5, seed=-1):

  reward_list = []
  t_list = []
  crop = (IM_CROP_YMIN, IM_CROP_YMAX, IM_CROP_XMIN, IM_CROP_XMAX)
  size = (IM_HEIGHT, IM_WIDTH)
  max_episode_length = 1000

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  corrective_samples = []
  for episode in range(num_episode):

    obs = env.reset()
    total_reward = 0.0
    for t in range(max_episode_length):

      if render_mode:
        env.render("human")
      
      sample = None
      if np.sum(np.abs(user_action))>0:
        action = np.array(user_action)
        sample = [obs, 0, np.array(action)]
      else:
        action = model.get_action(encoder, process_frame(obs, crop=crop, size=size))
      obs, reward, done, info = env.step(action)
      
      if sample is not None:
        sample[1] = reward
        corrective_samples.append(tuple(sample))
        
      total_reward += reward

      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list, corrective_samples
  
if __name__=="__main__":

  cmd_args = parseArguments(sys.argv[1:])
  num_episode = int(getValueFromDict(cmd_args, 'num_episode', 1))
  render_mode = str(getValueFromDict(cmd_args, 'render_mode', "True")).lower() in ["true", "1"]
  model_file = str(getValueFromDict(cmd_args, 'model_file', ""))
  vae_model_file = str(getValueFromDict(cmd_args, 'vae_model_file', ""))
  num_epochs_pn = int(getValueFromDict(cmd_args, 'num_epochs', 0))
  checkpoint_folder = str(getValueFromDict(cmd_args, 'checkpoint_folder', "./checkpoint"))
  do_save_checkpoints = str(getValueFromDict(cmd_args, 'do_save_checkpoints', "True")).lower() in ["true", "1"]
  min_num_corrective_samples = 0
  corrective_data_filename = "corrective_data_human_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
  data_folder = str(getValueFromDict(cmd_args, 'data_folder', ""))
  
  if model_file=="":
    print("model_file argument is necessary!")
    sys.exit(2)
  checkpoint_folder, checkpoint_file = os.path.split(model_file) 
    
  if vae_model_file=="":
    print("vae_model_file argument is necessary!")
    sys.exit(2)   
  vae_checkpoint_folder, vae_checkpoint_file = os.path.split(vae_model_file) 

  if data_folder=="":
    print("data_folder argument is necessary!")
    sys.exit(2)
  if not os.path.exists(data_folder):
    os.mkdir(data_folder)
  filename = "expert_data_human_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
    
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
  args_pn = {'num_epochs': num_epochs_pn,
             'lr': 0.001,
             'grad_clip': 0.1,
             'do_use_cuda': True,
             'num_classes': NUM_DISCRETE_ACTIONS,
             'in_channels' : args_vae['num_latent_features'],
             'num_channels': 64,
             'category_weights': None,
            }
       
  env_name = "CarRacing-v0"
  env = gym.make(env_name)
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  
  policy_network = LiNetWrapper(args_pn)  
  policy_network.load_checkpoint(folder=checkpoint_folder, filename=checkpoint_file)

  vae = ConvVAEWrapper(args_vae)
  vae.load_checkpoint(vae_checkpoint_folder, vae_checkpoint_file)
  
  reward_list, t_list, corrective_samples = run_episode(env, vae, policy_network, 
    render_mode=render_mode, num_episode=num_episode, seed=seed)
  
  print(f"Average reward for {num_episode} evaluation(s) : {np.mean(reward_list)}")
  
  if len(corrective_samples)>min_num_corrective_samples:
    with open(os.path.join(data_folder, corrective_data_filename), 'ab') as fp:
      pickle.dump(corrective_samples, fp)
    
    if num_epochs_pn>0:
      # Train
      batch_size = 128
      dataset = CarRacingDataset(corrective_samples, action_space='discrete', num_skip_frames=0)
      train_loader = DataLoader(dataset, batch_size=min(batch_size,len(dataset)), shuffle=True)

      print("Fine tuning PN")
      
      for epoch in range(num_epochs_pn):
        
        print("Epoch: ", epoch)
        
        train_loss = policy_network.train_epoch(train_loader, vae)
        
        # Save checkpoint
        if do_save_checkpoints:
          checkpoint_filename = 'checkpoint.policy.ft.epoch.'+str(epoch)+'.tar'
          policy_network.save_checkpoint(folder=checkpoint_folder, filename=checkpoint_filename)
  else:
    print("Not enough corrective samples collected, skipping training")

  