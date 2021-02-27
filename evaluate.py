import gym
from Models import ConvVAEWrapper, ResNetWrapper
from Utils import process_frame
import numpy as np
import random

def simulate(env, model, render_mode=True, num_episode=5, seed=-1):

  reward_list = []
  t_list = []
  crop = (0, 84, 0, 96)
  size = (64, 64)

  max_episode_length = 1000

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):

    obs = env.reset()

    total_reward = 0.0
    '''
    random_generated_int = np.random.randint(2**31-1)

    filename = "record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_action = []
    '''
    recording_reward = [0]
    
    for t in range(max_episode_length):

      if render_mode:
        env.render("human")
      else:
        env.render('rgb_array')

      #z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(process_frame(obs, crop=crop, size=size))

      '''
      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)
      '''
      obs, reward, done, info = env.step(action)

      recording_reward.append(reward)

      total_reward += reward

      if done:
        break

    #for recording:
    '''
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    
    if not render_mode:
      if recording_mode:
        np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)
    '''
    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list
  
if __name__=="__main__":

  seed = np.random.randint(10000)+10000 # Out of set of seeds used for data generation
  do_load_model = True
  num_episode = 1
  args_resnet = {'num_epochs': 1,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'grad_clip': 0.1,
        'steps_per_epoch': 1,
        'do_use_cuda': True,
        'num_classes': 3,
        'in_channels' : 3,
        'num_channels': 64,
        'shortcut': 'conv',
        'num_res_blocks': [3,3,3],
        'checkpoint_folder': 'checkpoint',
        'checkpoint_file': 'checkpoint.resnet.tar',
       }
       
  env_name = "CarRacing-v0"
  env = gym.make(env_name)
  
  resnet18 = ResNetWrapper(args_resnet)
  
  if do_load_model:
    resnet18.load_checkpoint(folder=args_resnet['checkpoint_folder'], filename=args_resnet['checkpoint_file'])
    
  reward_list, t_list = simulate(env, resnet18, render_mode=True, num_episode=num_episode, seed=seed)
  
  print(reward_list)
  print(t_list)
    
  