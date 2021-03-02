import gym
import random
import numpy as np
import os
from datetime import datetime
import pickle
from Globals import *
from Utils import parseArguments, getValueFromDict
from pyglet.window import key

env_name = "CarRacing-v0"
env = gym.make(env_name)

if __name__ == "__main__":

  cmd_args = parseArguments(sys.argv[1:])
  data_folder = str(getValueFromDict(cmd_args, 'data_folder', './data'))
  num_episode = int(getValueFromDict(cmd_args, 'num_episode', 1))

  a = np.array([0.0, 0.0, 0.0])
  
  filename = "expert_data_human_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
  if not os.path.exists(data_folder):
    os.mkdir(data_folder)
  
  def key_press(k, mod):
    global restart
    if k == 0xFF0D:
        restart = True
    if k == key.LEFT:
        a[0] = -GAIN_STEERING
    if k == key.RIGHT:
        a[0] = +GAIN_STEERING
    if k == key.UP:
        a[1] = +GAIN_THROTTLE
    if k == key.DOWN:
        a[2] = +GAIN_BRAKE  # set 1.0 for wheels to block to zero rotation

  def key_release(k, mod):
    if k == key.LEFT and a[0] == -GAIN_STEERING:
        a[0] = 0
    if k == key.RIGHT and a[0] == +GAIN_STEERING:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
        gas = 0
    if k == key.DOWN:
        a[2] = 0
  
  seed = np.random.randint(10000)
  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  for _ in range(num_episode):
    iscarstarted = False
    samples = []
    s = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s_new, r, done, info = env.step(a)
      #print('action :', a)
      total_reward += r
      steps += 1
      if a[1]>0:
        iscarstarted = True
      if iscarstarted:
        samples.append((s, r, np.array(a)))
      s = np.array(s_new)
      env.render()
      if done or restart:
        break
    
    print(f"You scored {total_reward}...")
    user_input = input("Do you want to save your trajectory?")
    if user_input.lower() in ["y", "yes"]:
      with open(os.path.join(data_folder,filename), 'ab') as fp:
        pickle.dump(samples, fp)
  
  env.close()
