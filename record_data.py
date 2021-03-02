import gym
import random
import numpy as np
import os
from datetime import datetime
import pickle
from Globals import *

env_name = "CarRacing-v0"
env = gym.make(env_name)


if __name__ == "__main__":
  from pyglet.window import key

  a = np.array([0.0, 0.0, 0.0, 0.0])
  
  path_record = "./data"
  filename = "expert_data_human_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
  if not os.path.exists(path_record):
    os.mkdir(path_record)
  
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

  num_episodes = 1
  for _ in range(num_episodes):
    iscarstarted = False
    samples = []
    s = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      #print(a)
      s_new, r, done, info = env.step(a)
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
      with open(os.path.join(path_record,filename), 'ab') as fp:
        pickle.dump(samples, fp)
  
  env.close()
