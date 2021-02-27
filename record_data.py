import gym
import random
import numpy as np
import os
from datetime import datetime
import pickle

env_name = "CarRacing-v0"
env = gym.make(env_name)


if __name__ == "__main__":
  from pyglet.window import key

  a = np.array([0.0, 0.0, 0.0])
  
  path_record = "./data"
  filename = "expert_data_human_"+datetime.now().strftime("%Y%m%d%H%M%S")+".pkl"
  if not os.path.exists(path_record):
    os.mkdir(path_record)

  def key_press(k, mod):
    global restart
    if k == 0xFF0D:
        restart = True
    if k == key.LEFT:
        a[0] = -0.2
    if k == key.RIGHT:
        a[0] = +0.2
    if k == key.UP:
        a[1] = +0.1
    if k == key.DOWN:
        a[2] = +0.08  # set 1.0 for wheels to block to zero rotation

  def key_release(k, mod):
    if k == key.LEFT and a[0] == -0.2:
        a[0] = 0
    if k == key.RIGHT and a[0] == +0.2:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0

  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  isopen = True
  iscarstarted = False
  while isopen:
    samples = []
    s = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s_new, r, done, info = env.step(a)
      total_reward += r
      if steps % 200 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      if a[1]>0:
        iscarstarted = True
      if iscarstarted:
        samples.append((s, r, a))
      s = np.array(s_new)
      isopen = env.render()
      if done or restart or isopen == False:
        break
    with open(os.path.join(path_record,filename), 'ab') as fp:
      pickle.dump(samples, fp)
  env.close()