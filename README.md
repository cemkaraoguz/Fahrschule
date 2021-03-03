# Fahrschule
Learning driving from expert data

## Environment and dependencies
The code is developed/tested under Windows 10 with python 3.8.5. The following dependencies are required:
- pytorch 1.7.1
- opencv-python 4.5.1.48
- tqdm 4.56.0
- matplotlib 3.3.2
- gym 0.18.0 (with swig and Box2D)

## Problem setting
For this project, we are going to train an agent to play car racing on OpenAI gym's CarRacing simulator using Imitation Learning (IL). IL is a research area closely related to the Reinforcement Learning. In Reinforcement Learning an agent interacts with an environment by following a policy. The agent takes action based on the policy and the current state, and as a result, receives a reward and transitions to a new state. The goal is to learn an optimal policy which maximizes the long-term cumulative rewards. However, in some settings learning such policies might be challenging due to the complexity of the environment or we might not even have access to agent's policy to optimize it and we may still need to infer a useful policy from the observed data. In such settings, IL can provide us with a solution by allowing us to learn from observational data provided by an expert an expert (typically a human). In this project we are going to generate expert data by manually playing the car racing game and use this dataset to train an agent.

## Methods
A very basic approach to IL is a method called Behavioural Cloning which formulates the setting as a supervised learning problem: expert’s demonstrations are divided into state-action pairs and use these pairs as input/label fashion to train a model. The main drawback of this method is that it is prone to generalization errors especially if the agent finds itself in situations where it was not demonstrated before. Therefore, it is very important to use models with good generalization capabilities.

As the problem setting is not very complex (fairly simple state space without much variation, non-presence of other agents in the environment, low dimensional action space) we can apply this method for this project.

### Input space and state representation:
Input space of the model is 96x96 pixel bird's eye view image of the simulation environment as shown in the figure below:

<img src="Figures/simulator.jpg" width=500><br>

While several convolutional layers can already provide a robust state representation, a better approach can be Variational Autoencoders (VAE). The general idea of autoencoders is setting an encoder and a decoder as neural networks and learning the optimal encoding-decoding scheme. The encoder compresses the input in scale to a latent space and the decoder expands this latent space back to the original dimension. When output is enforced to be the same as the input through the loss function, this bottleneck ensures only the main structured part of the information can go through and be reconstructed. However, the latent space of conventional autoencoders lack regularity: neighbouring points in the latent space do not necessarily correspond to neighbouring states in the input space. VAEs solve this problem by enforcing such regularity in the latent space. This is especially important in our case since this bestows us good generalization capabilities: if the model encounters a state which it has not seen during training but it has seen similar states, it can still infer the right action to take. 

VAEs are used intensively in related works dealing with similar problems. It also allows further extensions, for example an action-conditioned state transition model can be plugged seamlessly as in [1] and [2] or it can be trained in a generative adverserial setting as in [3] if generating realistic samples for simulation is also an objective.

In this project a VAE will serve as a feature extractor and the latent layer of the VAE will be used as an input to the policy network. An architecture similar to [1] is also used here. As pre-processing, the bottom panel from the images is cropped out and the remaining part of the image is downscaled to 64x64 to have consistent strides in the encoder/decoder. Cropping out the bottom panel dicards some state information (current velocity/steering represented as bars) however, the VAE would probably get rid of this information anyway and it turns out this information is not really necessary for the current setting. The VAE is trained separately for 10 epochs and to verify its function, several example images (top) and their reconstructions (bottom) are visualized below:

<img src="Figures/VAE_inputs.png" width=700 caption="inputs"><br>

<img src="Figures/VAE_reconstructions.png" width=700><br>

As we can see, the VAE suppresses information with high-variance (e.g. grass) and retains vital task relevant information (e.g. road, car). 

### Action space:
The human control interface for generating expert data allows very few selection of actions (1 value for acceleration, brake, left turn and right turn). Here is an histogram of actions taken during 25 episodes of expert demonstrations:

<img src="Figures/ActionPreferences_human_cont.png" width=1000><br>

For comparison, here is a RL agent trained following the methods in [1]:

<img src="Figures/ActionPreferences_RLagent_cont.png" width=1000><br>

If we were using an AI as expert, we could choose continous action space (actually this was tested and works quite well). But in this case, discrete action space seems like a better option. 5 actions are defined: Steer Left, Steer Right, Accelerate, Brake, Straight (i.e. No Action). This implies that combination of actions are not regarded and in such cases the action is mapped to one of these 5 cases depending on pre-defined heuristics. This desing choice was simply due to author's personal experience on playing the game, usually a player only uses one action at a time. The following figure shows the histogram of action preferences from 25 expert demonstrations after mapping:

<img src="Figures/ActionPreferences_human.png" width=500><br>

### Policy network and the loss function:
Policy network takes the latent representation of the VAE and outputs the action to take. For this job, a simple fully connected neural network with one hidden layer is used. The above figure illustrating the expert action preferences indicates that our database is highly unbalanced and has a bias towards taking no action. In order not to have a biased model, it is necessary to deal with this problem. One remedy for this is to use CrossEntropy loss instead of MSE loss so that the model is penalized more heavily for cases where model is very sure on wrong decisions. In addition to this, a common practice is to sample the database in inverse proportion to each action's presence in the database i.e. actions that are low in number are sampled more. A similar approach, which is implemented as an option, is to remove samples of overrepresented action categories from the dataset. An alternative approach, which is also implemented here, is to use weighted loss function i.e. actions that are low in number are weighted more in loss. Weights are calculated in inverse proportion to action's population in the database.

Another implemented feature is epsilon greedy action selection strategy: at any given time action Accelerate is selected for several consecutive timesteps with the probability of epsilon<<0. Experiments are conducted for values of epsilon 0 and 0.01. This strategy can trigger the agent to go out of dead-lock situations and can have significant effects on the outcomes of the race.

### Overall network design
The overall network architecture is shown in the figure below:

<img src="Figures/NetworkArchi.png" width=500><br>

In this work two separated networks (VAE and Policy Network) are implemented and trained separately as done in [1] and [3]. This modular apporach is very useful in various ways:
- Debugging: we can more easily isolate and analyze different parts of the system 
- Research: we can quickly implement and test new ideas in different parts of the system without breaking the rest of the system

Evidently, the two networks can be merged into one network with a common lost function that is a combination of VAE and Policy Network losses, as done in [2]. This has the benefit of jointly training the two networks, optimizing further the VAE parameters with the task information.

## Implementation
### Generating expert data
To generate expert data `record_data.py` script can be used:

`python record_data.py --data_folder ./data_human --num_episode 5`

This will setup the game environment for manual play for 5 episodes of games and save each episode in a separate file under the subfolder data_human. The user has to decide whether to keep the recorded data or discard it at the end of each episode. Samples are recorded as list of tuples, where each tuple is composed of (current state, received reward, taken action). A dataset class compatible with pytorch's data loader is also implemented and can be used with other recorded data if it follows this format.

For this work, 25 episodes of expert plays (24884 samples in total) are collected to serve as the human expert database.

### Training models
Training is composed of two stages: training the VAE and training the policy network. This can be achieved via `train.py` script:

`python train.py --data_folder ./data_human --num_epochs 20 --num_epochs_vae 10`

This command will first train the VAE for 10 epochs and then the policy network for 20 epochs using the data provided in subdirectory data_human. A policy network can also be trained separately if there exists a pre-trained VAE:

`python train.py --data_folder ./data_human --num_epochs 20 --do_load_vae 1 --vae_model_file ./checkpoint/checkpoint.vae.epoch.9.tar`

or training can continue from a previous checkpoint:

`python train.py --data_folder ./data_human --num_epochs 40 --do_load_vae 1 --vae_model_file ./checkpoint/checkpoint.vae.epoch.9.tar --model_file ./checkpoint/checkpoint.policy.epoch.19.tar --load_epoch 19`

This command will load the given policy network from epoch 19 and continue training until epoch 40. Other parameters that can be set via command line can be listed via -h argument e.g. `python train.py -h`. If necessary, more detailed modifications on hyperparameters can be done from within the script.

### Evaluating models
A simple evaluation script `evaluate.py` is implemented to test the model in simulation environment:

`python evaluate.py  --vae_model_file ./checkpoint/checkpoint.vae.epoch.9.tar --model_file ./checkpoint/checkpoint.policy.epoch.19.tar --num_episode 5 --epsilon 0.01`

This command will run the given model for 5 episodes of game with epsilon set to 0.01. 

### Expert corrections on learned policies
After the model is trained, it can be executed in the simulation environment in interactive mode where the human expert can override the actions of the model. Whenever the expert overrides an action, this is considered as a correction and respective samples (state, reward, override action) are saved for re-training after the simulations. This is a very useful feature and has great value in real-world applications such as autonomous driving where driver interventions can be collected to improve models. This is realized using `retrain_with_expert.py` script:

`python retrain_with_expert.py  --vae_model_file ./checkpoint/checkpoint.vae.epoch.9.tar --model_file ./checkpoint/checkpoint.policy.epoch.19.tar --num_episode 5 --data_folder ./data_human_ft`

If num_epochs argument is provided, the script automatically re-trains the network after the game episodes are done but data is also saved for training later. 

## Results
The evolution of policy network training in terms of loss is shown in the figure below:

<img src="Figures/loss_base.png" width=500><br>

Training loss decreases rapidly in the first few epochs and still has tendency to decrease at epoch 20. Further experiments with longer training periods indeed showed lower loss values but no significant improvement in the agent's performance. Validation loss also decreases rapidly and stays below the training loss. Everything looks normal.

The resulting system is evaluated in simulation over multiple game episodes with different tracks generated for each episode. Table below compares the performance of different systems. The values shown are mean and standard deviation of total rewards acquired for 120 different game episodes except for the human expert which is evaluated over 25 episodes:

epsilon | Human Expert | Baseline | Fine tuning | Mixed data
------------ | ------------ | ------------- | ------------- | -------------
0 | 770 (61) | 487 (143) | 702 (167) | 616 (190)
0.01 | 770 (61) | 639 (142) | 752 (208) | 700 (213)

Here:
- **Baseline** is the agent trained with expert data for 20 epochs
- **Fine tuning** is the baseline agent trained with 2 episodes of expert corrections for 1 epoch
- **Mixed data** is the agent trained with mixed expert and correction data for 20 episodes 

At this stage, Baseline agent is more conservative in driving with respect to human expert. This is probably due to hesitant usage of acceleration by the human agent and bias towards No Action (see action preferences figure above). However, adding very few expert feedback where more acceleration is used (Fine tuning case) greatly increases the performance. Action preferences of the 2 expert feedback episodes is shown below:

<img src="Figures/ActionPreferences_ft.png" width=500><br>

To be clear, the system is not trained from scratch with data from expert feedback episodes, a training epoch is realized over the already learned model with only feedback data.

In the case of Mixed data the performance stays somewhere between the Baseline and Fine tuning. Newly added samples probably boost the number of samples of the actions accelerate, steer left and steer right but there is still bias in the dataset towards No Action. To remedy this, we have several options: doing more interactive feedback sessions with the expert, tuning weights of action categories for the loss function or get balanced samples during training. Further tests conducted with a more balanced training set (achieved by removing samples from excessive action categories) resulted in an average total reward of 751 with standard deviation of 216 over 25 game episodes.

Another point to remark is that a slight increase in the epsilon boosts the performance of agents considerably by introducing occasional bursts of thrusts and saving agents from deadlocks. 

Since the objective is not only learning to drive from an expert but also getting as close to the expert behaviour as possible (at least that is the author's interpretation of Behavioural Cloning) it is interesting to compare the behaviour of the expert and the agent. For example, we can compare the action preferences of the two to have a general idea how closely the agent mimics the expert:

<img src="Figures/ActionPreferences_human.png" width=500><br>
<img src="Figures/ActionPreferences_agent.png" width=500><br>

As we can see the behaviour of the agent is similar to that of the expert. The agent tends to accelerate more and use right turns less. We can also do a qualitative comparison of the behaviours of the agent and the human expert. Upon a visual inspection, it can be seen that the agent tends to take sharp curves from inside and occasionally over the grass like the human expert. An example case is shown in the figure below:

<img src="Figures/curve_behaviour.png" width=500><br>

Finally, an additional dataset is generated from a non-expert driver. Since the non-expert driver's behaviour is significantly different than the expert's behaviour, models trained with this new dataset should behave significantly differently, closer to the behaviour of the non-expert. This hypothesis is confirmed by visually comparing the behaviours of these agents and the behaviours of their counterpart experts. 

## Sources
- [Code](https://github.com/cemkaraoguz/Fahrschule)
- [Models](https://github.com/cemkaraoguz/Fahrschule/tree/main/models)
  - vae.epoch.9.tar : VAE model
  - policy.epoch.19.tar : Policy network trained on expert data
  - policy.ft.epoch.20.tar : Policy network trained on expert data and fine tuned with expert corrections
- [Datasets](https://drive.google.com/drive/folders/1IAjsZQ8uWzMVeMHCNOHFhTFlUep04BzW?usp=sharing)
  - data_human : expert data from human
  - data_human_ft : expert corrections from interactive mode

## References
[1] Ha, David, and Jürgen Schmidhuber. "World models." arXiv preprint arXiv:1803.10122 (2018). [Link](https://arxiv.org/pdf/1803.10122.pdf)

[2] Henaff, Mikael, Alfredo Canziani, and Yann LeCun. "Model-predictive policy learning with uncertainty regularization for driving in dense traffic." arXiv preprint arXiv:1901.02705 (2019). [Link](https://arxiv.org/pdf/1901.02705.pdf)

[3] Santana, Eder, and George Hotz. "Learning a driving simulator." arXiv preprint arXiv:1608.01230 (2016). [Link](https://arxiv.org/pdf/1608.01230.pdf)
