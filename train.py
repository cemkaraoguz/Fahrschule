import torch
from torch.utils.data import DataLoader, random_split
from Datasets import CarRacingDataset
from Models import ConvVAEWrapper, LiNetWrapper
from Utils import saveLogData, parseArguments, getValueFromDict
import sys
import os
from Globals import *

if __name__=="__main__":
  
  cmd_args = parseArguments(sys.argv[1:])
  
  num_epochs_pn = int(getValueFromDict(cmd_args, 'num_epochs', 1))
  num_epochs_vae = int(getValueFromDict(cmd_args, 'num_epochs_vae', 1))
  checkpoint_folder = str(getValueFromDict(cmd_args, 'checkpoint_folder', "./checkpoint"))
  do_save_checkpoints = str(getValueFromDict(cmd_args, 'do_save_checkpoints', "True")).lower() in ["true", "1"]
  do_balance_dataset = str(getValueFromDict(cmd_args, 'do_balance_dataset', "False")).lower() in ["true", "1"]
  load_epoch = int(getValueFromDict(cmd_args, 'load_epoch', -1))
  model_file = str(getValueFromDict(cmd_args, 'model_file', ""))
  do_load_vae = str(getValueFromDict(cmd_args, 'do_load_vae', "False")).lower() in ["true", "1"]
  if do_load_vae:
    vae_model_file = str(getValueFromDict(cmd_args, 'vae_model_file', ""))
    if vae_model_file=="":
      print("vae_model_file argument is necessary!")
      sys.exit(2)   
    vae_checkpoint_folder, vae_checkpoint_file = os.path.split(vae_model_file) 
  
  data_folder = str(getValueFromDict(cmd_args, 'data_folder', ""))
  if data_folder=="":
    print("data_folder argument is necessary!")
    sys.exit(2)
  
  batch_size = 128
  num_skip_frames_dataset = 0 # Used for skipping zooming-in frames in the beginning of each episode

  # VAE hyperparameters
  args_vae = {'in_channels': IM_CHANNELS,                   # Input dimensions
              'rows' : IM_HEIGHT,
              'cols' : IM_WIDTH,
              'num_hidden_features': [32, 64, 128, 256],    # Hidden block channels
              'num_latent_features': 32,                    # Latent space dims
              'strides': [2, 2, 2, 2],                      # Strides for each hidden block
              'do_use_cuda': True,                          # Use CUDA?
              'comments': 'VAE for behaviour cloning'
             }
             
  # Policy network hyperparameters
  args_pn = {'num_epochs': num_epochs_pn,                     # Number of epochs to train
            'lr': 0.001,                                      # Learning rate
            'grad_clip': 0.1,                                 # Gradient clip
            'do_use_cuda': True,                              # Use CUDA?
            'num_classes': NUM_DISCRETE_ACTIONS,              # Output dimension of the network, number of actions in this case
            'in_channels': args_vae['num_latent_features'],   # Input channels
            'num_channels': 64,                               # Channels of the first block
            'category_weights': None,                         # Category weights for loss function
            'comments': 'policy network for behaviour cloning',
           }
  
  log_data = {}
  log_data['args_vae'] = args_vae # Save model hyperparams for further reference
  log_data['args_pn'] = args_pn # Save model hyperparams for further reference
  
  # Datasets and loaders
  dataset = CarRacingDataset(data_folder, action_space='discrete', num_skip_frames=num_skip_frames_dataset)
  if do_balance_dataset:
    dataset.balance_dataset()
  if R_TRAIN<1.0:
    train_set, valid_set = random_split(dataset, [int(len(dataset)*R_TRAIN), len(dataset)-int(len(dataset)*(R_TRAIN))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
  else:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = None
  
  args_pn['category_weights'] = dataset.calculate_weights()*100

  # -------------------
  #      VAE
  # -------------------
  vae = ConvVAEWrapper(args_vae)
  
  if do_load_vae:
    # Continue training not implemented, just load existing model to skip VAE training
    print("Loading VAE")
    vae.load_checkpoint(vae_checkpoint_folder, vae_checkpoint_file)   
    valid_loss_vae = vae.test_epoch(valid_loader)
  else:
    # Training
    print("Training VAE")
    for epoch in range(num_epochs_vae):
      
      print("Epoch: ", epoch)
      
      train_loss_vae = vae.train_epoch(train_loader)
      
      # Validate
      if valid_loader is not None:
        valid_loss_vae = vae.test_epoch(valid_loader)
      else:
        valid_loss_vae = 0
      
      if do_save_checkpoints: 
        vae_checkpoint_filename = 'checkpoint.vae.epoch.'+str(epoch)+'.tar'
        vae.save_checkpoint(folder=checkpoint_folder, filename=vae_checkpoint_filename)  
    
    # After training
    if valid_loader is not None:
      vae.visualize_decoder(valid_loader)

  # -------------------
  #      Policy
  # -------------------
  policy_network = LiNetWrapper(args_pn)  

  if load_epoch>=0:
    print("Loading PN")
    checkpoint_folder, checkpoint_file = os.path.split(model_file) 
    policy_network.load_checkpoint(checkpoint_folder, checkpoint_file)
    epoch_start = load_epoch + 1
  else:
    epoch_start = 0
  
  print("Training PN")
  
  for epoch in range(epoch_start, num_epochs_pn):
    
    print("Epoch: ", epoch)
    
    # Train
    train_loss_pn = policy_network.train_epoch(train_loader, vae)
    
    # Validate
    if valid_loader is not None:
      valid_loss_pn = policy_network.test_epoch(valid_loader, vae)
    else:
      valid_loss_pn = 0
    
    # Save checkpoint
    if do_save_checkpoints:
      checkpoint_filename = 'checkpoint.policy.epoch.'+str(epoch)+'.tar'
      policy_network.save_checkpoint(folder=checkpoint_folder, filename=checkpoint_filename)
      
      # Logging
      if epoch not in log_data.keys():
        log_data[epoch] = {}
      log_data['last_iteration'] = epoch
      log_data[epoch]['train_loss_pn'] = train_loss_pn.avg
      if valid_loader is not None:
        log_data[epoch]['valid_loss_pn'] = valid_loss_pn.avg
      if not do_load_vae:
        log_data[epoch]['train_loss_vae'] = train_loss_vae.avg
        if valid_loader is not None:
          log_data[epoch]['valid_loss_vae'] = valid_loss_vae.avg
      saveLogData(log_data, checkpoint_folder)
  