import torch
from torch.utils.data import DataLoader, random_split
from Datasets import CarRacingDataset
from Models import ConvVAEWrapper, ResNetWrapper
from Utils import saveLogData, parseArguments, getValueFromDict
import sys
from Globals import *

if __name__=="__main__":
  
  cmd_args = parseArguments(sys.argv[1:])
  
  num_epochs = int(getValueFromDict(cmd_args, 'num_epochs', 1))
  checkpoint_step = int(getValueFromDict(cmd_args, 'checkpoint_step', 1))
  checkpoint_folder = str(getValueFromDict(cmd_args, 'checkpoint_folder', "./checkpoint"))
  do_save_checkpoints = str(getValueFromDict(cmd_args, 'do_save_checkpoints', "True")).lower() in ["true", "1"]

  data_folder = str(getValueFromDict(cmd_args, 'data_folder', ""))
  if data_folder=="":
    print("data_folder argument is necessary!")
    sys.exit(2)
  
  batch_size = 64
  
  args = {'num_epochs': num_epochs,        # Number of epochs to train
          'lr': 0.01,                      # Learning rate
          'weight_decay': 1e-4,            # Weight decay
          'grad_clip': 0.1,                # Gradient clip
          'steps_per_epoch': None,         # Steps per epoch, updated after loading data, set None for no lr scheduling
          'do_use_cuda': True,             # Use CUDA?
          'num_classes': 3,                # Output dimension of the network, number of actions in this case
          'in_channels': IM_CHANNELS,      # Input channels
          'num_channels': 64,              # Channels of the first block, will be doubled after each scale block
          'shortcut': 'conv',              # Shortcut connections of the residual blocks, for now only conv is implemented
          'num_res_blocks': [3,3,3],       # Number of scale blocks and number of residual blocks in each
          'comments': 'policy network for behaviour cloning',
         }
  
  log_data = {}
  log_data['args'] = args # Save model hyperparams for further reference
  
  # Datasets and loaders
  dataset = CarRacingDataset(data_folder, size=(IM_HEIGHT,IM_WIDTH))
  if R_TRAIN>0:
    train_set, valid_set = random_split(dataset, [int(len(dataset)*R_TRAIN), len(dataset)-int(len(dataset)*(R_TRAIN))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
  else:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = None

  args['steps_per_epoch'] = len(train_loader)

  policy_network = ResNetWrapper(args)  
  
  for epoch in range(num_epochs):
    
    print("Epoch: ", epoch)
    
    # Train
    train_loss = policy_network.train_epoch(train_loader)
    
    # Validate
    if valid_loader is not None:
      valid_loss = policy_network.test_epoch(valid_loader)
    else:
      valid_loss = 0
    
    # Save checkpoint
    if do_save_checkpoints and (epoch%checkpoint_step)==0:
      checkpoint_filename = 'checkpoint.resnet.epoch.'+str(epoch)+'.tar'
      policy_network.save_checkpoint(folder=checkpoint_folder, filename=checkpoint_filename)
      
      # Logging
      if epoch not in log_data.keys():
        log_data[epoch] = {}
      log_data['last_iteration'] = epoch
      log_data[epoch]['train_loss'] = train_loss.avg
      log_data[epoch]['valid_loss'] = valid_loss.avg
      saveLogData(log_data, checkpoint_folder)
  