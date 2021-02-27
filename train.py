import torch
from torch.utils.data import DataLoader, random_split
from Datasets import CarRacingDataset
from Models import ConvVAEWrapper, ResNetWrapper
from Utils import saveLogData

R_TRAIN = 0.7 # Ratio of training samples in dataset

if __name__=="__main__":
  
  im_width = 64
  im_height = 64
  num_epochs = 1
  checkpoint_step = 1
  checkpoint_folder = "./checkpoint"
  
  #data_path = "./data_ai"
  data_path = "./data_ai_s"
  #data_path = "./data_ai_m"
  
  batch_size = 64
  
  args_vae = {'type_network': 'convvae', # pca, rica, ae, convae, vae, convvae
          'in_channels': 3,
          # ConvVAE
          'rows' : im_height,
          'cols' : im_width,
          'num_hidden_features': [32, 64, 128, 256],
          'strides': [2, 2, 2, 2],
          'num_epochs': 1,
          'do_use_cuda': True,
          'in_features': im_height*im_width,
         }

  args_resnet = {'num_epochs': num_epochs,
          'lr': 0.01,
          'weight_decay': 1e-4,
          'grad_clip': 0.1,
          'steps_per_epoch': None, # Update after loading data
          'do_use_cuda': True,
          'num_classes': 3,
          'in_channels': 3,
          'num_channels': 64,
          'shortcut': 'conv',
          'num_res_blocks': [3,3,3],
          'comments': 'policy network for behaviour cloning',
         }
  
  log_data = {}
  log_data['args'] = args_resnet # Save model hyperparams for further reference
  
  # Datasets and loaders
  dataset = CarRacingDataset(data_path, size=(im_height,im_width))
  if R_TRAIN>0:
    train_set, valid_set = random_split(dataset, [int(len(dataset)*R_TRAIN), len(dataset)-int(len(dataset)*(R_TRAIN))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
  else:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = None

  args_vae['steps_per_epoch'] = len(train_loader)
  args_resnet['steps_per_epoch'] = len(train_loader)

  policy_network = ResNetWrapper(args_resnet)  
  
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
    if (epoch%checkpoint_step)==0:
      checkpoint_filename = 'checkpoint.resnet.epoch.'+str(epoch)+'.tar'
      policy_network.save_checkpoint(folder=checkpoint_folder, filename=checkpoint_filename)
      
      # Logging
      if epoch not in log_data.keys():
        log_data[epoch] = {}
      log_data['last_iteration'] = epoch
      log_data[epoch]['train_loss'] = train_loss.avg
      log_data[epoch]['valid_loss'] = valid_loss.avg
      saveLogData(log_data, checkpoint_folder)
  
  '''
  ae = ConvVAEWrapper(args_vae)
  
  # Before training
  #ae.test(valid_loader, (im_height, im_width))
  
  # Training
  ae.train(train_loader)
  
  # After training
  ae.test(train_loader, (im_height, im_width))
  '''
  
  