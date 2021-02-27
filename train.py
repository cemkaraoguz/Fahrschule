import torch
from torch.utils.data import DataLoader, random_split
from Datasets import CarRacingDataset
from Models import ConvVAEWrapper, ResNetWrapper

R_TRAIN = 0.7 # Ratio of training samples in dataset

if __name__=="__main__":
  
  im_width = 64
  im_height = 64
  
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

  args_resnet = {'num_epochs': 10,
          'lr': 0.01,
          'weight_decay': 1e-4,
          'grad_clip': 0.1,
          'steps_per_epoch': None, # Update after loading data
          'do_use_cuda': True,
          'num_classes': 3,
          'in_channels' : 3,
          'num_channels': 64,
          'shortcut': 'conv',
          'num_res_blocks': [3,3,3],
         }
         
  #data_path = "./data_ai"
  data_path = "./data_ai_s"
  #data_path = "./data_ai_m"
  batch_size = 64

  # Datasets and loaders
  dataset = CarRacingDataset(data_path, size=(im_height,im_width))
  train_set, valid_set = random_split(dataset, [int(len(dataset)*R_TRAIN), len(dataset)-int(len(dataset)*(R_TRAIN))])
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

  args_vae['steps_per_epoch'] = len(train_loader)
  args_resnet['steps_per_epoch'] = len(train_loader)

  resnet18 = ResNetWrapper(args_resnet)  
  resnet18.train(train_loader, valid_loader)
  
  '''
  ae = ConvVAEWrapper(args_vae)
  
  # Before training
  #ae.test(valid_loader, (im_height, im_width))
  
  # Training
  ae.train(train_loader)
  
  # After training
  ae.test(train_loader, (im_height, im_width))
  '''
  
  