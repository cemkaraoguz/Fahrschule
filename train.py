import torch
from DataLoaders import RaceCarDataLoader
from Models import ConvVAEWrapper, ResNetWrapper

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

  args_resnet = {'num_epochs': 1,
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
         
  #data_path = "./data_ai_small"
  data_path = "./data_ai"
  batch_size = 64

  # define the data loaders
  train_loader = RaceCarDataLoader(data_path, batch_size=batch_size, shuffle=True, size=(im_height,im_width))

  args_vae['steps_per_epoch'] = len(train_loader)
  args_resnet['steps_per_epoch'] = len(train_loader)

  resnet18 = ResNetWrapper(args_resnet)  
  resnet18.train(train_loader)
  
  '''
  ae = ConvVAEWrapper(args_vae)
  
  # Before training
  #ae.test(valid_loader, (im_height, im_width))
  
  # Training
  ae.train(train_loader)
  
  # After training
  ae.test(train_loader, (im_height, im_width))
  '''
  
  