import torch
from DataLoaders import RaceCarDataLoader
from Models import AutoEncoderWrapper

if __name__=="__main__":

  im_width = 64
  im_height = 64
  
  args = {'type_network': 'convvae', # pca, rica, ae, convae, vae, convvae
          'in_channels': 3,
          # ConvVAE
          'rows' : im_height,
          'cols' : im_width,
          'num_hidden_features': [256, 128, 64],
          'strides': [1, 4, 2],          
          # VAE
          #'num_hidden_features': [512, 32],
          # ConvAE
          #'num_hidden_features': [16, 32],
          #'strides': [1, 1],
          'num_epochs': 1,
          'do_use_cuda': True,
          'in_features': im_height*im_width,
         }
  
  data_path = "./data_ai_small"
  batch_size = 64

  # define the data loaders
  train_loader = RaceCarDataLoader("./data_ai_small", batch_size=batch_size, shuffle=True, size=(im_height,im_width))

  args['steps_per_epoch'] = len(train_loader)
  
  ae = AutoEncoderWrapper(args)
  
  # Before training
  #ae.test(valid_loader, (im_height, im_width))
  
  # Training
  ae.train(train_loader)
  
  # After training
  ae.test(train_loader, (im_height, im_width))