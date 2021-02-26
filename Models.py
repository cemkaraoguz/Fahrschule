import torch
from torch import nn
import torch.nn.functional as functional
import torch.optim as optim
from tqdm import tqdm
from Utils import AverageMeter, getValueFromDict
import pylab as pl
import numpy as np

class ConvVariationalAutoEncoder(nn.Module):
  ''' Convolutional Variational Autoencoder '''
  
  def __init__(self, args):
    super().__init__()
    self.rows = args['rows']
    self.cols = args['cols']
    self.in_channels = args['in_channels']
    self.num_hidden_features = getValueFromDict(args, 'num_hidden_features', [128, 32])
    self.n_latent_features = getValueFromDict(args, 'n_latent_features', 16)
    self.strides = getValueFromDict(args, 'strides', [1, 1])
    assert len(self.num_hidden_features)>=2    
    num_channels = [self.in_channels] + self.num_hidden_features
    # Encoder
    encoder_blocks = []
    for i in range(len(num_channels)-1):
      encoder_blocks.append(nn.Conv2d(in_channels=num_channels[i], out_channels=num_channels[i+1], 
                              kernel_size=3, padding=1, stride=self.strides[i]))
      encoder_blocks.append(nn.BatchNorm2d(num_channels[i+1]))
      encoder_blocks.append(nn.ReLU())
    self.encoder = nn.Sequential(*encoder_blocks)
    # Latent layer
    self.encoder_rows = self.rows//np.prod(self.strides)
    self.encoder_cols = self.cols//np.prod(self.strides)
    self.fc1 = nn.Linear(num_channels[-1]*self.encoder_rows*self.encoder_cols, self.n_latent_features)
    self.fc2 = nn.Linear(num_channels[-1]*self.encoder_rows*self.encoder_cols, self.n_latent_features)
    self.fc3 = nn.Linear(self.n_latent_features, num_channels[-1]*self.encoder_rows*self.encoder_cols)
    # Decoder
    decoder_blocks = []
    for i in range(len(num_channels)-1,0,-1):
      decoder_blocks.append(nn.ConvTranspose2d(in_channels=num_channels[i], out_channels=num_channels[i-1], 
                              kernel_size=3, padding=1, stride=self.strides[i-1], output_padding=self.strides[i-1]-1))
      decoder_blocks.append(nn.BatchNorm2d(num_channels[i-1]))
      if(i==1):
        decoder_blocks.append(nn.Sigmoid())
      else:
        decoder_blocks.append(nn.ReLU())
    self.decoder = nn.Sequential(*decoder_blocks)
  
  def forward(self, x):
    h = self.encoder(x).view(-1, self.num_hidden_features[-1]*self.encoder_rows*self.encoder_cols)
    mu = self.fc1(h)
    log_var = self.fc2(h)
    z = self.reparameterize(mu, log_var)
    z = self.fc3(z).view(-1, self.num_hidden_features[-1], self.encoder_rows, self.encoder_cols)
    out = self.decoder(z)
    return (mu, log_var), out
    
  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5*log_var) # standard deviation
    eps = torch.randn_like(std) # `randn_like` as we need the same size
    sample = mu + (eps * std) # sampling as if coming from the input space
    return sample
    
class AutoEncoderWrapper():
  
  def __init__(self, args):
    self.num_epochs = args['num_epochs']
    self.in_features = args['in_features']
    self.in_channels = getValueFromDict(args, 'in_channels', None)
    self.type_network = args['type_network']
    weight_decay = getValueFromDict(args, 'weight_decay', 0.01)
    self.net = ConvVariationalAutoEncoder(args)
    self.lossFunction = self.vae_loss
    self.do_use_cuda = args['do_use_cuda'] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.do_use_cuda else "cpu")
    self.net.to(self.device)
    self.optimizer = optim.AdamW(self.net.parameters(), weight_decay=weight_decay)
    print(self.net)
    
  def train(self, train_loader):
    for epoch in range(self.num_epochs):
      print("Epoch:", epoch)
      self.train_epoch(train_loader)
      
  def train_epoch(self, data_loader):
    self.net.train()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Training")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      self.optimizer.zero_grad()
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      latent_data, output_data = self.net(batch_frame.to(self.device))
      loss = self.lossFunction(output_data.to(self.device), batch_frame.to(self.device), 
        latent_data[0].to(self.device), latent_data[1].to(self.device))
      loss.backward()
      self.optimizer.step()
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
      
  def test(self, data_loader, im_shape):
    self.net.eval()
    valid_batch = next(iter(data_loader))
    latents = []
    recnsts = []
    vis_rows = 4
    vis_cols = 4
    for i in range(vis_rows*vis_cols):
      frame = torch.FloatTensor(np.array(valid_batch[0][i]).astype(np.float64))
      if self.type_network!='convae' and self.type_network.lower()!='convvae':
        latent, recnst = self.net(frame.view(-1, self.in_features).to(self.device))
      else:
        latent, recnst = self.net(frame.view(-1, self.in_channels, im_shape[0], im_shape[1]).to(self.device))
      latents.append(latent)
      recnsts.append(recnst) 
    # Visualize reals
    pl.figure()
    for i, recnst in enumerate(recnsts):
      pl.subplot(vis_rows,vis_cols,i+1)
      img = frame.data.numpy().reshape((self.in_channels, im_shape[0], im_shape[1]))
      img = np.transpose(img, [1,2,0])
      pl_img = pl.imshow(np.array(img, dtype=int))
      pl_img.set_cmap('gray')
      pl.axis('off')
      pl.title("Real")
    # Visualize reconstructions
    pl.figure()
    for i, recnst in enumerate(recnsts):
      pl.subplot(vis_rows,vis_cols,i+1)
      img = recnst.data.cpu().numpy().reshape((self.in_channels, im_shape[0], im_shape[1]))
      img = np.transpose(img, [1,2,0])
      pl_img = pl.imshow(img)
      pl_img.set_cmap('gray')
      pl.axis('off')
      pl.title("Reconst")
    pl.show()
  
  def vae_loss(self, network_output, target, mu, log_var):
    reconstruction_loss = functional.binary_cross_entropy(network_output, target, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kld_loss