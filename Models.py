import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from Utils import AverageMeter, getValueFromDict
import pylab as pl
import numpy as np
import os
from Globals import *

#-------------------------
#      Architectures
#-------------------------

class ConvVAE(nn.Module):
  ''' 
  Convolutional Variational Autoencoder 
  '''
  
  def __init__(self, args):
    super().__init__()
    self.rows = args['rows']
    self.cols = args['cols']
    self.in_channels = args['in_channels']
    self.num_hidden_features = getValueFromDict(args, 'num_hidden_features', [128, 32])
    self.num_latent_features = getValueFromDict(args, 'num_latent_features', 32)
    self.strides = getValueFromDict(args, 'strides', [1, 1])
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
    self.fc1 = nn.Linear(num_channels[-1]*self.encoder_rows*self.encoder_cols, self.num_latent_features)
    self.fc2 = nn.Linear(num_channels[-1]*self.encoder_rows*self.encoder_cols, self.num_latent_features)
    self.fc3 = nn.Linear(self.num_latent_features, num_channels[-1]*self.encoder_rows*self.encoder_cols)
    # Decoder
    decoder_blocks = []
    for i in range(len(num_channels)-1,0,-1):
      decoder_blocks.append(nn.ConvTranspose2d(in_channels=num_channels[i], out_channels=num_channels[i-1], 
                              kernel_size=3, padding=1, stride=self.strides[i-1], output_padding=self.strides[i-1]-1))
      decoder_blocks.append(nn.BatchNorm2d(num_channels[i-1]))
      if(i==1):
        pass
      else:
        decoder_blocks.append(nn.ReLU())
    self.decoder = nn.Sequential(*decoder_blocks)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    h = self.encoder(x).view(-1, self.num_hidden_features[-1]*self.encoder_rows*self.encoder_cols)
    mu = self.fc1(h)
    log_var = self.fc2(h)
    z = self._reparameterize(mu, log_var)
    z = self.fc3(z).view(-1, self.num_hidden_features[-1], self.encoder_rows, self.encoder_cols)
    logits = self.decoder(z)
    out = self.sigmoid(logits)
    return (mu, log_var), logits, out
  
  def encode(self, x):
    h = self.encoder(x).view(-1, self.num_hidden_features[-1]*self.encoder_rows*self.encoder_cols)
    mu = self.fc1(h)
    log_var = self.fc2(h)
    z = self._reparameterize(mu, log_var)
    return z
    
  def _reparameterize(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample

class LiNet(nn.Module):
  '''
  Linear connection network
  '''
  def __init__(self, args):
    super(LiNet, self).__init__()
    self.in_channels = args['in_channels']
    self.num_classes = args['num_classes']
    num_channels = args['num_channels']
    self.fc1 = nn.Linear(in_features=self.in_channels, out_features=num_channels)
    self.fc2 = nn.Linear(in_features=num_channels, out_features=self.num_classes)
    
  def forward(self, x):
    out = self.fc1(x)
    out = F.relu(out)
    logits = self.fc2(out)
    probs = F.softmax(logits, dim=1)    
    return logits, probs
    
#-------------------------
#  Architecture Wrappers
#-------------------------
class ModelWrapper():

  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.net.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save({'state_dict': self.net.state_dict()}, filepath)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.net.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError("No model in path {}".format(filepath))
    print("Loading model file {}".format(filepath))
    map_location = None if self.do_use_cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    self.net.load_state_dict(checkpoint['state_dict'])
    
class ConvVAEWrapper(ModelWrapper):
  
  def __init__(self, args):
    self.in_channels = getValueFromDict(args, 'in_channels', None)
    weight_decay = getValueFromDict(args, 'weight_decay', 0.01)
    self.lr = getValueFromDict(args, 'lr', 1e-3)
    self.net = ConvVAE(args)
    self.lossFunction = self._vae_loss
    self.do_use_cuda = args['do_use_cuda'] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.do_use_cuda else "cpu")
    self.net.to(self.device)
    self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=weight_decay)
  
  def encode(self, obs):
    return self.net.encode(obs)
  
  def train_epoch(self, data_loader):
    self.net.train()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Training")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      self.optimizer.zero_grad()
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      latent_data, logits_data, output_data = self.net(batch_frame.to(self.device))
      loss = self.lossFunction(logits_data.to(self.device), batch_frame.to(self.device), 
        latent_data[0].to(self.device), latent_data[1].to(self.device))
      loss.backward()
      self.optimizer.step()
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
      
  def visualize_decoder(self, data_loader):
    im_shape = (IM_HEIGHT, IM_WIDTH)
    self.net.eval()
    valid_batch = next(iter(data_loader))
    latents = []
    recnsts = []
    vis_rows = 4
    vis_cols = 4
    for i in range(vis_rows*vis_cols):
      frame = torch.FloatTensor(np.array(valid_batch[0][i]).astype(np.float64))
      latent, logits, recnst = self.net(frame.view(-1, self.in_channels, im_shape[0], im_shape[1]).to(self.device))
      latents.append(latent)
      recnsts.append(recnst) 
    # Visualize reals
    pl.figure()
    pl.title("Real images")
    for i, recnst in enumerate(recnsts):
      pl.subplot(vis_rows,vis_cols,i+1)
      frame = torch.FloatTensor(np.array(valid_batch[0][i]).astype(np.float64))
      img = frame.data.numpy().reshape((self.in_channels, im_shape[0], im_shape[1]))
      img = np.transpose(img, [1,2,0])
      pl_img = pl.imshow(np.array(img*IM_DEPTH_NORM, dtype=int))
      pl_img.set_cmap('gray')
      pl.axis('off')
    # Visualize reconstructions
    pl.figure()
    pl.title("Reconstructions")
    for i, recnst in enumerate(recnsts):
      pl.subplot(vis_rows,vis_cols,i+1)
      img = recnst.data.cpu().numpy().reshape((self.in_channels, im_shape[0], im_shape[1]))
      img = np.transpose(img, [1,2,0])
      pl_img = pl.imshow(img)
      pl_img.set_cmap('gray')
      pl.axis('off')
    pl.show()
  
  def _vae_loss(self, network_output, target, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy_with_logits(network_output, target, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kld_loss  

class LiNetWrapper(ModelWrapper):
  
  def __init__(self, args):
    self.net = LiNet(args)
    self.num_epochs = args['num_epochs']
    self.category_weights = getValueFromDict(args, 'category_weights', None)
    self.lr = getValueFromDict(args, 'lr', 1e-3)
    self.grad_clip = getValueFromDict(args, 'grad_clip', 0.0)
    self.epsilon = getValueFromDict(args, 'epsilon', 0.01) # Probability of overriding network output with STRAIGHT action
    self.max_count_override = getValueFromDict(args, 'max_count_override', 5) # Maximum action override in the epsilon case
    steps_per_epoch = getValueFromDict(args, 'steps_per_epoch', None)
    self.do_use_cuda = args['do_use_cuda'] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.do_use_cuda else "cpu")
    self.net.to(self.device)
    self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    if steps_per_epoch is None:
      self.lr_scheduler = None
    else:
      self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, 
        epochs=self.num_epochs, steps_per_epoch=steps_per_epoch)
    if self.category_weights is None:
      self.loss_function = nn.CrossEntropyLoss()
    else:
      self.loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.category_weights).to(self.device))
    self.do_override = False
    self.count_override = 0

  def train_epoch(self, data_loader, encoder):
    self.net.train()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Training")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      self.optimizer.zero_grad()
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      batch_action = torch.LongTensor(np.array(batch_action).astype(np.float64))
      batch_state = encoder.encode(batch_frame.to(self.device))
      logits, probs = self.net(batch_state.to(self.device))
      loss = self.loss_function(logits.to(self.device), batch_action.to(self.device))
      loss.backward()
      if self.grad_clip:
        nn.utils.clip_grad_value_(self.net.parameters(), self.grad_clip)
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
    return avg_loss
  
  def test_epoch(self, data_loader, encoder):
    self.net.eval()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Testing")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      batch_action = torch.LongTensor(np.array(batch_action).astype(np.float64))
      with torch.no_grad():
        batch_state = encoder.encode(batch_frame.to(self.device))
        logits, probs = self.net(batch_state.to(self.device))
      loss = self.loss_function(logits.to(self.device), batch_action.to(self.device))
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
    return avg_loss
      
  def get_action(self, encoder, obs):
    self.net.eval()
    obs_tensor = torch.FloatTensor(np.array(obs).astype(np.float64)).view(-1, IM_CHANNELS, IM_HEIGHT, IM_WIDTH)
    with torch.no_grad():
      state = encoder.encode(obs_tensor.to(self.device))
      logits, probs = self.net(state.to(self.device))
    if np.random.rand()<self.epsilon or self.do_override:
      action_idx = 1
      self.do_override = True
      self.count_override += 1
      if self.count_override>=self.max_count_override:
        self.do_override = False
    else:
      action_idx = np.argmax(probs.detach().cpu().numpy()[0])
      self.count_override = 0     
    return ACTION_MAPPING[action_idx]  