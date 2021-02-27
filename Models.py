import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from Utils import AverageMeter, getValueFromDict
import pylab as pl
import numpy as np
import os

#-------------------------
#      Architectures
#-------------------------

class ConvVAE(nn.Module):
  ''' Convolutional Variational Autoencoder '''
  
  def __init__(self, args):
    super().__init__()
    self.rows = args['rows']
    self.cols = args['cols']
    self.in_channels = args['in_channels']
    self.num_hidden_features = getValueFromDict(args, 'num_hidden_features', [128, 32])
    self.n_latent_features = getValueFromDict(args, 'n_latent_features', 32)
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

class ResidualBlock(nn.Module):
  
  def __init__(self, in_channels, out_channels, shortcut='conv'):
    super(ResidualBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.shortcut = shortcut
    first_layer_stride = 1
    if not self.in_channels==self.out_channels:
      first_layer_stride = 2
      if shortcut=='conv':
        self.sc_layer = nn.Sequential(
          nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2),
          nn.BatchNorm2d(num_features=self.out_channels)
        )  
      else:
        raise(NotImplementedError)
    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=first_layer_stride, padding=1)
    self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=self.out_channels)
    self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)
  
  def forward(self, x):
    if not (self.in_channels==self.out_channels):
      identity = self.sc_layer(x)
    else:
      identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + identity
    return F.relu(out) 
    
class ResNet(nn.Module):
  
  def __init__(self, args):
    super(ResNet, self).__init__()
    self.num_res_blocks = args['num_res_blocks']
    self.in_channels = args['in_channels']
    self.num_classes = args['num_classes']
    shortcut = args['shortcut']
    num_channels = args['num_channels']
    self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=num_channels)
    res_blocks = []   
    for i, n in enumerate(self.num_res_blocks):
      if i>0:
        # Dimension change
        res_blocks.append(ResidualBlock(num_channels, num_channels*2))
        num_channels = num_channels*2
        num_blocks_remaining = n-1
      else:
        num_blocks_remaining = n
      for _ in range(num_blocks_remaining):
        res_blocks.append(ResidualBlock(num_channels, num_channels))
    self.res_blocks = nn.Sequential(*res_blocks)   
    self.pool2 = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(in_features=num_channels, out_features=self.num_classes)
    
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.res_blocks(out)
    out = self.pool2(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out
    #return F.log_softmax(out, dim=1)

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
    self.num_epochs = args['num_epochs']
    self.in_features = args['in_features']
    self.in_channels = getValueFromDict(args, 'in_channels', None)
    self.type_network = args['type_network']
    weight_decay = getValueFromDict(args, 'weight_decay', 0.01)
    self.net = ConvVAE(args)
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
    #for _, idx_batch in zip(t, range(len(data_loader))):
      #batch_frame, batch_reward, batch_action = data_loader.get_batch(idx_batch)
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
    #valid_batch = data_loader.get_batch(np.random.randint(len(data_loader)))
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
      frame = torch.FloatTensor(np.array(valid_batch[0][i]).astype(np.float64))
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
    reconstruction_loss = F.binary_cross_entropy(network_output, target, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kld_loss  
    
class ResNetWrapper(ModelWrapper):
  
  def __init__(self, args):
    self.net = ResNet(args)
    self.num_epochs = args['num_epochs']
    self.lr = getValueFromDict(args, 'lr', 1e-3)
    self.weight_decay = getValueFromDict(args, 'weight_decay', 1e-2)
    self.grad_clip = getValueFromDict(args, 'grad_clip', 0.0)
    steps_per_epoch = args['steps_per_epoch']
    self.do_use_cuda = args['do_use_cuda'] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.do_use_cuda else "cpu")
    self.net.to(self.device)
    self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    if steps_per_epoch is None:
      self.lr_scheduler = None
    else:
      self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, 
        epochs=self.num_epochs, steps_per_epoch=steps_per_epoch)
    self.loss_function = nn.MSELoss()
  
  def train_epoch(self, data_loader):
    self.net.train()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Training")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      self.optimizer.zero_grad()
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      batch_action = torch.FloatTensor(np.array(batch_action).astype(np.float64))
      actions = self.net(batch_frame.to(self.device))
      loss = self.loss_function(batch_action.to(self.device), actions.to(self.device))
      loss.backward()
      if self.grad_clip:
        nn.utils.clip_grad_value_(self.net.parameters(), self.grad_clip)
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
    return avg_loss
  
  def test_epoch(self, data_loader):
    self.net.eval()
    avg_loss = AverageMeter()
    t = tqdm(range(len(data_loader)), desc="Testing")
    for _, (batch_frame, batch_reward, batch_action) in zip(t, data_loader):
      batch_frame = torch.FloatTensor(np.array(batch_frame).astype(np.float64))
      batch_action = torch.FloatTensor(np.array(batch_action).astype(np.float64))
      with torch.no_grad():
        actions = self.net(batch_frame.to(self.device))
      loss = self.loss_function(batch_action.to(self.device), actions.to(self.device))
      avg_loss.update(loss.item(), batch_frame.size(0))
      t.set_postfix(Loss=avg_loss)
    return avg_loss
      
  def get_action(self, obs):
    self.net.eval()
    obs_tensor = torch.FloatTensor(np.array(obs).astype(np.float64)).view(-1, 3, 64, 64)
    with torch.no_grad():
      actions = self.net(obs_tensor.to(self.device))
    return actions.detach().cpu().numpy()[0]

  