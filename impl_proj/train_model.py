from sprites_datagen.moving_sprites import ImplDataset
from sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward
from sprites_env import *
from general_utils import AttrDict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

device = torch.device('cuda:1') # TODO: fix cuda issue
dtype = torch.float32


def loadData(spec, new=False, path='', batch_size=32, train_num=5000, val_num=1000, test_num=1000):

    """Return dataloaders for train/val/test datasets"""
    train = path + '/train.data'
    val = path + '/val.data'
    test = path + '/test.data'
    try:
        train_dataset = ImplDataset(spec, train_num, train) if new else torch.load(train)
        val_dataset = ImplDataset(spec, val_num, val) if new else torch.load(val)
        test_dataset = ImplDataset(spec, test_num, test) if new else torch.load(test)
    except Exception as err:
        print('Error when loading dataset: ' + err)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class Encoder(nn.Module):
    """cnn encoder that reduces resolution by a factor of 2 in every layer"""

    def __init__(self, encoder_params=None):
        super().__init__()

        self.num_layers = 6 # self.num_layers = np.log2(encoder_params['resolution'])
        num_channels = 3 # num_channels = encoder_params['init_num_channels']
        kernel_size = 33 # encoder_params['resolution'] / 2 + 1
        cnn_layers = []

        for i in range(self.num_layers):
            cnn_layers.append(nn.Conv2d(num_channels, num_channels*2, int(kernel_size)))
            cnn_layers.append(nn.ReLU())
            num_channels *= 2
            kernel_size = (kernel_size - 1) / 2 + 1

        self.layers = nn.ModuleList(cnn_layers)
        self.mapping = nn.Linear(192, 64)
        # nn.Linear(num_channels*np.power(2,self.num_layers), encoder_params['obs_dim'])



    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = torch.squeeze(out)
        out = self.mapping(out)
        return out


class MLP(nn.Module):

    def __init__(self, params=None):
        """
        Sub MLP module contained in SineWaveLSTM.

        Inputs:
        - lstm_params: Dict storing parameters of the network
            - num_layers, input_dim, output_dim, l1_dim, etc.
        """
        super().__init__()

        self.num_layers = params['num_layers']
        fc_layers = []
        for i in range(self.num_layers):
            input_dim = params['input_dim'] if i == 0 else params['l' + str(i) + '_dim']
            output_dim = params['output_dim'] if i == self.num_layers-1 else params['l' + str(i+1) + '_dim']
            fc_layers.append(nn.Linear(input_dim, output_dim))

            if i != self.num_layers-1:
                fc_layers.append(nn.ReLU())

        self.layers = nn.ModuleList(fc_layers)

    def forward(self, x):
        """
        Inputs:
        - x: Input of shape (batch_size, input_dim)

        Returns:
        - out: Output of shape (batch_size, lstm_dim)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Predictor(nn.Module):
    """LSTM predictor for reward preiction"""

    def __init__(self, predictor_params=None):
        super().__init__()

        self.input_size = 384 # predictor_params['lstm_size'] # assert == embed_stack.shape
        self.hidden_size = 512 # predictor_params['hidden_size']
        self.sequence_length = 27 # predictor_params['sequence_length']

        self.mlp_params = {} # predictor_params['mlp_params']
        self.mlp_params['num_layers'] = 3
        self.mlp_params['input_dim'] = 64 # assert == encoder output size
        self.mlp_params['l1_dim'] = 512
        self.mlp_params['l2_dim'] = 256
        self.mlp_params['output_dim'] = 128

        self.mlp = MLP(self.mlp_params)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        # self.fc = (?)

    def forward(self, ztm2, ztm1, z):
        ztm2_embed = self.mlp(ztm2)
        ztm1_embed = self.mlp(ztm1)
        z_embed = self.mlp(z)
        embed = torch.cat((ztm2_embed,ztm1_embed,z_embed), dim=1)
        embed_stack = torch.stack([embed] * self.sequence_length, dim=1)
        lstm_out, _  = self.lstm(embed_stack) # (batch_size, seq_len, hidden_dim)
        return lstm_out


class Model(nn.Module):
    """complete architecture"""

    def __init__(self, model_params=None):
        super().__init__()

        # self.encoder_params = {} # model_params['encoder_params]
        # implement encoder_params
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.encoder3 = Encoder()

        # self.predictor_params = {} # model_params['predictor_params]
        # implement predictor_params
        self.predictor = Predictor()

        self.reward_params = {} # model_params['reward_mlp_params']
        self.reward_params['num_layers'] = 3
        self.reward_params['input_dim'] = 512 # assert == self.predictor.hidden_size
        self.reward_params['l1_dim'] = 256
        self.reward_params['l2_dim'] = 64
        self.reward_params['output_dim'] = 1

        self.reward_heads = []
        for i in range(27): # model_params['T']
            self.reward_heads.append(MLP(self.reward_params))

    def forward(self, tm2, tm1, t):
        ztm2 = self.encoder1(tm2)
        ztm1 = self.encoder2(tm1)
        zt = self.encoder3(t)
        h1tT = self.predictor(ztm2, ztm1, zt) # (batch_size, seq_len, hidden_dim)
        reward_estimates = []
        for i in range(h1tT.shape[1]):
            head = self.reward_heads[i]
            input = h1tT[:,i,:]
            reward_estimates.append(head(input))
        return reward_estimates



def trainModel(model, optimizer, loader, file, epochs=50, print_every=100):
    print("Training starts.")
    print()
    model = model.to(device=device)
    criterion = nn.MSELoss()
    for e in range(epochs):
        for t, sample in enumerate(loader):
            model.train()

            tm2 = sample['images'][:,0,:,:,:]
            tm1 = sample['images'][:,1,:,:,:]
            t = sample['images'][:,2,:,:,:]
            y = sample['rewards']['agent_x']

            tm2 = tm2.to(device=device, dtype=dtype)
            tm1 = tm1.to(device=device, dtype=dtype)
            t = t.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            out = model(tm2, tm1, t)
            loss = criterion(out, y) # TODO: list has no size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                print()

    torch.save(model.state_dict(), file)
    print("Training complete. Model saved to disk.")
    print()



def runModel(parms=None):

    """Train encoder from scratch"""
    print('CUDA enabled - ' + str(torch.cuda.is_available()))

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=4,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )

    # train_loader, val_loader, test_loader = loadData(spec, new=True, path='./data')
    # for i,sample in enumerate(train_loader):
    #     x = sample['images']
    #     y = sample['rewards']

    file = './model/encoder.pt'
    train_loader, val_loader, test_loader = loadData(spec, False, path='./data')
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    trainModel(model, optimizer, train_loader, file)


if __name__ == "__main__":

    params = {}
    runModel(params)