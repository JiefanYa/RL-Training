from sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward
from training_utils import loadEncoderData, trainEncoder, MLP
from general_utils import AttrDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """cnn encoder that reduces resolution by a factor of 2 in every layer"""

    def __init__(self, encoder_params=None):
        super().__init__()

        self.num_layers = 6 # self.num_layers = np.log2(encoder_params['resolution'])
        num_channels = 4 # num_channels = encoder_params['first_num_channels']
        cnn_layers = []

        for i in range(self.num_layers):
            cnn_layers.append(nn.Conv2d(num_channels if i!=0 else 3, num_channels*2 if i!=0 else num_channels,
                                        kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels*2 if i!=0 else num_channels))
            cnn_layers.append(nn.LeakyReLU())
            if i!=0:
                num_channels *= 2

        self.layers = nn.ModuleList(cnn_layers)
        self.mapping = nn.Linear(128, 64)
        # nn.Linear(num_channels*np.power(2,self.num_layers), encoder_params['obs_dim'])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = torch.squeeze(out)
        out = self.mapping(out)
        return out


class Decoder(nn.Module):
    """cnn decoder that decodes the reward-induced representations"""

    def __init__(self, decode_params=None):
        super().__init__()

        self.decoder_input = nn.Linear(64, 128)
        self.num_layers = 6
        num_channels = 128

        cnn_layers = []
        for i in range(self.num_layers):
            cnn_layers.append(nn.ConvTranspose2d(num_channels, num_channels//2 if i!=self.num_layers-1 else 3,
                                                 kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels//2 if i!=self.num_layers-1 else 3))
            cnn_layers.append(nn.LeakyReLU())
            if i!= self.num_layers-1:
                num_channels //= 2
        self.layers = nn.ModuleList(cnn_layers)

    def forward(self, x):
        input = self.decoder_input(x)
        out = input.unsqueeze(2).unsqueeze(3)
        for layer in self.layers:
            out = layer(out)
        return out


class Predictor(nn.Module):
    """LSTM predictor for reward preiction"""

    def __init__(self, predictor_params=None):
        super().__init__()

        self.input_size = 128 # predictor_params['lstm_size'] # assert == embed_stack.shape
        self.hidden_size = 256 # predictor_params['hidden_size']
        self.sequence_length = 27 # predictor_params['sequence_length']

        self.mlp_params = {} # predictor_params['mlp_params']
        self.mlp_params['num_layers'] = 3
        self.mlp_params['input_dim'] = 192
        self.mlp_params['l1_dim'] = 512
        self.mlp_params['l2_dim'] = 256
        self.mlp_params['output_dim'] = 128

        self.mlp = MLP(self.mlp_params)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, ztm2, ztm1, z):
        z_cat = torch.cat((ztm2, ztm1, z),dim=1)
        z_embed = self.mlp(z_cat)
        embed_stack = torch.stack([z_embed] * self.sequence_length, dim=1)
        lstm_out, _  = self.lstm(embed_stack) # (batch_size, seq_len, hidden_dim)
        return lstm_out


class EncoderModel(nn.Module):
    """complete architecture"""

    def __init__(self, model_params=None):
        super().__init__()

        # self.encoder_params = {} # model_params['encoder_params]
        # implement encoder_params
        self.encoder = Encoder()
        # self.decoder = Decoder()

        # self.predictor_params = {} # model_params['predictor_params]
        # implement predictor_params
        self.predictor = Predictor()

        self.reward_params = {} # model_params['reward_mlp_params']
        self.reward_params['num_layers'] = 3
        self.reward_params['input_dim'] = 256 # assert == self.predictor.hidden_size
        self.reward_params['l1_dim'] = 256
        self.reward_params['l2_dim'] = 64
        self.reward_params['output_dim'] = 1

        reward_heads = []
        for i in range(4): # self.K = model_params['K']
            reward_head = MLP(self.reward_params) # AgentXReward, AgentYReward, TargetXReward, TargetYReward
            reward_heads.append(reward_head)
        self.reward_heads = nn.ModuleList(reward_heads)


    def forward(self, tm2, tm1, t):
        ztm2 = self.encoder(tm2)
        ztm1 = self.encoder(tm1)
        zt = self.encoder(t)

        # ztm2_decode = self.decoder(ztm2.detach())
        # ztm1_decode = self.decoder(ztm1.detach())
        # zt_decode = self.decoder(zt.detach())
        # decoder_output = [ztm2_decode, ztm1_decode, zt_decode]

        h1tT = self.predictor(ztm2, ztm1, zt) # (batch_size, seq_len, hidden_dim)
        reward_estimates = []
        for head in self.reward_heads:
            rewards = []
            for t in range(h1tT.shape[1]):
                input = h1tT[:,t,:]
                rewards.append(head(input))
            reward_estimate = torch.cat(rewards, dim=1)
            reward_estimates.append(reward_estimate)

        return reward_estimates
        # return reward_estimates, decoder_output


def run(parms=None):
    """Train encoder from scratch"""

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=4,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # dtype = torch.float32

    # Training model
    file = './models/encoder.pt'
    train_loader, val_loader, test_loader = loadEncoderData(spec, False, path='./data_large')
    model = EncoderModel()
    model.load_state_dict(torch.load(file))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    trainEncoder(model, optimizer, train_loader, val_loader, file, device, epochs=500)

    # Trying to overfit a small dataset
    # file = './models/encoder_overfit.pt'
    # train_loader, val_loader, test_loader = loadEncoderData(spec, False, batch_size=4)
    # model = EncoderModel()
    # model.load_state_dict(torch.load(file))
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    # trainEncoder(model, optimizer, train_loader, val_loader, file, device, epochs=200, print_every=5)


if __name__ == "__main__":

    params = {}
    run(params)