from sprites_datagen.rewards import *
from training_utils import loadVAEData, trainVAERewardPrediction, trainVAEReconstruction, MLP
from general_utils import AttrDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse


class Encoder(nn.Module):
    """cnn encoder that reduces resolution by a factor of 2 in every layer"""

    def __init__(self):
        super().__init__()

        cnn_layers = [nn.Conv2d(1,4,kernel_size=4, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(4,8,kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                      nn.Conv2d(8,16,kernel_size=4, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(16,32,kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                      nn.Conv2d(32,64,kernel_size=4, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(64,128,kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()]
        self.layers = nn.ModuleList(cnn_layers)  # (batch, 1, 64, 64) => (batch, 128, 1, 1)
        self.mapping = nn.Linear(128, 64)

    def forward(self, x, rl=False):
        def naive_forward(x):
            out = x
            for layer in self.layers:
                out = layer(out)
            out = torch.reshape(out, (out.shape[0], -1)) # (batch, 128, 1, 1) => (batch, 128)
            out = self.mapping(out)  # (batch, 128) => (batch, 64)
            return out

        if rl:
            x = torch.unsqueeze(x, dim=1)  # insert 1 channel dim (batch, 64, 64) => (batch, 1, 64, 64)
        return naive_forward(x)


class EncoderCNN(nn.Module):

    def __init__(self):
        super().__init__()

        cnn_layers = [nn.Conv2d(1,16,kernel_size=3,stride=2), nn.ReLU(),
                      nn.Conv2d(16,16,kernel_size=3,stride=2), nn.ReLU(),
                      nn.Conv2d(16,16,kernel_size=3,stride=2), nn.ReLU()]
        self.layers = nn.ModuleList(cnn_layers)

    def forward(self, x, rl=True):
        out = torch.unsqueeze(x, dim=1)  # insert 1 channel dim
        for layer in self.layers:
            out = layer(out)
        out = torch.reshape(out, (out.shape[0], -1))  # flatten
        return out


class EncoderOracle(nn.Module):

    def __init__(self):
        super(EncoderOracle, self).__init__()

    def forward(self, x, rl=True):
        with torch.no_grad():
            return x


class Decoder(nn.Module):
    """cnn decoder that decodes the reward-induced representations"""

    def __init__(self):
        super().__init__()

        # self.sequence_length = 20
        # self.mlp_params = {'num_layers': 2, 'input_dim': 256, 'l1_dim': 64, 'output_dim': 128}
        # self.mlp = MLP(self.mlp_params)
        self.mapping = nn.Linear(64, 128)

        cnn_layers = [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                      nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                      nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                      nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1), nn.Tanh()]
        self.layers = nn.ModuleList(cnn_layers)

    def forward(self, x):
        out = self.mapping(x)
        out = out.unsqueeze(2).unsqueeze(3)
        for layer in self.layers:
            out = layer(out)
        return out


class Predictor(nn.Module):
    """LSTM predictor for reward preiction"""

    def __init__(self, params):
        super().__init__()

        self.input_sequence_length = params['input_sequence_length']
        self.output_sequence_length = params['output_sequence_length']
        self.sequence_length = self.input_sequence_length + self.output_sequence_length

        self.mlp_params = {'num_layers': 3,
                           'input_dim': 64 * self.input_sequence_length,
                           'l1_dim': 512,
                           'l2_dim': 256,
                           'output_dim': 512}
        self.init_hidden = MLP(self.mlp_params)
        self.init_cell = MLP(self.mlp_params)

        self.lstm = nn.LSTM(64, 512, batch_first=True)
        self.mapping = nn.Linear(512, 64)  # hidden_state => embedding

    def forward(self, zts):
        # z_cat = torch.cat(zs, dim=1)
        # z_embed = self.mlp(z_cat)
        # embed_stack = torch.stack([z_embed] * self.sequence_length, dim=1)
        # lstm_out, _ = self.lstm(embed_stack)  # (batch_size, seq_len, hidden_dim)
        # return lstm_out

        # batch_size = zts.shape[0]
        # z_cat = zts.reshape(batch_size,-1)  # (batch, input_seq * 64)
        # hidden_init = self.mlp(z_cat).reshape(1,batch_size,512)  # (1, batch, 512)
        # hidden = (hidden_init, torch.zeros(1,batch_size,512).cuda())  # initialized hidden state
        # out, hidden = self.lstm(zts, hidden)  # (batch_size, seq_len, hidden_dim)

        # zts = unbatchify(zts_batch, self.sequence_length, toTensor=True)  # (batch, seq, 64)
        # out, _ = self.lstm(zts)  # Question: need clarification
        # out = self.mapping(out)
        # return out

        batch_size = zts.size(0)
        mlp_input = zts.reshape(batch_size, -1)
        hidden = self.init_hidden(mlp_input)
        cell = self.init_cell(mlp_input)
        hidden = torch.unsqueeze(hidden, dim=0)  # (1, 32, 512)
        cell = torch.unsqueeze(cell, dim=0)  # (1, 32, 512)
        input = torch.zeros(batch_size, self.output_sequence_length, 64).cuda()  # (batch, 20, 64)
        out, _ = self.lstm(input, (hidden, cell))  # (batch, 20, 512)
        out = self.mapping(out)  # (batch, 20, 64)
        return out


class VAEReconstructionModel(nn.Module):

    def __init__(self):
        super(VAEReconstructionModel, self).__init__()

        # Note: don't detach between encoder, decoder, only use decoder loss

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.predictor_params = {'input_sequence_length': 10, 'sequence_length': 20}
        self.predictor = Predictor(self.predictor_params)

    def forward(self, ts, rl=False):

        if rl:
            with torch.no_grad():
                ts = torch.unsqueeze(ts, dim=1)  # insert 1 channel dim
                zts = self.encoder(ts)
                return zts
        else:
            ts_batch, num = batchify(ts)
            zts_batch = self.encoder(ts_batch)
            zts = unbatchify(zts_batch, num)
            h1tT = self.predictor(zts)
            h1tT_batch, size = batchify(h1tT)
            h1tT_decode_batch = self.decoder(h1tT_batch)
            h1tT_decode = unbatchify(h1tT_decode_batch, size)
            return h1tT_decode


class VAERewardPredictionModel(nn.Module):
    """complete architecture"""

    def __init__(self, num_reward_heads=4, train_decoder=False):
        super().__init__()

        self.encoder = Encoder()
        self.train_decoder = train_decoder
        if train_decoder:
            self.decoder = Decoder()

        self.predictor_params = {'input_sequence_length': 10, 'output_sequence_length': 20}
        self.predictor = Predictor(self.predictor_params)

        self.reward_mlp_params = {'num_layers': 3, 'input_dim': 64, 'l1_dim': 256, 'l2_dim': 128, 'output_dim': 1}
        self.K = num_reward_heads
        reward_heads = []

        for i in range(self.K):
            reward_head = MLP(self.reward_mlp_params)
            reward_heads.append(reward_head)
        self.reward_heads = nn.ModuleList(reward_heads)

    def forward(self, ts, rl=False):
        if rl:
            with torch.no_grad():
                ts = torch.unsqueeze(ts, dim=1)  # insert 1 channel dim
                zts = self.encoder(ts)
                return zts

        ts_batch, num = batchify(ts)
        zts_batch = self.encoder(ts_batch)
        zts = unbatchify(zts_batch, num, toTensor=True)

        h1tT = self.predictor(zts)
        h1tT_batch, size = batchify(h1tT)

        reward_estimates = []
        for head in self.reward_heads:
            rewards_batch = head(h1tT_batch)
            rewards = unbatchify(rewards_batch, size, toTensor=True)
            rewards = torch.squeeze(rewards, dim=2)
            reward_estimates.append(rewards)

        if self.train_decoder:
            h1tT_batch_detach = h1tT_batch.detach()
            h1tT_decode_batch = self.decoder(h1tT_batch_detach)
            h1tT_decode = unbatchify(h1tT_decode_batch, size)
            return reward_estimates, h1tT_decode

        return reward_estimates


def batchify(input):
    # convert an iterable to a batch to speed up computation
    # returns: out - batchified input, l - compressed dim
    if type(input) == list:
        l = len(input)
        out = torch.cat(input, dim=0)
        return out, l
    elif type(input) == torch.Tensor:
        # (batch_size, seq_len, hidden_dim)
        l = input.size(1)
        out = torch.reshape(input, (input.size(0) * input.size(1), input.size(2)))
        return out, l


def unbatchify(input, num, toTensor=False):
    batch_size = int(input.size(0) / num)
    if toTensor:
        # convert a batch back to original dim
        out = torch.reshape(input, (batch_size, int(num), int(input.size(1))))
        return out
    else:
        # convert a batch back to a list
        out = torch.split(input, batch_size, dim=0)
        return list(out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=30)
    parser.add_argument('--max_speed', type=float, default=0.05)
    parser.add_argument('--obj_size', type=float, default=0.2)
    parser.add_argument('--shapes_per_traj', '-spt', type=int, default=3)
    parser.add_argument('--reward_indices', '-r', type=int, nargs='*')
    parser.add_argument('--model_path', '-mp', type=str)
    parser.add_argument('--load_model', '-lm', action='store_true')
    parser.add_argument('--train_decoder', '-td', action='store_true')
    parser.add_argument('--save_to_disk', action='store_true')
    parser.add_argument('--data_path', '-dp', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--epochs', '-e', type=int, default=150)
    parser.add_argument('--reconstruction', '-recon', action='store_true')
    args = parser.parse_args()

    return args


def main(DEBUG=False):
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    rewards = {
        'rewards_class':
            [ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
        'rewards_name':
            ['zero', 'vertical_position', 'horizontal_position', 'agent_x', 'agent_y', 'target_x', 'target_y']
    }

    reward_classes = np.array(rewards['rewards_class'])[args.reward_indices] \
        if args.reward_indices else np.array(rewards['rewards_class'])
    reward_names = np.array(rewards['rewards_name'])[args.reward_indices] \
        if args.reward_indices else None
    num_reward_heads = len(args.reward_indices) if args.reward_indices else None

    spec = AttrDict(
        resolution=args.resolution,
        max_seq_len=args.max_seq_len,
        max_speed=args.max_speed,
        obj_size=args.obj_size,
        shapes_per_traj=args.shapes_per_traj,
        rewards=reward_classes
    )

    # train_loader, val_loader, test_loader = loadVAEData(spec, batch_size=args.batch_size)
    # DEBUG:
    train_loader, val_loader, test_loader = loadVAEData(spec, save_to_disk=True, batch_size=args.batch_size)

    if args.reconstruction:
        model = VAEReconstructionModel()
    else:
        model = VAERewardPredictionModel(num_reward_heads, args.train_decoder)

    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
        if args.reconstruction:
            print('VAEReconstruction model loaded\n')
        else:
            print('VAERewardPrediction model loaded\n')

    if not DEBUG:
        wandb.init(project="impl_jiefan")
        wandb.config.update(args)
        wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.reconstruction:
        trainVAEReconstruction(model,
                               optimizer,
                               train_loader,
                               val_loader,
                               args.model_path,
                               device,
                               dtype,
                               epochs=args.epochs,
                               DEBUG=DEBUG)
    else:
        trainVAERewardPrediction(model,
                                 reward_names,
                                 args.train_decoder,
                                 optimizer,
                                 train_loader,
                                 val_loader,
                                 args.model_path,
                                 device,
                                 dtype,
                                 epochs=args.epochs,
                                 DEBUG=DEBUG)


if __name__ == "__main__":

    DEBUG=False
    main(DEBUG=DEBUG)
