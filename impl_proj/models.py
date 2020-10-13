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

        self.num_layers = 6
        num_channels = 4
        cnn_layers = []

        for i in range(self.num_layers):
            cnn_layers.append(nn.Conv2d(num_channels if i != 0 else 1, num_channels * 2 if i != 0 else num_channels,
                                        kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels * 2 if i != 0 else num_channels))
            cnn_layers.append(nn.LeakyReLU())
            if i != 0:
                num_channels *= 2

        self.layers = nn.ModuleList(cnn_layers)
        self.mapping = nn.Linear(128, 64)

    def forward(self, x, rl=False):
        def naive_forward(x):
            out = x
            for layer in self.layers:
                out = layer(out)
            out = torch.squeeze(out, dim=2)  # use dim (batch can be 1)
            out = torch.squeeze(out, dim=2)
            out = self.mapping(out)
            return out

        if rl:
            with torch.no_grad():
                x = torch.unsqueeze(x, dim=1) # insert 1 channel dim
                return naive_forward(x)
        else:
            return naive_forward(x)


class EncoderCNN(nn.Module):

    def __init__(self):
        super().__init__()

        cnn_layers = [nn.Conv2d(1,16,kernel_size=3,stride=2), nn.ReLU(),
                      nn.Conv2d(16,16,kernel_size=3,stride=2), nn.ReLU(),
                      nn.Conv2d(16,16,kernel_size=3,stride=2), nn.ReLU()]
        self.layers = nn.ModuleList(cnn_layers)

    def forward(self, x, rl=True):
        def naive_forward(x):
            out = x
            for layer in self.layers:
                out = layer(out)
            out = torch.reshape(out, (out.shape[0],-1)) # flatten
            return out

        if rl:
            with torch.no_grad(): # Question: do we finetune i.e. train the cnn?
                x = torch.unsqueeze(x, dim=1) # insert 1 channel dim
                return naive_forward(x)
        else:
            return naive_forward(x)


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

        self.decoder_input = nn.Linear(64, 128)
        self.num_layers = 6
        num_channels = 128

        cnn_layers = []
        for i in range(self.num_layers):
            cnn_layers.append(nn.ConvTranspose2d(num_channels, num_channels // 2 if i != self.num_layers - 1 else 1,
                                                 kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels // 2 if i != self.num_layers - 1 else 1))
            cnn_layers.append(nn.LeakyReLU())
            if i != self.num_layers - 1:
                num_channels //= 2
        self.layers = nn.ModuleList(cnn_layers)

    def forward(self, x):
        out = self.decoder_input(x)
        out = out.unsqueeze(2).unsqueeze(3)
        for layer in self.layers:
            out = layer(out)
        return out


class Predictor(nn.Module):
    """LSTM predictor for reward preiction"""

    def __init__(self, params):
        super().__init__()

        self.input_size = 128
        self.hidden_size = 256
        self.input_sequence_length = params['input_sequence_length']
        self.sequence_length = params['sequence_length']

        self.mlp_params = {'num_layers': 3, 'input_dim': 64 * self.input_sequence_length, 'l1_dim': 512, 'l2_dim': 256,
                           'output_dim': self.input_size}
        self.mlp = MLP(self.mlp_params)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, zs):
        z_cat = torch.cat(zs, dim=1)
        z_embed = self.mlp(z_cat)
        embed_stack = torch.stack([z_embed] * self.sequence_length, dim=1)
        lstm_out, _ = self.lstm(embed_stack)  # (batch_size, seq_len, hidden_dim)
        return lstm_out


class VAEReconstructionModel(nn.Module):

    def __init__(self):
        super(VAEReconstructionModel, self).__init__()

        # Note: don't detach between encoder, decoder, only use decoder loss

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, ts, rl=False):

        def naive_forward(ts):
            ts_batch, num = batchify(ts)
            zts_batch = self.encoder(ts_batch)
            zts_decode_batch = self.decoder(zts_batch)
            zts_decode = unbatchify(zts_decode_batch, num)
            return zts_decode

        if rl:
            with torch.no_grad():
                return naive_forward(ts)
        else:
            return naive_forward(ts)


class VAERewardPredictionModel(nn.Module):
    """complete architecture"""

    def __init__(self, num_reward_heads=7, train_decoder=False):
        super().__init__()

        self.encoder = Encoder()
        self.train_decoder = train_decoder
        if self.train_decoder:
            self.decoder = Decoder()

        self.predictor_params = {'input_sequence_length': 10, 'sequence_length': 20}
        self.predictor = Predictor(self.predictor_params)

        self.reward_mlp_params = {'num_layers': 3, 'input_dim': 256, 'l1_dim': 256, 'l2_dim': 64, 'output_dim': 1}
        self.K = num_reward_heads
        reward_heads = []

        for i in range(self.K):
            reward_head = MLP(self.reward_mlp_params)
            reward_heads.append(reward_head)
        self.reward_heads = nn.ModuleList(reward_heads)

    def forward(self, ts, rl=False):
        if rl:
            with torch.no_grad():
                ts_batch, num = batchify(ts)
                ts_batch = torch.unsqueeze(ts_batch, dim=1) # insert 1 channel dim
                zts_batch = self.encoder(ts_batch)
                zts = unbatchify(zts_batch, num)
                return zts

        ts_batch, num = batchify(ts)
        zts_batch = self.encoder(ts_batch)
        zts = unbatchify(zts_batch, num)

        h1tT = self.predictor(zts)  # (batch_size, seq_len, hidden_dim)
        h1tT_batch, size = batchify(h1tT)

        reward_estimates = []
        for head in self.reward_heads:
            rewards_batch = head(h1tT_batch)
            rewards = unbatchify(rewards_batch, size, toTensor=True)
            rewards = torch.squeeze(rewards, dim=2)
            reward_estimates.append(rewards)

        if self.train_decoder:
            zts_detach = [zt.detach() for zt in zts]
            zts_detach_batch, num = batchify(zts_detach)
            zts_decode_batch = self.decoder(zts_detach_batch)
            zts_decode = unbatchify(zts_decode_batch, num)
            return reward_estimates, zts_decode
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

    train_loader, val_loader, test_loader = loadVAEData(spec, batch_size=args.batch_size)

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
