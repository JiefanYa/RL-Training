from sprites_datagen.rewards import *
from training_utils import loadVAEData, trainEncoderDecoder, MLP
from general_utils import AttrDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


class Encoder(nn.Module):
    """cnn encoder that reduces resolution by a factor of 2 in every layer"""

    def __init__(self):
        super().__init__()

        self.num_layers = 6
        num_channels = 4
        cnn_layers = []

        for i in range(self.num_layers):
            cnn_layers.append(nn.Conv2d(num_channels if i != 0 else 3, num_channels * 2 if i != 0 else num_channels,
                                        kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels * 2 if i != 0 else num_channels))
            cnn_layers.append(nn.LeakyReLU())
            if i != 0:
                num_channels *= 2

        self.layers = nn.ModuleList(cnn_layers)
        self.mapping = nn.Linear(128, 64)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = torch.squeeze(out, dim=2) # use dim (batch can be 1)
        out = torch.squeeze(out, dim=2)
        out = self.mapping(out)
        return out


class Decoder(nn.Module):
    """cnn decoder that decodes the reward-induced representations"""

    def __init__(self):
        super().__init__()

        self.decoder_input = nn.Linear(64, 128)
        self.num_layers = 6
        num_channels = 128

        cnn_layers = []
        for i in range(self.num_layers):
            cnn_layers.append(nn.ConvTranspose2d(num_channels, num_channels // 2 if i != self.num_layers - 1 else 3,
                                                 kernel_size=4, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(num_channels // 2 if i != self.num_layers - 1 else 3))
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


class EncoderDecoderModel(nn.Module):
    """complete architecture"""

    def __init__(self, num_reward_heads, train_decoder=False):
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

    def forward(self, ts):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=30)
    parser.add_argument('--max_speed', type=float, default=0.05)
    parser.add_argument('--obj_size', type=float, default=0.2)
    parser.add_argument('--shapes_per_traj', '-spt', type=int, default=3)
    parser.add_argument('--reward_indices', '-r', type=int, nargs='*')
    parser.add_argument('--num_reward_heads', '-nr', type=int, default=7)
    parser.add_argument('--model_path', '-mp', type=str)
    parser.add_argument('--load_model', '-lm', action='store_true')
    parser.add_argument('--train_decoder', '-td', action='store_true')
    parser.add_argument('--save_to_disk', action='store_true')
    parser.add_argument('--data_path', '-dp', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--use_scheduler', '-us', action='store_true')
    parser.add_argument('--gamma', '-g', type=float, default=0.9)
    parser.add_argument('--step_size', '-ss', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', '-e', type=int, default=150)
    args = parser.parse_args()
    params = vars(args)

    wandb.config.update(args)

    num_reward_heads = params['num_reward_heads']
    reward_indices = params['reward_indices']
    model_path = params['model_path']
    load_model = params['load_model']
    train_decoder = params['train_decoder']
    save_to_disk = params['save_to_disk']
    data_path = params['data_path']
    batch_size = params['batch_size']
    lr = params['learning_rate']
    use_scheduler = params['use_scheduler']
    gamma = params['gamma']
    step_size = params['step_size']
    weight_decay = params['weight_decay']
    epochs = params['epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    rewards = {
        'rewards_class':
            [ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
        'rewards_name':
            ['zero', 'vertical_position', 'horizontal_position', 'agent_x', 'agent_y', 'target_x', 'target_y']
    }
    reward_classes = np.array(rewards['rewards_class'])[reward_indices]
    reward_names = np.array(rewards['rewards_name'])[reward_indices]

    spec = AttrDict(
        resolution=params['resolution'],
        max_seq_len=params['max_seq_len'],
        max_speed=params['max_speed'],
        obj_size=params['obj_size'],
        shapes_per_traj=params['shapes_per_traj'],
        rewards=reward_classes
    )

    if save_to_disk:
        train_loader, val_loader, test_loader = loadVAEData(spec,
                                                            save_to_disk=True,
                                                            path=data_path,
                                                            decoder=train_decoder,
                                                            batch_size=batch_size)
    else:
        train_loader, val_loader, test_loader = loadVAEData(spec, batch_size=batch_size)

    model = EncoderDecoderModel(num_reward_heads, train_decoder)

    if load_model:
        model.load_state_dict(torch.load(model_path))
        print('EncoderDecoder model loaded')

    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma) if use_scheduler else None
    trainEncoderDecoder(model,
                        reward_names,
                        train_decoder,
                        optimizer,
                        scheduler,
                        train_loader,
                        val_loader,
                        model_path,
                        device,
                        dtype,
                        epochs=epochs)


if __name__ == "__main__":

    import os
    wandb.init(project="impl_jiefan")
    main()
