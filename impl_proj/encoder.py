from sprites_datagen.rewards import *
from training_utils import loadVAEData, trainEncoder, trainDecoder, MLP
from general_utils import AttrDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """cnn encoder that reduces resolution by a factor of 2 in every layer"""

    def __init__(self, encoder_params=None):
        super().__init__()

        self.num_layers = 6
        num_channels = 4
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

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = torch.squeeze(out) # if batch size = 1 it will break, give it axis
        out = self.mapping(out)
        return out


class Decoder(nn.Module):
    """cnn decoder that decodes the reward-induced representations"""

    def __init__(self, decoder_params=None):
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

    def __init__(self, predictor_params):
        super().__init__()

        self.input_size = 128 # predictor_params['lstm_size'] # assert == embed_stack.shape(0)
        self.hidden_size = 256 # predictor_params['hidden_size']
        self.input_sequence_length = predictor_params['input_sequence_length']
        self.sequence_length = predictor_params['sequence_length']

        self.mlp_params = {}
        self.mlp_params['num_layers'] = 3
        self.mlp_params['input_dim'] = 64 * self.input_sequence_length
        self.mlp_params['l1_dim'] = 512
        self.mlp_params['l2_dim'] = 256
        self.mlp_params['output_dim'] = self.input_size

        self.mlp = MLP(self.mlp_params)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, zs):
        z_cat = torch.cat(zs, dim=1)
        z_embed = self.mlp(z_cat)
        embed_stack = torch.stack([z_embed] * self.sequence_length, dim=1)
        lstm_out, _  = self.lstm(embed_stack) # (batch_size, seq_len, hidden_dim)
        return lstm_out


class EncoderModel(nn.Module):
    """complete architecture"""

    def __init__(self, model_params):
        super().__init__()

        self.encoder = Encoder()

        self.predictor_params = model_params['predictor_params']
        self.predictor = Predictor(self.predictor_params)

        self.reward_params = model_params['reward_mlp_params']
        self.K = model_params['reward_heads_num']
        reward_heads = []
        for i in range(self.K):
            reward_head = MLP(self.reward_params)
            reward_heads.append(reward_head)
        self.reward_heads = nn.ModuleList(reward_heads)

    def forward(self, ts, decode_flag=False):
        zts = [self.encoder(t) for t in ts]

        if decode_flag:
            return [zt.detach() for zt in zts]

        h1tT = self.predictor(zts) # (batch_size, seq_len, hidden_dim)
        reward_estimates = []
        for head in self.reward_heads:
            rewards = []
            # use batchify, and batchify back
            for t in range(h1tT.shape[1]):
                input = h1tT[:,t,:]
                rewards.append(head(input))
            reward_estimate = torch.cat(rewards, dim=1)
            reward_estimates.append(reward_estimate)

        return reward_estimates


class DecoderModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder_model = None

    def load_encoder(self, encoder_model):
        self.encoder_model = encoder_model

    def forward(self, ts):
        zts = self.encoder_model(ts, decode_flag=True)
        zts_decode = [self.decoder(zt) for zt in zts]
        return zts_decode


def run(params):
    """Train encoder from scratch"""

    # Training model
    spec = params['spec']
    device = params['device']
    dtype = params['dtype']
    model_params = params['model_params']
    file = params['model_path']
    new_file = params['new_model_path'] if params['new_model_path'] else params['model_path']
    data_path = params['data_path']
    lr = params['learning_rate']
    gamma = params['scheduler_gamma']
    step_size = params['scheduler_step_size']
    weight_decay = params['weight_decay']
    epochs = params['epochs']
    load_model = params['load_model']
    load_new_data = params['load_new_data']
    rewards = params['rewards']

    train_decoder = params['train_decoder']
    encoder_vert_file = params['encoder_vert_path']
    encoder_hori_file = params['encoder_hori_path']
    decoder_vert_file = params['decoder_vert_path']
    decoder_hori_file = params['decoder_hori_path']
    load_decoder_models = params['load_decoder_models']
    train_decoder_only = params['train_decoder_only']
    train_encoder_only = params['train_encoder_only']

    train_loader, val_loader, test_loader = \
        loadVAEData(spec, data_path, decoder=train_decoder, new=load_new_data)

    if not train_decoder:
        model = EncoderModel(model_params)
    else:
        encoder_vert = EncoderModel(model_params)
        encoder_hori = EncoderModel(model_params)
        decoder_vert = DecoderModel()
        decoder_hori = DecoderModel()

    if load_model and not train_decoder:
        model.load_state_dict(torch.load(file))
        print('Encoder model loaded')
    if load_model and train_decoder:
        encoder_vert.load_state_dict(torch.load(encoder_vert_file))
        encoder_hori.load_state_dict(torch.load(encoder_hori_file))
        print('Encoder models (vert & hori) loaded')
        decoder_vert.load_encoder(encoder_vert)
        decoder_hori.load_encoder(encoder_hori)
        print('Encoder models loaded into decoders (vert & hori)')

    if load_decoder_models:
        decoder_vert.load_state_dict(torch.load(decoder_vert_file))
        decoder_hori.load_state_dict(torch.load(decoder_hori_file))
        print('Decoder models (vert & hori) loaded')

    if not train_decoder:
        # train encoder model
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        trainEncoder(model,
                     rewards,
                     optimizer,
                     scheduler,
                     train_loader,
                     val_loader,
                     new_file,
                     device,
                     dtype,
                     epochs=epochs)
    else:
        # train decoder model, first train encoder
        if not train_decoder_only:
            optimizer_encoder_vert = optim.Adam(encoder_vert.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_encoder_hori = optim.Adam(encoder_hori.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler_encoder_vert = optim.lr_scheduler.ExponentialLR(optimizer_encoder_vert, gamma)
            scheduler_encoder_hori = optim.lr_scheduler.ExponentialLR(optimizer_encoder_hori, gamma)

            trainEncoder(encoder_vert,
                         ['vertical_position'],
                         optimizer_encoder_vert,
                         scheduler_encoder_vert,
                         train_loader,
                         val_loader,
                         encoder_vert_file,
                         device,
                         dtype,
                         epochs=epochs)
            print('Encoder_vert model finished training')
            trainEncoder(encoder_hori,
                         ['horizontal_position'],
                         optimizer_encoder_hori,
                         scheduler_encoder_hori,
                         train_loader,
                         val_loader,
                         encoder_hori_file,
                         device,
                         dtype,
                         epochs=epochs)
            print('Encoder_hori model finished training')
            decoder_vert.load_encoder(encoder_vert)
            decoder_hori.load_encoder(encoder_hori)
            print('Encoder models loaded into decoders (vert & hori)')

        if not train_encoder_only:
            # then train decoder
            optimizer_decoder_vert = optim.Adam(decoder_vert.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_decoder_hori = optim.Adam(decoder_hori.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler_decoder_vert = optim.lr_scheduler.ExponentialLR(optimizer_decoder_vert, gamma)
            scheduler_decoder_hori = optim.lr_scheduler.ExponentialLR(optimizer_decoder_hori, gamma)

            trainDecoder(decoder_vert,
                         optimizer_decoder_vert,
                         scheduler_decoder_vert,
                         train_loader,
                         val_loader,
                         decoder_vert_file,
                         device,
                         dtype,
                         epochs=epochs)
            print('Decoder_vert model finished training')
            trainDecoder(decoder_hori,
                         optimizer_decoder_hori,
                         scheduler_decoder_hori,
                         train_loader,
                         val_loader,
                         decoder_hori_file,
                         device,
                         dtype,
                         epochs=epochs)
            print('Decoder_hori model finished training')


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # encoder training spec
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=3,
        rewards=[ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )

    # # decoder training spec
    # spec = AttrDict(
    #     resolution=64,
    #     max_seq_len=30,
    #     max_speed=0.05,
    #     obj_size=0.2,
    #     shapes_per_traj=1,
    #     rewards=[VertPosReward, HorPosReward],
    # )

    params = {}

    # predictor params
    predictor_params = {}
    predictor_params['input_sequence_length'] = 10
    predictor_params['sequence_length'] = 20

    # reward params
    reward_mlp_params = {}
    reward_mlp_params['num_layers'] = 3
    reward_mlp_params['input_dim'] = 256  # assert == predictor.hidden_size
    reward_mlp_params['l1_dim'] = 256
    reward_mlp_params['l2_dim'] = 64
    reward_mlp_params['output_dim'] = 1

    # encoder_model params
    model_params = {}
    model_params['predictor_params'] = predictor_params
    model_params['reward_mlp_params'] = reward_mlp_params
    model_params['reward_heads_num'] = 7 # encoder: 7
    params['model_params'] = model_params

    # encoder params
    params['model_path'] = './models/encoder_09_17.pt'
    params['rewards'] = \
        ['zero', 'vertical_position', 'horizontal_position', 'agent_x', 'agent_y', 'target_x', 'target_y']
    # decoder params
    params['encoder_vert_path'] = './models/encoder_vert_09_17.pt'
    params['encoder_hori_path'] = './models/encoder_hori_09_17.pt'
    params['decoder_vert_path'] = './models/decoder_vert_09_17.pt'
    params['decoder_hori_path'] = './models/decoder_hori_09_17.pt'

    # training params (seldom changed)
    params['spec'] = spec
    params['device'] = device
    params['dtype'] = dtype
    params['new_model_path'] = None
    params['load_new_data'] = False
    # training params (need changing)
    params['data_path'] = './data/data_decoder' # encoder: './data/data_encoder'
    params['load_model'] = True # load encoder_models
    params['load_decoder_models'] = False # load decoder_models
    params['train_decoder'] = False  # encoder: False
    params['train_decoder_only'] = False
    params['train_encoder_only'] = False

    # hyper params
    params['learning_rate'] = 1e-3 # 3.5e-4
    params['scheduler_gamma'] = 0.9
    params['scheduler_step_size'] = 500
    params['weight_decay'] = 1e-3
    params['epochs'] = 200

    run(params)
