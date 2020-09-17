from sprites_datagen.moving_sprites import EncoderDataset, DecoderDataset
from sprites_datagen.rewards import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from general_utils import AttrDict


'''MLP module'''
class MLP(nn.Module):

    def __init__(self, params):
        """
        params:
            num_layers
            input_dim, l1_dim, l2_dim, ..., output_dim
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
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def save(self, file):
        torch.save(self.state_dict(), file)

    def restore(self, file):
        self.load_state_dict(torch.load(file))


'''Encoder/decoder stuff here'''
def loadVAEData(spec,
                path,
                decoder=False,
                new=False,
                batch_size=16,
                train_num=5000,
                val_num=1000,
                test_num=1000):

    """Return dataloaders for train/val/test datasets"""
    train = path + '/train.data'
    val = path + '/val.data'
    test = path + '/test.data'
    try:
        if decoder:
            train_dataset = DecoderDataset(spec, train_num, train) if new else torch.load(train)
            val_dataset = DecoderDataset(spec, val_num, val) if new else torch.load(val)
            test_dataset = DecoderDataset(spec, test_num, test) if new else torch.load(test)
        else:
            train_dataset = EncoderDataset(spec, train_num, train) if new else torch.load(train)
            val_dataset = EncoderDataset(spec, val_num, val) if new else torch.load(val)
            test_dataset = EncoderDataset(spec, test_num, test) if new else torch.load(test)
    except Exception as err:
        print('Error when loading dataset: ' + err)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def trainEncoder(model,
                 rewards,
                 optimizer,
                 scheduler,
                 loader_train,
                 loader_val,
                 file,
                 device,
                 dtype,
                 epochs=100,
                 print_every=100):

    """Training function"""
    print("Training Encoder starts.")
    print()
    model = model.to(device=device)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for i, sample in enumerate(loader_train):
            model.train()

            ts = [sample['images'][:,index,:,:,:].to(device=device, dtype=dtype) for index in range(10)]
            ys = [sample['rewards'][reward].narrow(1,10,20).to(device=device, dtype=dtype) for reward in rewards]
            # (1, predictor_params['input_sequence_length'], predictor_params['sequence_length'])

            out = model(ts)
            losses = [criterion(out[index], ys[index]) for index in range(7)]
            loss_encoder = sum(losses)

            optimizer.zero_grad()
            loss_encoder.backward()
            optimizer.step()

            if i % print_every == 0:
                print('Epoch %d, Iteration %d, encoder loss = %.4f' % (e, i, loss_encoder.item()))
                validateEncoder(model, rewards, loader_val, device)
                print()

        scheduler.step()

    torch.save(model.state_dict(), file)
    print("Training complete. Model saved to disk.")
    print()


def validateEncoder(model, rewards, loader, device, dtype=torch.float32):

    """Validation function"""
    criterion = nn.MSELoss()
    model = model.to(device=device)
    model.eval()
    loss_encoder = 0.0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            ts = [sample['images'][:,index,:,:,:].to(device=device, dtype=dtype) for index in range(10)]
            ys = [sample['rewards'][reward].narrow(1, 10, 20).to(device=device, dtype=dtype) for reward in rewards]

            out = model(ts)
            losses = [criterion(out[index], ys[index]) for index in range(7)]
            loss_encoder += sum(losses)
            count += 1

        print('Evaulation on validation dataset: Encoder got average loss: %.4f' % (loss_encoder / count))
        print()


def trainDecoder(model,
                 optimizer,
                 scheduler,
                 loader_train,
                 loader_val,
                 file,
                 device,
                 dtype,
                 vertical,
                 epochs=100,
                 print_every=100):

    """Training encoder first"""
    print("Training Decoder starts.")
    print()

    model = model.to(device=device)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for i, sample in enumerate(loader_train):
            model.train()

            ts = sample['images'].to(device=device, dtype=dtype)
            ys = sample['rewards']['vertical_position' if vertical else 'horizontal_position']\
                .to(device=device, dtype=dtype)

            out = model(ts)
            loss_decoder = criterion(out, ys) # TODO

            optimizer.zero_grad()
            loss_decoder.backward()
            optimizer.step()

            if i % print_every == 0:
                print('Epoch %d, Iteration %d, decoder loss = %.4f' % (e, i, loss_decoder.item()))
                validateDecoder(model, loader_val, vertical, device)
                print()

        scheduler.step()

    torch.save(model.state_dict(), file)
    print("Training complete. Model saved to disk.")
    print()


def validateDecoder(model, loader, vertical, device, dtype=torch.float32):

    """Validation function"""
    criterion = nn.MSELoss()
    model = model.to(device=device)
    model.eval()
    loss_decoder = 0.0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):

            ts = sample['images'].to(device=device, dtype=dtype)
            ys = sample['rewards']['vertical_position' if vertical else 'horizontal_position'] \
                .to(device=device, dtype=dtype)

            out = model(ts)
            loss_decoder += criterion(out, ys) # TODO
            count += 1

        print('Evaulation on validation dataset: Decoder got average loss: %.4f' % (loss_decoder / count))
        print()


'''RL stuff here'''
def convert_paths_to_components(paths):
    obs = np.concatenate([path['obs'] for path in paths])
    acs = np.concatenate([path['acs'] for path in paths])
    next_obs = np.concatenate([path['next_obs'] for path in paths])
    terminals = np.concatenate([path['terminals'] for path in paths])
    rews = np.concatenate([path['rews'] for path in paths])
    return obs, acs, next_obs, terminals, rews


def sample_trajectories(env, policy, max_path_length, min_timesteps_per_batch):
    # Question: why using min_timesteps_per_batch (?)
    timesteps_this_batch = 0
    paths = []
    while True:
        data = sample_trajectory(env, policy, max_path_length)
        paths.append(data)
        timesteps_this_batch += len(data['obs'])
        if timesteps_this_batch >= min_timesteps_per_batch:
            break
    return paths


def sample_trajectory(env, policy, max_path_length):
    ob = env.reset()
    obs, acs, next_obs, terminals, rews = [], [], [], [], []
    steps = 0

    while True:
        obs.append(ob)
        ac = policy.get_action(ob)
        acs.append(ac)

        ob, rew, done, _ = env.step(ac)
        next_obs.append(ob)
        rews.append(rew)

        steps += 1
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    data = {'obs': np.array(obs, dtype=np.float32),
            'acs': np.array(acs, dtype=np.float32),
            'next_obs': np.array(next_obs, dtype=np.float32),
            'terminals': np.array(terminals, dtype=np.float32),
            'rews': np.array(rews, dtype=np.float32)}
    return data


'''Driver program to generate encoder dataset'''
if __name__ == "__main__":

    # # complete encoder spec
    # spec = AttrDict(
    #     resolution=64,
    #     max_seq_len=30,
    #     max_speed=0.05,
    #     obj_size=0.2,
    #     shapes_per_traj=3,
    #     rewards=[ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    # )

    # decoder training spec
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=1,
        rewards=[VertPosReward, HorPosReward],
    )

    # Generate a small encoder dataset for overfitting test
    train_loader, val_loader, test_loader = loadVAEData(spec,
                                                        path='./data/data_decoder',
                                                        decoder=True,
                                                        new=True,
                                                        train_num=1000,
                                                        val_num=200,
                                                        test_num=200)
