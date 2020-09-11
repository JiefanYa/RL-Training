from sprites_datagen.moving_sprites import ImplDataset
from sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward
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


'''Encoder/decoder stuff here'''
def loadEncoderData(spec, new=False, path='./data', batch_size=16, train_num=5000, val_num=1000, test_num=1000):

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


def trainEncoder(model,
                 optimizer,
                 loader_train,
                 loader_val,
                 file,
                 device,
                 dtype=torch.float32,
                 epochs=100,
                 print_every=100):

    """Training function"""
    print("Training starts.")
    print()
    model = model.to(device=device)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for i, sample in enumerate(loader_train):
            model.train()

            tm2 = sample['images'][:,0,:,:,:].to(device=device, dtype=dtype)
            tm1 = sample['images'][:,1,:,:,:].to(device=device, dtype=dtype)
            t = sample['images'][:,2,:,:,:].to(device=device, dtype=dtype)

            yax = sample['rewards']['agent_x'].narrow(1,3,27).to(device=device, dtype=dtype)
            yay = sample['rewards']['agent_y'].narrow(1,3,27).to(device=device, dtype=dtype)
            ytx = sample['rewards']['target_x'].narrow(1,3,27).to(device=device, dtype=dtype)
            yty = sample['rewards']['target_y'].narrow(1,3,27).to(device=device, dtype=dtype)

            out = model(tm2, tm1, t)
            # out, decoder_out = model(tm2, tm1, t)
            loss_ax = criterion(out[0], yax)
            loss_ay = criterion(out[1], yay)
            loss_tx = criterion(out[2], ytx)
            loss_ty = criterion(out[3], yty)
            loss_encoder = loss_ax + loss_ay + loss_tx + loss_ty

            # loss_tm2 = criterion(decoder_out[0], tm2)
            # loss_tm1 = criterion(decoder_out[1], tm1)
            # loss_t = criterion(decoder_out[2], t)
            # loss_decoder = loss_tm2 + loss_tm1 + loss_t

            optimizer.zero_grad()
            loss_encoder.backward()
            # loss_decoder.backward()
            optimizer.step()

            if i % print_every == 0:
                # print('Epoch %d, Iteration %d, encoder loss = %.4f, decoder loss = %.4f'
                #       % (e, i, loss_encoder.item(), loss_decoder.item()))
                print('Epoch %d, Iteration %d, encoder loss = %.4f' % (e, i, loss_encoder.item()))
                validateEncoder(model, loader_val, device)
                print()

    torch.save(model.state_dict(), file)
    print("Training complete. Model saved to disk.")
    print()


def validateEncoder(model, loader, device, dtype=torch.float32):

    """Validation function"""
    criterion = nn.MSELoss()
    model = model.to(device=device)
    model.eval()
    loss_encoder = 0.0
    # loss_decoder = 0.0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(loader):
            tm2 = sample['images'][:, 0, :, :, :].to(device=device, dtype=dtype)
            tm1 = sample['images'][:, 1, :, :, :].to(device=device, dtype=dtype)
            t = sample['images'][:, 2, :, :, :].to(device=device, dtype=dtype)

            yax = sample['rewards']['agent_x'].narrow(1, 3, 27).to(device=device, dtype=dtype)
            yay = sample['rewards']['agent_y'].narrow(1, 3, 27).to(device=device, dtype=dtype)
            ytx = sample['rewards']['target_x'].narrow(1, 3, 27).to(device=device, dtype=dtype)
            yty = sample['rewards']['target_y'].narrow(1, 3, 27).to(device=device, dtype=dtype)

            out = model(tm2, tm1, t)
            # out, decoder_out = model(tm2, tm1, t)
            loss_ax = criterion(out[0], yax)
            loss_ay = criterion(out[1], yay)
            loss_tx = criterion(out[2], ytx)
            loss_ty = criterion(out[3], yty)
            loss_encoder += loss_ax + loss_ay + loss_tx + loss_ty

            # loss_tm2 = criterion(decoder_out[0], tm2)
            # loss_tm1 = criterion(decoder_out[1], tm1)
            # loss_t = criterion(decoder_out[2], t)
            # loss_decoder += loss_tm2 + loss_tm1 + loss_t

            count += 1

        print('Evaulation on validation dataset: Encoder got average loss: %.4f' % (loss_encoder / count))
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

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=4,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )

    # Generate a small encoder dataset for overfitting test
    train_loader, val_loader, test_loader = \
        loadEncoderData(spec, new=True, batch_size=4, train_num=20, val_num=4, test_num=4)