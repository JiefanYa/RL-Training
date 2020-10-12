from sprites_datagen.moving_sprites import EncoderDataset, DecoderDataset, ImplDataset
from sprites_datagen.rewards import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from general_utils import AttrDict
import wandb
from datetime import datetime


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
                batch_size=32,
                save_to_disk=False,
                decoder=False,
                new=False,
                path=None,
                train_num=5000,
                val_num=1000,
                test_num=1000):

    """Return dataloaders for train/val/test datasets"""
    global train_dataset, val_dataset, test_dataset
    if save_to_disk:
        # for test/inspection: save dataset to disk, decoder dataset only has one object
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
            print('Error when loading dataset: ' + str(err))
    else:
        train_dataset = ImplDataset(spec, mode='test')
        val_dataset = ImplDataset(spec, mode='val')
        test_dataset = ImplDataset(spec, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def trainVAERewardPrediction(model,
                             rewards,
                             train_decoder,
                             optimizer,
                             loader_train,
                             loader_val,
                             model_path,
                             device,
                             dtype,
                             epochs=100,
                             print_every=15,
                             save_every=30,
                             validate_every=10):
    """Training function"""
    print("Training VAERewardPrediction model starts.\n")
    model = model.to(device=device)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for i, sample in enumerate(loader_train):
            model.train()

            ts = [sample['images'][:, index, 0, :, :].unsqueeze(1).to(device=device, dtype=dtype) for index in
                  range(10)]
            ys = [sample['rewards'][reward].narrow(1, 10, 20).to(device=device, dtype=dtype) for reward in rewards]

            if train_decoder:
                out, images = model(ts)
                loss_decoder = [criterion(images[index], ts[index]) for index in range(len(images))]
                loss_decoder = sum(loss_decoder)
            else:
                out = model(ts)
                loss_decoder = 0

            loss_encoder = [criterion(out[index], ys[index]) for index in range(len(rewards))]
            loss_encoder = sum(loss_encoder)
            loss = loss_encoder + loss_decoder

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print('Training set: Epoch %d, Iteration %d, model loss = %.4f, encoder loss = %.4f, decoder loss = %.4f\n'
                      % (e, i, loss.item(), loss_encoder.item(), loss_decoder.item() if type(loss_decoder) != int else 0))
                wandb.log({'Training encoder loss': loss_encoder.item(),
                           'Training decoder loss': loss_decoder.item() if type(loss_decoder) != int else 0})

        if e % validate_every == 0:
            validateVAERewardPrediction(model, rewards, train_decoder, loader_val, device, dtype)

        if e % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print('Model saved to disk on Epoch %d at %s\n' % (e, datetime.now()))

    torch.save(model.state_dict(), model_path)
    print("Training complete. VAERewardPrediction model saved to disk.\n")


def validateVAERewardPrediction(model, rewards, train_decoder, loader, device, dtype):
    """Validation function"""
    print('Running validation on VAERewardPrediction model...\n')
    criterion = nn.MSELoss()
    model = model.to(device=device)
    model.eval()
    loss_encoder_total = 0.0
    loss_decoder_total = 0.0
    count = 0
    log_image = False

    with torch.no_grad():
        for i, sample in enumerate(loader):

            ts = [sample['images'][:, index, 0, :, :].unsqueeze(1).to(device=device, dtype=dtype) for index in range(10)]
            ys = [sample['rewards'][reward].narrow(1, 10, 20).to(device=device, dtype=dtype) for reward in rewards]

            if train_decoder:
                out, images = model(ts)
                loss_decoder = [criterion(images[index], ts[index]) for index in range(len(images))]
                if not log_image:
                    index = np.random.randint(ts[0].size(0)) # TODO: log one image to wandb
                    originals = [t[index,:,:,:].cpu().numpy() for t in ts]
                    originals = [wandb.Image(np.moveaxis(x, 0, 2), caption='Ground Truth') for x in originals]
                    predictions = [image[index,:,:,:].cpu().numpy() for image in images]
                    predictions = [wandb.Image(np.moveaxis(x,0,2), caption='Prediction') for x in predictions]
                    log_image = True
            else:
                out = model(ts)
                loss_decoder = [0]

            loss_encoder = [criterion(out[index], ys[index]) for index in range(len(rewards))]
            loss_encoder_total += sum(loss_encoder)
            loss_decoder_total += sum(loss_decoder)
            count += 1

    print('Validation set: average encoder loss: %.4f, average decoder loss: %.4f\n'
          % (loss_encoder_total / count, loss_decoder_total / count))
    if train_decoder:
        wandb.log({'Encoder loss': loss_encoder_total / count,
                   'Decoder loss': loss_decoder_total / count,
                   'Ground Truths': originals,
                   'Predictions wrt rewards': predictions})
    else:
        wandb.log({'Encoder loss': loss_encoder_total / count,
                   'Decoder loss': loss_decoder_total / count})


def trainVAEReconstruction(model,
                           optimizer,
                           loader_train,
                           loader_val,
                           model_path,
                           device,
                           dtype,
                           epochs=100,
                           print_every=15,
                           save_every=30,
                           validate_every=10):
    """Training function"""
    print("Training VAEReconstruction model starts.\n")
    print()
    model = model.to(device=device)
    criterion = nn.MSELoss()

    for e in range(epochs):
        for i, sample in enumerate(loader_train):
            model.train()

            ts = [sample['images'][:, index, 0, :, :].unsqueeze(1).to(device=device, dtype=dtype) for index in
                  range(10)]

            out = model(ts)
            losses = [criterion(out[index], ts[index]) for index in range(len(ts))]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print('Training set: Epoch %d, Iteration %d, model loss = %.4f\n' % (e, i, loss.item()))
                wandb.log({'Training reconstruction loss': loss.item()})

        if e % validate_every == 0:
            validateVAEReconstruction(model, loader_val, device, dtype)

        if e % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print('Model saved to disk on Epoch %d at %s\n' % (e, datetime.now()))

    torch.save(model.state_dict(), model_path)
    print('Training complete. VAEReconstruction model saved to disk.\n')


def validateVAEReconstruction(model, loader_val, device, dtype):
    print('Running validation on VAEReconstruction model...\n')
    criterion = nn.MSELoss()
    model = model.to(device=device)
    model.eval()
    count = 0
    loss_total = 0

    with torch.no_grad():

        for i, sample in enumerate(loader_val):
            ts = [sample['images'][:, index, 0, :, :].unsqueeze(1).to(device=device, dtype=dtype) for index in
                  range(10)]

            out = model(ts)
            losses = [criterion(out[index], ts[index]) for index in range(len(ts))]
            loss_total += sum(losses)
            count += 1

        print('Validation set: average model loss: %.4f\n' % (loss_total / count))
        wandb.log({'Reconstruction loss': loss_total / count})


'''RL stuff here'''
def unpack_paths_to_rollout_components(paths):
    obs = np.concatenate([path['obs'] for path in paths])
    acs = np.concatenate([path['acs'] for path in paths])
    next_obs = np.concatenate([path['next_obs'] for path in paths])
    terminals = np.concatenate([path['terminals'] for path in paths])
    rews = np.concatenate([path['rews'] for path in paths])
    return obs, acs, next_obs, terminals, rews


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length=None):
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


def sample_trajectory(env, policy, max_path_length=None):
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
        if done or (max_path_length is not None and steps > max_path_length):
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


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


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
