import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

device = torch.device('cpu')
dtype = torch.float32


class SineWaveDataset(Dataset):

    def __init__(self, data_in, data_out, train):
        """
        Inputs:
        - data_in: Input sine waves of shape (x, IN)
        - data_out: Output sine waves of shape (x, OUT)
        - train: Flag for train/validation vs. test set
        """
        if data_in.shape[0] != data_out.shape[0]:
            raise Exception('data dimension conflict!')
        self.data_in = data_in
        self.data_out = data_out
        self.train = train

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_in[idx], self.data_out[idx]


def buildData(x, delta, in_num, out_num, type, dir):
    """
    Generates random sine waves.

    Input:
    - x: Number of sine waves produced
    - delta: Linspace of the sine waves, number of separation = in_num, out_num
    - in_num: Number of input numbers in the interval defined by delta
    - out_num: Number of output numbers in the interval defined by delta
    - type: Data type (train/val/test)
    - dir: Directory to save data

    Returns a tuple of:
    - input: Input sine waves of shape (x, IN)
    - output: Output sine waves of shape (x, OUT)
    """
    # frequency w
    frequency = np.random.randn(x) + 1
    frequency[frequency == 0] = 1

    # amplitude A
    amplitude = (np.random.randn(x) + 1) * 2
    amplitude[amplitude == 0] = 1
    # phase Phi
    phase = np.random.randn(x) * np.pi

    # ndarray that stores these parameters
    params = np.zeros((x, 3))
    params[:, 0] = frequency
    params[:, 1] = amplitude
    params[:, 2] = phase

    input = np.zeros((x, in_num))
    output = np.zeros((x, out_num))

    linespace = np.linspace(delta[0], delta[1], delta[2])
    for i in range(x):
        data = np.sin(linespace * params[i, 0] + params[i, 2]) * params[i, 1]
        input[i, :] = data[:in_num]
        output[i, :] = data[in_num:]

    input = np.asarray(input,dtype=float)
    output = np.asarray(output,dtype=float)

    input_path, output_path = path(type, dir)

    np.save(input_path, input)
    np.save(output_path, output)
    print(type+" data saved to disk.")

    return input, output


def loadData(dir, batch_size=64):
    """
    Returns data loaders from disk.
    """
    train_in_path, train_out_path = path('train', dir)
    val_in_path, val_out_path = path('val', dir)
    test_in_path, test_out_path = path('test', dir)

    train_in = np.load(train_in_path)
    train_out = np.load(train_out_path)
    val_in = np.load(val_in_path)
    val_out = np.load(val_out_path)
    test_in = np.load(test_in_path)
    test_out = np.load(test_out_path)

    dset_train = SineWaveDataset(train_in, train_out, train=True)
    dset_val = SineWaveDataset(val_in, val_out, train=True)
    dset_test = SineWaveDataset(test_in, test_out, train=False)

    loader_train = DataLoader(dset_train, batch_size=batch_size)
    loader_val = DataLoader(dset_val, batch_size=batch_size)
    loader_test = DataLoader(dset_test, batch_size=batch_size)

    return loader_train, loader_val, loader_test


def path(type, dir):
    """
    Returns input/output dataset path.
    """
    in_path = os.path.join(dir,type+'_input.npy')
    out_path = os.path.join(dir,type+'_output.npy')
    return in_path, out_path


def visualize(input, gt, delta=None, output=None, index=None):
    """
    Plots a random sine wave from the given dataset.

    Inputs:
    - input: Input sine waves of shape (x, IN)
    - gt: Ground truth output of shape (x, OUT)
    - output: Predicted output of shape (x, OUT), default None
    - delta: Linspace of the sine waves, number of separation = IN + OUT
    - index: Array of indices of sine waves to show, or False to show all,
            or a number of random sine waves to show
    """
    if index is None:
        index = [0]
    if delta is None:
        delta = [0, 2 * np.pi, 50]
    x, IN = input.shape
    linspace = np.linspace(delta[0], delta[1], delta[2])
    indices = []
    if index == False:
        indices = list(range(x))
    elif type(index) == list:
        indices = index
    elif type(index) == int:
        if index > x:
            raise Exception('index out of bound')
        indices = np.random.randint(0,x,size=index)
    for i in indices:
        plt.plot(linspace[:IN], input[i])
        plt.plot(linspace[IN:], gt[i])
        if output != None:
            plt.plot(linspace[IN:], output[i])
        plt.xlabel('Angle (rad)')
        plt.ylabel('A*sin(wt+phi)')
        plt.axis('tight')
        plt.show()


def train(model, optimizer, loader_train, loader_val, file, epochs=10, print_every=100):
    """
    Training loop.

    Inputs:
    - model: torch.nn.module
    - optimizer: torch.optim
    - loader_train: Training data
    - loader_val: Validation data
    - file: Path to save trained model
    - epochs: Number of epochs to train
    - print_every: Accuracy logging frequency
    """
    print("Training starts.")
    print()
    model = model.to(device=device)
    criterion = nn.MSELoss()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                evaluate(loader_val, model)
                print()


    torch.save(model.state_dict(), file)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    print("Training complete. Model saved to disk.")
    print()


def evaluate(loader, model, delta=None):
    """
    Compute loss on validation/test set.

    Inputs:
    - loader: Data
    - model: torch.nn.module
    - delta: Linspace to aid visualization, None to stop drawing charts
    """
    if loader.dataset.train:
        print('Checking loss on Validation set')
    else:
        print('Checking loss on Test set')

    criterion = nn.MSELoss()
    loss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            out = model(x)

            loss += criterion(out, y)
            count = t + 1

        print('Got average loss: %.4f' % (loss / count))
        wandb.log({"Average loss": loss / count})

    # get a random batch from loader, pick 10 sine waves to draw
    if delta != None:
        print("Drawing charts.")
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=dtype)

                out = model(x)
                visualize(x, y, delta, out, index=10)
                break
        print("Charts complete.")


class SubMLP(nn.Module):

    def __init__(self, lstm_params):
        """
        Sub MLP module contained in SineWaveLSTM.

        Inputs:
        - lstm_params: Dict storing parameters of the network
        """
        super().__init__()

        self.num_layers = lstm_params['num_mst_layers']
        fc_layers = []
        for i in range(self.num_layers):

            input_dim = lstm_params['delta'][3] if i == 0 \
                else lstm_params['l' + str(i) + '_out_dim']
            output_dim = lstm_params['lstm_dim'] if i == self.num_layers-1 \
                else lstm_params['l' + str(i+1) + '_out_dim']

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


class SineWaveLSTM(nn.Module):

    def __init__(self, lstm_params):
        """
        Sine wave LSTM network.

        Inputs:
        - lstm_params: Dict storing parameters of the network
        """
        super().__init__()

        self.num_mst_layers = lstm_params['num_mst_layers']
        self.num_lstm_layers = lstm_params['num_lstm_layers']
        self.input_dim = lstm_params['delta'][3]
        self.sequence_num = lstm_params['delta'][4]
        self.lstm_dim = lstm_params['lstm_dim']
        self.hidden_dim = lstm_params['hidden_dim']

        self.mlp = SubMLP(lstm_params)
        self.lstm = nn.LSTM(self.lstm_dim, self.hidden_dim, batch_first=True, num_layers=self.num_lstm_layers)

        self.fc = nn.Linear(self.hidden_dim, 1)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        """
        Inputs:
        - x: Input of shape (batch_size, input_dim)

        Returns:
        - out: Output (1D) on each time step (batch_size, sequence_num)
        """
        embed = self.mlp(x)
        embed_stack = torch.stack([embed] * self.sequence_num, dim=1)
        lstm_out, _ = self.lstm(embed_stack)
        out = self.fc(lstm_out)
        out = torch.reshape(out,(-1,self.sequence_num))
        return out


def runLSTM(
    data_params,
    lstm_params,
    new_data=True,
    batch_size=64,
    load_model=False,
    training=True,
    learning_rate=4e-3,
    epochs=15,
    print_every=100,
    debug=False,
    eval='test'
):
    """
    Driver function to run LSTM network.

    Inputs:
    - data_params: Dict storing dataset parameters
    - lstm_params: Dict storing network parameters
    - new_data: Flag to build new data or use old ones
    - batch_size: Batch size for network
    - load_model: Flag to load trained parameters
    - training: Flag to train the network
    - learning_rate: Fixed learning rate
    - Epochs: Number of epochs to train
    - print_every: Accuray logging frequency
    - debug: Flag to turn on DEBUG mode
    - eval: Inference dataset
    """

    dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(dir, './data')
    model_path = os.path.join(dir, 'model.pt')

    train_num = data_params['train_num']
    val_num = data_params['val_num']
    test_num = data_params['test_num']
    delta = data_params['delta'][:3]
    in_num = data_params['delta'][3]
    out_num = data_params['delta'][4]

    if new_data:
        # Generate dataset
        buildData(train_num, delta, in_num, out_num, 'train', dataset_dir)
        buildData(val_num, delta, in_num, out_num, 'val', dataset_dir)
        buildData(test_num, delta, in_num, out_num, 'test', dataset_dir)

    # load data
    loader_train, loader_val, loader_test = loadData(dataset_dir, batch_size=batch_size)

    # initialize model
    model = SineWaveLSTM(lstm_params)

    if load_model:
        # load pre-trained parameters
        model.load_state_dict(torch.load(model_path))

    wandb.watch(model, log="all")

    if training:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train(model, optimizer, loader_train, loader_val, model_path, epochs=epochs, print_every=print_every)

    if debug:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if eval == 'test':
        # check on test data
        evaluate(loader_test, model, delta=delta)
    elif eval == 'overfit':
        # check on train data
        evaluate(loader_train, model, delta=delta)
    elif eval == 'all':
        # check accuracy on all sets
        evaluate(loader_train, model)
        evaluate(loader_val, model)
        evaluate(loader_test, model)


if __name__ == "__main__":

    wandb.init(project="lstm-sine")

    data_params = {}
    data_params['train_num'] = 5000
    data_params['val_num'] = 500
    data_params['test_num'] = 500
    data_params['delta'] = [0,2*np.pi,50,10,40]

    lstm_params = {}
    lstm_params['delta'] = [0,2*np.pi,50,10,40]
    lstm_params['num_mst_layers'] = 3
    lstm_params['num_lstm_layers'] = 2
    lstm_params['l1_out_dim'] = 256
    lstm_params['l2_out_dim'] = 128
    lstm_params['lstm_dim'] = 16
    lstm_params['hidden_dim'] = 512

    runLSTM(data_params=data_params,
            lstm_params=lstm_params,
            batch_size=64,
            print_every=50,
            learning_rate=1e-3,
            new_data=False,
            load_model=True,
            training=True,
            epochs=10,
            eval='test')


# Potential improvements:
# 1. plot train vs. val loss over epochs
# 2. add learning_rate decay mechanism