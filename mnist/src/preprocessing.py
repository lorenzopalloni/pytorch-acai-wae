import torch
import torchvision

from config import data_dir, knobs


def get_data():
    train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    valid = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    return torch.cat((train.data.float(), valid.data.float()))


def standardize(x: torch.tensor) -> torch.tensor:
    whole_data_mean = 33.38596343994140625
    whole_data_std = 78.6543731689453125
    whole_data_extreme_value = 2.8175680637359619140625
    x = x - whole_data_mean
    x = x / whole_data_std
    x = x / whole_data_extreme_value
    return x


def inv_standardize(x: torch.tensor) -> torch.tensor:
    whole_data_mean = 33.38596343994140625
    whole_data_std = 78.6543731689453125
    whole_data_extreme_value = 2.8175680637359619140625
    x = x * whole_data_extreme_value
    x = x * whole_data_std
    x = x + whole_data_mean
    return x


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def preprocess(x):
    return x.view(-1, 1, 28, 28).to(knobs['device'])


def get_loader():
    whole_data = get_data()
    whole_data = standardize(whole_data)
    ds = torch.utils.data.TensorDataset(whole_data)
    dl = torch.utils.data.DataLoader(ds, knobs['batch_size'], shuffle=True)
    return WrappedDataLoader(dl, preprocess)


if __name__ == '__main__':
    data = get_data()
    torch.set_printoptions(precision=30)
    mean = data.mean()
    std = data.std()
    print('\nWhole Dataset (Train + Valid)\nmean:\t', mean, '\nstd:\t', std)

    def describe(x):
        import pandas as pd
        print(pd.Series(x.flatten().numpy()).describe())

    an_example = data[0]
    describe(an_example)
    describe(standardize(an_example))
    describe(inv_standardize(standardize(an_example)))

    loader = get_loader()
    example = next(iter(loader))[0].to('cpu')
    describe(example)
