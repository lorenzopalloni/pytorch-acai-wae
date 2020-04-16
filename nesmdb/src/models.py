import torch
from config import knobs


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FastEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 8, 4, 2),  # 17640 -> 4410
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(32, 64, 8, 4, 2),  # 4410 -> 1102
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(64, 128, 8, 4, 2),  # 1102 -> 275
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            Lambda(lambda x: x.view(-1, 275 * 128)),
            torch.nn.Linear(275 * 128, knobs['hidden_dim']),
        )

    def forward(self, x):
        return self.main(x)


class FastDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(knobs['hidden_dim'], 275 * 128),
            torch.nn.ReLU(inplace=True),
            Lambda(lambda x: x.view(-1, 128, 275)),
            torch.nn.ConvTranspose1d(128, 64, 8, 4, 1),  # 275 -> 1102
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(64, 32, 8, 4, 1),  # 1102 -> 4410
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(32, 1, 8, 4, 2),  # 4410 -> 17640
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class FastDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            Encoder(),
            Lambda(lambda x: torch.mean(x, dim=1))
        )

    def forward(self, x):
        return self.main(x)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 4, 2, 2),  # 17640 -> 8821
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(32, 32, 4, 2, 2),  # 8821 -> 4411
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(32, 64, 4, 2, 2),  # 4410 -> 2206
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(64, 64, 4, 2, 2),  # 2206-> 1104
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(64, 128, 4, 2, 2),  # 1104 -> 553
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(128, 128, 4, 2, 2),  # 553 -> 277
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv1d(128, 256, 4, 2, 2),  # 277 -> 139
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            Lambda(lambda x: x.view(-1, 256 * 139)),
            torch.nn.Linear(256 * 139, knobs['hidden_dim']),
        )

    def forward(self, x):
        return self.main(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(knobs['hidden_dim'], 256 * 139),
            Lambda(lambda x: x.view(-1, 256, 139)),
            torch.nn.ConvTranspose1d(256, 128, 4, 2, 2),  # 139 -> 276
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(128, 128, 4, 2, 1),  # 276 -> 552
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(128, 64, 4, 2, 1),  # 552 -> 1104
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(64, 64, 4, 2, 2),  # 1104 -> 2206
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(64, 32, 4, 2, 2),  # 2206 -> 4410
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(32, 32, 4, 2, 1),  # 4410 -> 8818
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose1d(32, 1, 4, 2, 1),  # 8820 -> 17640
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            Encoder(),
            Lambda(lambda x: torch.mean(x, dim=1))
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    a = torch.ones(5, 1, 17640)
    b = torch.ones(5, 8)
    encoder = Encoder()
    hola = encoder(a)
    loss = torch.mean(b - hola)
    loss.backward()
    iterator_parameters = encoder.parameters()
    length = 0
    for num, (k, v) in enumerate(encoder.state_dict().items()):
        if k.find('bias') != -1 or k.find('weight') != -1:
            print(k, v.shape, next(iterator_parameters).shape)

