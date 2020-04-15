import torch
from config import knobs


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, 4, 2, 1),  # 28x28 -> 14x14
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1),  # 14x14 -> 7x7
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(256, 512, 4, 2, 5),  # 7x7 -> 7x7
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(512, 1024, 4, 2, 5),  # 7x7 -> 7x7
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            Lambda(lambda x: x.view(-1, 7 * 7 * 1024)),
            torch.nn.Linear(7 * 7 * 1024, knobs["hidden_dim"]),
        )

    def forward(self, x):
        return self.main(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(knobs["hidden_dim"], 7 * 7 * 1024),
            torch.nn.ReLU(inplace=True),
            Lambda(lambda x: x.view(-1, 1024, 7, 7)),
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 7x7 -> 14x14
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14x14 -> 28x28
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 1, 4, 2, 15),  # 28x28 -> 28x28
            torch.nn.Tanh(),
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


class FastEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 4, 2, 1),  # 28x28 -> 14x14
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, 4, 2, 5),  # 7x7 -> 7x7
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            Lambda(lambda x: x.view(-1, 7 * 7 * 128)),
            torch.nn.Linear(7 * 7 * 128, knobs["hidden_dim"]),

            # torch.nn.Conv2d(128, 256, 4, 2, 5),  # 7x7 -> 7x7
            # torch.nn.BatchNorm2d(256),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            # Lambda(lambda x: x.view(-1, 7 * 7 * 256)),
            # torch.nn.Linear(7 * 7 * 256, knobs["hidden_dim"]),
        )

    def forward(self, x):
        return self.main(x)


class FastDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(knobs["hidden_dim"], 7 * 7 * 128),
            torch.nn.ReLU(inplace=True),
            Lambda(lambda x: x.view(-1, 128, 7, 7)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7x7 -> 14x14
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14x14 -> 28x28
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 1, 4, 2, 15),  # 28x28 -> 28x28
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class FastDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            FastEncoder(),
            Lambda(lambda x: torch.mean(x, dim=1))
        )

    def forward(self, x):
        return self.main(x)

