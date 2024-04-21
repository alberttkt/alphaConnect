from torch import nn, optim, utils
from torch import flatten, tanh, zeros
import lightning as L
from torch.optim.lr_scheduler import StepLR


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=3, stride=1):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.n(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.n = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels),
        )
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(x + self.n(x))


class PolicyHead(nn.Module):
    def __init__(self, in_channels=256, num_moves=7):
        super().__init__()
        self.conv = ConvBlock(in_channels, 2, kernel_size=1, stride=1)
        self.fc = nn.Linear(84, num_moves)

    def forward(self, x):
        x = self.conv(x)
        x = flatten(x, 1)  # Flattening except for the batch dimension
        return self.fc(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = ConvBlock(in_channels, 1, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(
            42, 256
        )  # Assuming global average pooling is done before this layer
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = flatten(x, 1)  # Flattening except for the batch dimension
        x = self.relu(self.fc1(x))
        x = tanh(self.fc2(x))
        return x


class AlphaZeroModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(3, 256),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )

        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv_layers(x)
        # flat = x.view(x.size(0), -1)
        # print(flat.shape)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def training_step(self, batch, batch_idx):
        x, y_policy, y_value = batch
        policy_pred, value_pred = self(x)
        # print(y_hat.shape, y.shape)
        policy_loss = nn.functional.cross_entropy(policy_pred, y_policy)

        # Calculate value loss (MSE)
        value_loss = nn.functional.mse_loss(
            value_pred.view(-1), y_value.float()
        )  # Ensure y_value is float for MSE

        # Combine losses
        total_loss = (
            policy_loss + 10 * value_loss
        )  # You might want to add a weighting factor here

        # Log losses
        self.log("policy_loss", policy_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("value_loss", value_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]


class MyModel(L.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.net = network
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)


class MyDataset(utils.data.Dataset):
    def __init__(self, features, policies, values):
        self.features = features
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # the tensor is flattened, put it back into the shape (6,7)

        input_tensor = self.features[index].view(1, 6, 7)

        output_tensor = zeros((3, 6, 7))
        output_tensor[0] = (input_tensor == -1).float()

        # Fill the second channel with 1s for squares belonging to the current player (0 in the input tensor)
        output_tensor[1] = (input_tensor == 0).float()

        # Fill the third channel with 1s for squares belonging to the other player (1 in the input tensor)
        output_tensor[2] = (input_tensor == 1).float()

        return output_tensor, self.policies[index], self.values[index]
