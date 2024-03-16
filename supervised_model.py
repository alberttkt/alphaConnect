import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torch.utils as utils




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
        self.log('train_loss', loss,on_epoch=True, on_step=False)
        return loss
    

    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)
    
class MyDataset(utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

