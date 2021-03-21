import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import torchvision.datasets as datasets
from torchvision import transforms
from torchsummary import summary

import argparse


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


device = torch.device("cpu")


class VAE(nn.Module):
    def __init__(self, num_latent_dimensions, num_channels):
        super().__init__()

        self.num_latent_dimensions = num_latent_dimensions

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = num_channels

        ## Construct the encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.latent_mean = nn.Linear(in_features=hidden_dims[-1],
                                     out_features=num_latent_dimensions)
        self.latent_sigma = nn.Linear(in_features=hidden_dims[-1],
                                      out_features=num_latent_dimensions)

        ## Construct the decoder
        modules = []
        self.decoder_input = nn.Linear(in_features=num_latent_dimensions,
                                       out_features=hidden_dims[-1])
        hidden_dims.reverse()

        kernel_sizes = [3, 3, 2, 2, 2]
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                   out_channels=hidden_dims[i+1],
                                   kernel_size=kernel_sizes[i],
                                   stride=2),
                                   #padding=1,
                                   #output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))

        self.decoder = nn.Sequential(*modules)

        ## Add a final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1],
                      out_channels=1,
                      kernel_size=3,
                      padding=1),
            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)
        result = result.view(result.size(0), -1)
        mean = self.latent_mean(result)
        log_var = self.latent_sigma(result)
        return [mean, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = result.view(result.size(0), 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, input):
        [mean, log_var] = self.encode(input)
        z = self.reparameterize(mean, log_var)
        #print("Latent dimension" + str(z.size()))
        result = self.decode(z)
        #print(result.size())
        return result, mean, log_var


def vae_loss(predictions, labels, mean, log_var):
    reconstruction_loss = F.mse_loss(predictions, labels)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim = 1), dim = 0)

    return reconstruction_loss + kl_loss


def train(model, epoch, data_loader):
    # Set the model to train mode
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0
    for num_batch, (batch, _) in enumerate(data_loader):
        # Reset the optimizer
        optimizer.zero_grad()

        predicted, mean, log_var = model(batch)
        loss = vae_loss(predicted, batch, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print("Epoch {}, Loss: {}".format(epoch, train_loss))


if __name__ == '__main__':
    # Create model
    vae = VAE(2, 1)
    print(vae)

    # Load data
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    print(len(mnist_trainset))

    mnist_train_data_loader = utils.data.DataLoader(mnist_trainset, batch_size=1000,  shuffle=True)
    mnist_test_data_loader = utils.data.DataLoader(mnist_testset, batch_size=1000, shuffle=True)

    print(type(mnist_train_data_loader))

    for epoch in range(args.epochs):
        train(vae, epoch, mnist_train_data_loader)
