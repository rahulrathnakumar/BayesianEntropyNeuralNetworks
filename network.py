import torch.nn as nn
import torch

class CVAE(nn.Module):
    def __init__(self, num_constraints, latent_size=256):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.num_constraints = num_constraints
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(128),
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_size)
        
        self.fc_z = nn.Linear(latent_size, 128 * 4 * 4)

        # Predict eta using an MLP unrelated to the latent space
        self.mlp = nn.Sequential(
            nn.Linear(num_constraints, 32),
            nn.ReLU(),
            nn.Linear(32,num_constraints),
            nn.Sigmoid()
        )
        
        
        
        # Decoder - Image reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=4),
            nn.Sigmoid()
        )
        
        

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # noise for self.mlp input
        eta_ = torch.randn(z.size(0), self.num_constraints)
        eta_ = self.mlp()
        z = self.fc_z(z)
        z = z.view(z.size(0), 128, 4, 4)
        z = self.decoder(z)
        # resize the decoder output to match the input size
        z = nn.functional.interpolate(z, size=(256, 256), mode='bilinear', align_corners=False)
        return z, eta_

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, eta_ = self.decode(z)
        return recon_x, mu, logvar, eta_

class MLP(nn.Module):
    def __init__(self, num_constraints):
        super(MLP, self).__init__()
        self.num_constraints = num_constraints
        self.mlp = nn.Sequential(
            nn.Linear(num_constraints, 32),
            nn.ReLU(),
            nn.Linear(32,num_constraints),
            nn.Sigmoid()
        )
    def forward(self, eta):
        eta_ = self.mlp(eta)
        return eta_


class BaseCVAE(nn.Module):
    def __init__(self, latent_size=256):
        super(BaseCVAE, self).__init__()
        self.latent_size = latent_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(128),
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_size)
        
        self.fc_z = nn.Linear(latent_size, 128 * 4 * 4)
    
        
        # Decoder - Image reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=4),
            nn.Sigmoid()
        )
        
        

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(z.size(0), 128, 4, 4)
        z = self.decoder(z)
        # resize the decoder output to match the input size
        z = nn.functional.interpolate(z, size=(256, 256), mode='bilinear', align_corners=False)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



if __name__ =='__main__':
    # Create a VAE instance
    vae = CVAE(latent_size=256)

    # Generate random input data
    batch_size = 1
    input_data = torch.randn(batch_size, 1, 256, 256)  # Assuming input size is 64x64

    # Forward pass through the VAE
    recon_x, mu, logvar = vae(input_data)

    # Plot the original and reconstructed images
    original_img = input_data[0].squeeze().detach().numpy()
    reconstructed_img = recon_x[0].squeeze().detach().numpy()
    
    print(original_img.shape)
    print(reconstructed_img.shape)

