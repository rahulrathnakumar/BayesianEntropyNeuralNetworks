import torch.nn as nn
import torch.nn.functional as F
import torch

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(CVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 24, kernel_size=6, stride=1, padding=0)
        self.conv2 = nn.Conv2d(24, 40, kernel_size=9, stride=1, padding=0)
        self.conv3 = nn.Conv2d(40, 288, kernel_size=9, stride=1, padding=0)

        self.fc1 = nn.Linear(18 * 18 * 288, 200)  # Adjust based on your specific architecture
        self.fc2_mean = nn.Linear(200, latent_size)
        self.fc2_logvar = nn.Linear(200, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, 200)
        self.fc4 = nn.Linear(200, 18 * 18 * 288)  # Adjust based on your specific architecture

        self.deconv1 = nn.ConvTranspose2d(288, 40, kernel_size=9, stride=1, padding=0, dilation = 6)
        self.deconv2 = nn.ConvTranspose2d(40, 24, kernel_size=9, stride=1, padding=0, dilation = 8)
        self.deconv3 = nn.ConvTranspose2d(24, 1, kernel_size=6, stride=1, padding=0, dilation = 14)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2_mean(x), self.fc2_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(z.size(0), 288, 18, 18)  # Adjust based on your specific architecture
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        return torch.sigmoid(self.deconv3(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instantiate the CVAE
model = CVAE(input_size=(1, 200, 200), latent_size=30)
# model = model.to(DEVICE)
# visualize CVAE architecture
# from torchsummary import summary
# summary(cvae, input_size=(1, 200, 200))

if __name__ == '__main__':
    # random image for testing - 200x200
    x = torch.randn(1, 1, 200, 200)
    # forward pass
    out, mu, logvar = model(x)
    # check output size
    print(out.size())

