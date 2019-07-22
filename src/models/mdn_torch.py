import numpy as np # basic math and random numbers
import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
import src.utils.pdf as pdf
from torch.autograd import Variable # storing data while learning
from src.utils.helper import gaussian_distribution


class MixtureDensityNetwork(nn.Module):
    def __init__(self, hidden_layers, n_gaussians, n_inputs=1, n_outputs=1):
        super(MixtureDensityNetwork, self).__init__()
        last_layer_size = None
        self.network = []
        self.n_outputs = n_outputs
        self.n_gaussians = n_gaussians
        for ix, layer_size in enumerate(hidden_layers):

            if ix == 0:
                self.network.append(nn.Linear(n_inputs, layer_size))
            else:
                self.network.append(nn.Linear(last_layer_size, layer_size))

            self.network.append(nn.Tanh())

            last_layer_size = layer_size

        self.z_pi = nn.Linear(last_layer_size, n_gaussians)
        self.z_sigma = nn.Linear(last_layer_size, n_gaussians * n_outputs)
        self.z_mu = nn.Linear(last_layer_size, n_gaussians * n_outputs)

    def forward(self, x):
        for ix, layer in enumerate(self.network):

            if ix == 0:
                z_h = layer(x)
            else:
                z_h = layer(z_h)

        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

    def generate_data(self, n_samples=1000):
        # evenly spaced samples from -10 to 10

        x_test_data = np.linspace(-15, 15, n_samples)
        x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, 1))
        x_test_variable = Variable(x_test_tensor)

        pi_variable, sigma_variable, mu_variable = self(x_test_variable)

        pi_data = pi_variable.data.numpy()
        sigma_data = sigma_variable.data.numpy()
        mu_data = mu_variable.data.numpy()

        return x_test_data, pi_data, mu_data, sigma_data

    def mdn_loss_fn(self, pi, sigma, mu, y, epsilon=1e-9):

        result = gaussian_distribution(y, mu, sigma)
        first_idx = 0
        all_gaussians = []
        for pi_idx in range(self.n_outputs):
            result_per_gaussian = result[:, first_idx:first_idx + self.n_gaussians]
            all_gaussians.append(torch.sum((result_per_gaussian * pi), dim=1))
            first_idx += self.n_gaussians
        result = torch.stack(all_gaussians, 1)
        result = torch.where(result == 0, torch.tensor(epsilon), result)
        result = -torch.log(result)
        return torch.mean(result)

    def fit(self, x, y, epochs=10000, batch_size=100, verbose=True):

        optimizer = torch.optim.Adam(self.parameters())

        x_tensor = torch.from_numpy(np.float32(x))
        y_tensor = torch.from_numpy(np.float32(y))

        x_variable = Variable(x_tensor)
        y_variable = Variable(y_tensor, requires_grad=False)

        batch_size = len(x) if batch_size is None else batch_size

        print("Training mdn network on {} datapoints".format(len(x)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        for epoch in range(epochs):
            x_batch, y_batch = next(batch_gen_iter)
            pi_variable, sigma_variable, mu_variable = self(x_batch)
            loss = self.mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == 0:
                print("Initial Loss is: {}".format(loss.item()))

            elif epoch % 100 == 0 and verbose:
                if epoch != 0:
                    print(epoch, loss.item())

        print("Training Finished, final loss is {}".format(loss.item()))

    def get_mog(self, x):
        """
        Return the conditional mog at location x.
        :param network: an MDN network
        :param x: single input location
        :return: conditional mog at x
        """
        # gather mog parameters
        pi, sigma, mean = self(x)
        pi = pi.data.numpy().transpose()
        mean = mean.data.numpy().transpose()
        sigma = sigma.data.numpy().transpose()

        # return mog
        return pdf.MoG(a=pi, ms=mean, Ss=sigma)







