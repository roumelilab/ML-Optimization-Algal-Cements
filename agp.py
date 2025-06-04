import torch
from torch import nn
import gpytorch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import *
import warnings

# create the class for the sample-level models
class ExactAGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None, kernel=None, ls_dim=1):
        super(ExactAGP, self).__init__(train_x, train_y, likelihood)
        if mean is None:
            warnings.warn("Valid mean function is not specificed")
        if mean == 'sigmoid':
            self.mean_module = SigmoidMean(train_x.shape[-1])
        if mean == 'linear':
            self.mean_module = LinearMean(train_x.shape[-1])
        if mean == 'exponential':
            self.mean_module = ExponentialMean(train_x.shape[-1])

        if kernel is None:
            warnings.warn("Valid kernel function is not specified")
        if kernel == 'matern':
            self.covar_module = MaternKern(ls_dim)
        if kernel == 'rbf':
            self.covar_module = RBFKern(ls_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFKern(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, input_size):
        super(RBFKern, self).__init__()

        self.ls = torch.exp(torch.randn(input_size))
        self.scale = torch.exp(torch.randn(1))
        if input_size > 1:
            self.ARD = True
        else:
            self.ARD = False

    def forward(self, x1, x2, **params):
        if self.ARD:
            r = torch.zeros((x1.shape[0], x2.shape[0]))
            for d in range(x1.shape[-1]):
                r += (x1[:, d] - x2[:, d].T) / (2*self.ls[d]**2)
        elif x1.shape[-1] > 1:
            r = torch.norm(x1[:, None, :] - x2[None, :, :], p=2, dim=-1)**2 / (2 * self.ls**2)
        else:
            r = (x1 - x2.T)**2 / (2 * self.ls**2)

        return self.scale * torch.exp(-r)

    def _set_ls(self, ls):
        self.ls = ls

    def _set_scale(self, s):
        self.scale = s


class MaternKern(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, input_size):
        super(MaternKern, self).__init__()

        self.ls = torch.exp(torch.randn(input_size))
        self.scale = torch.exp(torch.randn(1))
        if input_size > 1:
            self.ARD = True
        else:
            self.ARD = False

    def forward(self, x1, x2, **params):
        if self.ARD:
            r = torch.zeros((x1.shape[0], x2.shape[0]))
            for d in range(x1.shape[-1]):
                r += torch.abs(x1[:, d] - x2[:, d].T) / (self.ls[d])
        elif x1.shape[-1] > 1:
            r = torch.norm(x1[:, None, :] - x2[None, :, :], p=2, dim=-1) / self.ls
        else:
            r = torch.abs(x1 - x2.T) / self.ls

        return self.scale * torch.exp(-r)

    def _set_ls(self, ls):
        self.ls = ls

    def _set_scale(self, s):
        self.scale = s


class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super(LinearMean, self).__init__()
        self.weights = torch.randn(*batch_shape, input_size, 1)
        self.bias = torch.randn(*batch_shape, 1)

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def _set_bias(self, b):
        self.bias = b

    def _set_weights(self, w):
        self.weights = w.unsqueeze(-1)


class SigmoidMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super(SigmoidMean, self).__init__()
        self.weights = torch.randn(*batch_shape, input_size, 1)
        self.bias = torch.randn(*batch_shape, 1)

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return torch.sigmoid(res)

    def _set_bias(self, b):
        self.bias = b

    def _set_weights(self, w):
        self.weights = w.unsqueeze(-1)

class ExponentialMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super(ExponentialMean, self).__init__()
        self.weights = torch.randn(*batch_shape, input_size, 1)
        self.bias = torch.randn(*batch_shape, 1)

    def forward(self, x):
        res = torch.divide(x, self.weights).squeeze(-1)
        return self.bias*(1 - torch.exp(-res))

    def _set_bias(self, b):
        self.bias = b

    def _set_weights(self, w):
        self.weights = w.unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=1, n_layers=1, sp=True):
        super(MLP, self).__init__()
        # MLP
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc_layers = nn.Sequential(*layers)
        if sp:
            self.fc_layers.add_module("softplus", nn.Softplus())

    def forward(self, x):
        x = self.fc_layers(x)
        return x


def dataset_with_indices(cls):
    """
    Modifies the given dataset class to return a tuple data, target, index instead of just data, target
    From: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
    :param cls:
    :return:
    """

    def __getitem__(self, index):
        data, target, meta_data = cls.__getitem__(self, index)
        return data, target, meta_data, index

    return type(cls.__name__, (cls,), { '__getitem__': __getitem__,})


class AGP():
    def __init__(self, train_x, train_y, train_z, sigma=0.1, mean=None, kernel=None, ARD=False, n_layers=1,
                 hidden_dim=10, ls_net=None, sc_net=None, w_net=None, b_net=None, n_net=None, seed=None):
        """
        :param train_x: input data for the GP, number of samples x number of time points x number of time-varying measures
        :param train_y: output data for the GP, number of samples x number of time points
        :param train_z: static conditioning set, number of samples x number of static measures
        :param sigma: likelihood variance for y | f(x)
        """
        torch.set_default_tensor_type(torch.DoubleTensor) #use double precision as default
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(0)
        self.seed = seed


        self.N = train_x.shape[0]
        self.D = train_x.shape[-1]
        self.M = train_z.shape[1]

        #store information about set-up
        self.h = hidden_dim
        self.n_layers = n_layers
        self.mean = mean
        self.ARD = ARD
        self.kernel = kernel

        device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if ARD:
            self.ls_dim = self.D
        else:
            self.ls_dim = 1

        self.lengthscale_net = ls_net.to(device) if ls_net else MLP(self.M, output_dim=self.ls_dim, n_layers=n_layers,
                                                                    hidden_dim=hidden_dim).to(device)
        self.scale_net = sc_net.to(device) if sc_net else MLP(self.M, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
        #note that the below assumes we are in a setting where a positive constraint on the weight vector is needed
        self.weight_net = w_net.to(device) if w_net else MLP(self.M, output_dim=self.D, sp=True, n_layers=n_layers,
                                                             hidden_dim=hidden_dim).to(device)
        self.bias_net = b_net.to(device) if b_net else MLP(self.M, sp=True, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
        self.noise_net_1 = n_net.to(device) if n_net else MLP(self.M, sp=True, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
        self.noise_net_2 = n_net.to(device) if n_net else MLP(self.M, sp=True, n_layers=n_layers,
                                                              hidden_dim=hidden_dim).to(device)


    def fit(self, train_x, train_y, train_z, niter=100, print_iter=25, sigma=torch.tensor(0.1), lr=0.1, batch_size=100,
            checkpoint=25, save_dir='./'):


        params = list(self.lengthscale_net.parameters()) + list(self.scale_net.parameters()) + \
                 list(self.weight_net.parameters()) + list(self.bias_net.parameters()) + \
                 list(self.noise_net_1.parameters()) + list(self.noise_net_2.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)

        store_loss = []

        # set up data loader
        dataset = dataset_with_indices(TensorDataset)(train_x, train_y, train_z)
        data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(niter):

            for (x, y, z, idx) in data:
                optimizer.zero_grad()
                # update the lengthscales
                ls = self.lengthscale_net(z)
                sc = self.scale_net(z)
                m = self.weight_net(z)
                b = self.bias_net(z)
                n1 = self.noise_net_1(z)
                n2 = self.noise_net_2(z)

                data_idx = ~torch.isnan(x[:, :, 0])

                calc_noise = [ (n1[j] * (1 - torch.exp(-x[j, data_idx[j]]/n2[j]))).squeeze()  + 1e-4 for j in range(x.shape[0])]

                likelihood_list = [gpytorch.likelihoods.FixedNoiseGaussianLikelihood(calc_noise[j],
                                                                                    learn_additional_noise=False)
                                   for j in range(x.shape[0])]

                model_list = [ExactAGP(x[j, data_idx[j], :], y[j, data_idx[j]],
                                        likelihood_list[j], self.mean, self.kernel, self.ls_dim) for j in range(x.shape[0])]

                for model in model_list:
                    model.train()
                for likelihood in likelihood_list:
                    likelihood.train()

                mll_list = [gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_list[j], model_list[j]) for j in
                            range(x.shape[0])]

                loss = 0
                for j, midx in enumerate(idx):
                    model_list[j].covar_module._set_ls(ls[j])
                    model_list[j].covar_module._set_scale(sc[j])
                    model_list[j].mean_module._set_bias(b[j])
                    model_list[j].mean_module._set_weights(m[j])
                    output = model_list[j](x[j, data_idx[j], :])
                    #print(output)
                    loss -= mll_list[j](output, y[j, data_idx[j]])


                loss.backward()
                store_loss.append(loss.item())
                optimizer.step()

            if i % checkpoint == 0:
                save_dict = {'ls_net': self.lengthscale_net.state_dict(),
                             'weight_net': self.weight_net.state_dict(),
                             'bias_net': self.bias_net.state_dict(),
                             'scale_net': self.scale_net.state_dict(),
                             'noise_net_1': self.noise_net_1.state_dict(),
                             'noise_net_2': self.noise_net_2.state_dict(),
                             'n_iter': i,
                             'lr': lr,
                             'loss': store_loss,
                             'batch_size': batch_size,
                             'matern': self.kernel,
                             'sigmoid': self.mean,
                             'ARD': self.ARD,
                             'h': self.h,
                             'd': self.n_layers}

                torch.save(save_dict, save_dir + 'model_' + str(i) + '_seed_' + str(self.seed) + '.pt')


        save_dict = {'ls_net': self.lengthscale_net.state_dict(),
                      'weight_net': self.weight_net.state_dict(),
                      'bias_net': self.bias_net.state_dict(),
                      'scale_net': self.scale_net.state_dict(),
                      'noise_net_1': self.noise_net_1.state_dict(),
                      'noise_net_2': self.noise_net_2.state_dict(),
                      'n_iter': i,
                      'loss': store_loss,
                      'batch_size': batch_size,
                      'lr':lr,
                      'matern': self.kernel,
                      'sigmoid': self.mean,
                      'ARD': self.ARD,
                      'h': self.h,
                      'd': self.n_layers}
        #
        torch.save(save_dict, save_dir + 'model_' + str(i) + '_seed_' + str(self.seed) + '.pt')

        return store_loss

    def set_MLP(self, bias_net, weight_net, scale_net, lengthscale_net):
        """
        Set-up MLPs
        """

        self.bias_net.load_state_dict(bias_net)
        self.bias_net.eval()

        self.weight_net.load_state_dict(weight_net)
        self.weight_net.eval()

        self.scale_net.load_state_dict(scale_net)
        self.scale_net.eval()

        self.lengthscale_net.load_state_dict(lengthscale_net)
        self.lengthscale_net.eval()


    def calc_likelihood_baseline(self, x, y, z, sigma=0.01, mfunc='linear', kfunc='rbf', eps=1e-10,
                                 condition_on_baseline=True):

        # set the correct GP parameters
        with torch.no_grad():
            ls = self.lengthscale_net(z)
            sc = self.scale_net(z)
            w = self.weight_net(z)
            b = self.bias_net(z)

        ### likelihood conditioned on baseline
        mll = 0
        if condition_on_baseline:
            for i in range(self.N):
                ls_i = ls[i].numpy()
                sc_i = sc[i].numpy()
                w_i = w[i].numpy()
                b_i = b[i].numpy()

                x_obs = x[i, :1, :].numpy()
                y_obs = y[i, :1].numpy()

                x_pred = x[i, 1:, :]
                y_pred = y[i, 1:]
                if torch.sum(~torch.isnan(x_pred)) > 0:
                    pred_idx = ~torch.isnan(x_pred[:, 0])
                    x_pred = x_pred[pred_idx, :].numpy()
                    y_pred = y_pred[pred_idx]

                    mean = calc_posterior_m(x_obs, y_obs, w_i, b_i, ls_i, sc_i, sigma, x_pred, mean=mfunc, kernel=kfunc,
                                            dim=ls_i.shape[0])
                    cov = calc_posterior_k(x_obs, ls_i, sc_i, sigma, x_pred, kernel=kfunc, dim=ls_i.shape[0])
                    mvn = create_mvn(torch.tensor(mean), torch.tensor(cov) + eps*torch.eye(cov.shape[0]))
                    mll += mvn.log_prob(y_pred)

        else:
            for i in range(self.N):
                ls_i = ls[i].numpy()
                sc_i = sc[i].numpy()
                w_i = w[i].numpy()
                b_i = b[i].numpy()

                x_i = x[i].numpy()
                y_i = y[i]

                obs_idx = ~np.isnan(x_i[:, 0])

                k = calc_k_rbf(x_i[obs_idx], ls_i, sc_i)
                m = calc_m_linear(x_i[obs_idx], w_i, b_i)

                mvn = create_mvn(torch.tensor(m), torch.tensor(k) + eps * torch.eye(k.shape[0]))
                mll += mvn.log_prob(y_i[obs_idx])

        return mll

    def calc_mll(self, x, y, z):
        mll_list = [gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_list[i], self.model_list[i]) for i in
                    range(x.shape[0])]

        data_idx = ~torch.isnan(x[:, :, 0])

        ls = self.lengthscale_net(z)
        sc = self.scale_net(z)
        m = self.weight_net(z)
        b = self.bias_net(z)

        tot_mll = 0
        for i in range(x.shape[0]):
            self.model_list[i].covar_module._set_ls(ls[i])
            self.model_list[i].covar_module._set_scale(sc[i])
            self.model_list[i].mean_module._set_bias(b[i])
            self.model_list[i].mean_module._set_weights(m[i])

            output = self.model_list[i](x[i, data_idx[i], :])
            tot_mll -= mll_list[i](output, y[i, data_idx[i]])

        return tot_mll
