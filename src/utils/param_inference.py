import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import delfi.distribution as dd
from src.data.franka_data_generator import FrankaDataGenerator
from src.models.mdn import MDNN, MDRFF, MDLSTM, MDRFFLSTM
from src.models.bayes_sim import BayesSim
from delfi.distribution.mixture.GaussianMixture import MoG


def get_mdn(n_components=2, nhidden=2, nunits=[24,24], output_dim=None, input_dim=None):
    model = MDNN(ncomp=n_components, outputd=output_dim, inputd=input_dim, nhidden=nhidden, nunits=nunits)
    return model


def get_mdrff(n_components=2, kernel="RBF", sigma=4, nfeat=154, output_dim=None, input_dim=None):
    model = MDRFF(ncomp=n_components, outputd=output_dim, inputd=input_dim, nfeat=nfeat,
                  sigma=sigma, kernel=kernel, quasiRandom=True)
    return model


def get_mdlstm(n_components=2, nhidden=3, nunits=[50, 24, 24], output_dim=None, input_dim=None):
    model = MDLSTM(ncomp=n_components, nhidden=nhidden, nunits=nunits, outputd=output_dim, inputd=input_dim)
    return model


def get_mdrfflstm(n_components=2, kernel="RBF", sigma=4, nfeat=154, nunits=50, output_dim=None, input_dim=None):
    model = MDRFFLSTM(ncomp=n_components, nfeat=nfeat, nunits=nunits, outputd=output_dim,
                      inputd=input_dim, sigma=sigma, kernel=kernel, quasiRandom=True)
    return model


def mog_data(tmp_posterior, shapes):
    for dim in range(shapes['params']):
        for k in range(tmp_posterior.n_components):
            print(r'component {}: mixture weight = {:.4f}; mean = {:.4f}; variance = {:.4f}'.format(
                k + 1, tmp_posterior.a[k], tmp_posterior.xs[k].m[dim], tmp_posterior.xs[k].S[dim][dim]))


def get_2d_posterior_data(posterior, xmin=0, xmax=2, ymin=0, ymax=2, nbins=100):
    xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]

    X = np.concatenate((xi.reshape(1, nbins * nbins), yi.reshape(1, nbins * nbins)), axis=0)
    zi = posterior.eval(X.T, ii=None, log=False)  # contour

    return xi, yi, zi


def plot_2d_posterior(env_params=None,
                      posterior=None,
                      true_obs=None,
                      xmin=0,
                      xmax=2,
                      ymin=0,
                      ymax=2,
                      data=None):

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((10, 10))
    cmap = cm.cool

    ax.set_xlim((xmin, xmax))
    ax.set_title(env_params[0] + ' vs ' + env_params[1], fontsize=20)
    ax.set_xlabel(env_params[0], fontsize=20)
    ax.set_ylabel(env_params[1], fontsize=20)  # Evaluate the mixture on a regular grid

    if posterior is not None:
        xi, yi, zi = get_2d_posterior_data(posterior, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    else:
        xi, yi, zi = data

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    plt.colorbar(spacing='proportional')
    ax.contour(xi, yi, zi.reshape(xi.shape), alpha=0.8)
    ax.set_xticks(np.arange(xmin, xmax, 0.5), minor=True)
    ax.set_yticks(np.arange(ymin, ymax, 0.5), minor=True)
    ax.scatter(true_obs[0][0], true_obs[0][1], 1000, 'y', marker='*', label='True value')
    plt.legend(fontsize=18)
    # And a corresponding grid
    ax.grid(which='both')  # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.6)
    ax.grid(which='major', alpha=0.8)
    plt.show()


def plot_1d_posterior(env_params=None, p_lower=[0.1, 0.1], p_upper=[2.0, 2.0], params_shape=2, posterior=None):
    p = dd.Uniform(lower=p_lower, upper=p_upper)

    for i in range(params_shape):

        minlim = p_lower[i] - 0.1 * p_lower[i]
        maxlim = p_upper[i] + 0.1 * p_upper[i]

        x_plot = np.arange(minlim, maxlim, 0.001).reshape(-1, 1)
        y_plot = posterior.eval(x_plot, ii=[i], log=False)
        y_plot_prior = p.eval(x_plot, ii=[i], log=False)
        plt.plot(x_plot, y_plot, '-b', label=r'Predicted posterior')
        plt.plot(x_plot, y_plot_prior, '-g', label=r'Uniform prior')
        cur_true_param = true_obs.ravel()[i]
        plt.axvline(cur_true_param, c='r', label=r'True value of ' + env_params[i])
        plt.legend(fontsize=10)
        plt.axis('on')
        plt.xlabel(env_params[i] + ' values')
        # fig.savefig('/tmp/'+env_params[i]+'.pdf', bbox_inches='tight')
        plt.show()


def plot_posterior(env_params=["length", "mass"], true_obs=None, posterior=None, params_shape=2,
                   p_lower=[0.1, 0.1], p_upper=[2.0, 2.0]):

    plot_2d_posterior(env_params=env_params, posterior=posterior, true_obs=true_obs)



def train(batch_size=250, epochs=500, params_dim=1, stats_dim=154, n_components=10,
               num_sampled_points=None, generator=None, model="MDN"):

    # num_sampled_points = generator.total_size if num_sampled_points is None else num_sampled_points

    if model == "MDN":
        model = get_mdn(n_components=n_components, output_dim=params_dim, input_dim=stats_dim)

    elif model == "MDRFF":
        model = get_mdrff(n_components=n_components, nfeat=1000, sigma=3, kernel="Matern32",
                          output_dim=params_dim, input_dim=stats_dim)
    elif model == "MDLSTM":
        model = get_mdlstm(n_components=n_components, output_dim=params_dim, input_dim=stats_dim)

    elif model == "MDRFFLSTM":
        model = get_mdrfflstm(n_components=n_components, nfeat=1000, sigma=3, kernel="Matern32",
                          output_dim=params_dim, input_dim=stats_dim)

    inf = BayesSim(generator=generator, model=model, params_dim=params_dim, stats_dim=stats_dim)
    log, train_data = inf.run(n_train=num_sampled_points, epochs=epochs, n_rounds=1,
                              batch_size=batch_size)

    plt.plot(log[0]['loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    return log, inf


def get_posterior_from_true_obs(generator=None, inf=None, true_obs=None):
    params, data = generator.run_forward_model(true_obs=true_obs, ntest=10)
    shapes = {"params": params.shape[0], "data": data.shape[0]}
    posterior = inf.predict(data.reshape(1, -1))
    mog_data(posterior, shapes)

    return posterior

def get_results_from_true_obs(true_obs=None, generator=None, inf=None, shapes=None, env_params=None,
                              p_lower=None, p_upper=None):

    if generator.params_scaler is not None:
        true_obs = generator.params_scaler.transform(true_obs)

    print("Scaled true obs: {}".format(true_obs))

    print("Params: {}".format(env_params))

    tmp_posterior = get_posterior_from_true_obs(generator=generator, inf=inf, true_obs=true_obs)

    plot_posterior(env_params=env_params, true_obs=true_obs, posterior=tmp_posterior, params_shape=shapes["params"],
                   p_lower=p_lower, p_upper=p_upper)


if __name__ == "__main__":
    cur_root_dir = os.path.split(os.getcwd())[0]
    data_file = os.path.join(os.path.join(cur_root_dir, "assets/data_stiffness_5k.pkl"))

    g = FrankaDataGenerator(data_file=data_file, load_from_disk=True, params_dim=1, data_dim=154)
    params, stats = g.gen(1)
    shapes = {"params": params.shape[1], "data": stats.shape[1]}
    print("Total data size: {}".format(g.total_size))

    log, inf = train(epochs=100, batch_size=500, generator=g, model="MDRFF")
    true_obs = np.array([[6000]])

    get_results_from_true_obs(true_obs, generator=g, inf=inf, shapes=shapes, param_name="Stiffness", model_name="MDN")

