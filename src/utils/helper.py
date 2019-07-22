import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import pickle
import delfi.distribution as dd
# def disp_imdata(xs, imsize, layout=(1,1)):
#     """
#     Displays an array of images, a page at a time. The user can navigate pages with
#     left and right arrows, start over by pressing space, or close the figure by esc.
#     :param xs: an numpy array with images as rows
#     :param imsize: size of the images
#     :param layout: layout of images in a page
#     :return: none
#     """
#
#     num_plots = np.prod(layout)
#     num_xs = xs.shape[0]
#     idx = [0]
#
#     # create a figure with suplots
#     fig, axs = plt.subplots(layout[0], layout[1])
#
#     if isinstance(axs, np.ndarray):
#         axs = axs.flatten()
#     else:
#         axs = [axs]
#
#     for ax in axs:
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#
#     def plot_page():
#         """Plots the next page."""
#
#         ii = np.arange(idx[0], idx[0]+num_plots) % num_xs
#
#         for ax, i in zip(axs, ii):
#             ax.imshow(xs[i].reshape(imsize), cmap='gray', interpolation='none')
#             ax.set_title(str(i))
#
#         fig.canvas.draw()
#
#     def on_key_event(event):
#         """Event handler after key press."""
#
#         key = event.key
#
#         if key == 'right':
#             # show next page
#             idx[0] = (idx[0] + num_plots) % num_xs
#             plot_page()
#
#         elif key == 'left':
#             # show previous page
#             idx[0] = (idx[0] - num_plots) % num_xs
#             plot_page()
#
#         elif key == ' ':
#             # show first page
#             idx[0] = 0
#             plot_page()
#
#         elif key == 'escape':
#             # close figure
#             plt.close(fig)
#
#     fig.canvas.mpl_connect('key_press_event', on_key_event)
#     plot_page()
#
#
# def isdistribution(p):
#     """
#     :param p: a vector representing a discrete probability distribution
#     :return: True if p is a valid probability distribution
#     """
#     return np.all(p >= 0.0) and np.isclose(np.sum(p), 1.0)
#
#
# def discrete_sample(p, n_samples=1):
#     """
#     Samples from a discrete distribution.
#     :param p: a distribution with N elements
#     :param n_samples: number of samples
#     :return: vector of samples
#     """
#
#     # check distribution
#     #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'
#
#     # cumulative distribution
#     c = np.cumsum(p[:-1])[np.newaxis, :]
#
#     # get the samples
#     r = rng.rand(n_samples, 1)
#     return np.sum((r > c).astype(int), axis=1)
#
#
# def ess_importance(ws):
#     """Calculates the effective sample size of a set of weighted independent samples (e.g. as given by importance
#     sampling or sequential monte carlo). Takes as input the normalized sample weights."""
#
#     ess = 1.0 / np.sum(ws ** 2)
#     return ess
#
#
# def ess_mcmc(xs):
#     """Calculates the effective sample size of a correlated sequence of samples, e.g. as given by markov chain monte
#     carlo."""
#
#     n_samples, n_dim = xs.shape
#
#     mean = np.mean(xs, axis=0)
#     xms = xs - mean
#
#     acors = np.zeros_like(xms)
#     for i in range(n_dim):
#         for lag in range(n_samples):
#             acor = np.sum(xms[:n_samples-lag, i] * xms[lag:, i]) / (n_samples - lag)
#             if acor <= 0.0: break
#             acors[lag, i] = acor
#
#     act = 1.0 + 2.0 * np.sum(acors[1:], axis=0) / acors[0]
#     ess = n_samples / act
#
#     return np.min(ess)
#
#
# def probs2contours(probs, levels):
#     """
#     Takes an array of probabilities and produces an array of contours at specified percentile levels
#     :param probs: probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
#     :param levels: percentile levels. have to be in [0.0, 1.0]
#     :return: array of same shape as probs with percentile labels
#     """
#
#     # make sure all contour levels are in [0.0, 1.0]
#     levels = np.asarray(levels)
#     assert np.all(levels <= 1.0) and np.all(levels >= 0.0)
#
#     # flatten probability array
#     shape = probs.shape
#     probs = probs.flatten()
#
#     # sort probabilities in descending order
#     idx_sort = probs.argsort()[::-1]
#     idx_unsort = idx_sort.argsort()
#     probs = probs[idx_sort]
#
#     # cumulative probabilities
#     cum_probs = probs.cumsum()
#     cum_probs /= cum_probs[-1]
#
#     # create contours at levels
#     contours = np.ones_like(cum_probs)
#     levels = np.sort(levels)[::-1]
#     for level in levels:
#         contours[cum_probs <= level] = level
#
#     # make sure contours have the order and the shape of the original probability array
#     contours = np.reshape(contours[idx_unsort], shape)
#
#     return contours
#
#

def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = rng.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)

def plot_pdf_marginals(pdf, lims, gt=None, levels=(0.68, 0.95), title=""):
    """Plots marginals of a pdf, for each variable and pair of variables."""

    if pdf.ndim == 1:

        fig, ax = plt.subplots(1, 1, num=title)
        xx = np.linspace(lims[0], lims[1], 200)

        pp = pdf.eval(xx[:, np.newaxis], log=False)
        ax.plot(xx, np.array(pp).flatten())
        ax.set_xlim(lims)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.title.set_text(title)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        fig, ax = plt.subplots(pdf.ndim, pdf.ndim, num=title)

        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf.ndim, 1]) if lims.ndim == 1 else lims

        for i in range(pdf.ndim):
            for j in range(pdf.ndim):

                if i == j:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 500)
                    pp = pdf.eval(xx, ii=[i], log=False)
                    ax[i, j].plot(xx, pp)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                # else:
                #     xx = np.linspace(lims[i, 0], lims[i, 1], 200)
                #     yy = np.linspace(lims[j ,0], lims[j, 1], 200)
                #     X, Y = np.meshgrid(xx, yy)
                #     xy = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                #     pp = pdf.eval(xy, ii=[i, j], log=False)
                #     pp = pp.reshape(list(X.shape))
                #     ax[i, j].contour(X, Y, probs2contours(pp, levels), levels)
                #     ax[i, j].set_xlim(lims[i])
                #     ax[i, j].set_ylim(lims[j])
                #     if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)
    return fig, ax
#
#
# def plot_hist_marginals(data, lims=None, gt=None):
#     # Plots marginal histograms and pairwise scatter plots of a dataset.
#
#     n_bins = int(np.sqrt(data.shape[0]))
#
#     if data.ndim == 1:
#
#         fig, ax = plt.subplots(1, 1)
#         ax.hist(data, n_bins, normed=True)
#         ax.set_ylim([0, ax.get_ylim()[1]])
#         if lims is not None: ax.set_xlim(lims)
#         if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')
#
#     else:
#
#         n_dim = data.shape[1]
#         fig, ax = plt.subplots(n_dim, n_dim)
#         ax = np.array([[ax]]) if n_dim == 1 else ax
#
#         if lims is not None:
#             lims = np.asarray(lims)
#             lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims
#
#         for i in range(n_dim):
#             for j in range(n_dim):
#
#                 if i == j:
#                     ax[i, j].hist(data[:, i], n_bins, normed=True)
#                     ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
#                     if lims is not None: ax[i, j].set_xlim(lims[i])
#                     if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')
#
#                 else:
#                     ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
#                     if lims is not None:
#                         ax[i, j].set_xlim(lims[i])
#                         ax[i, j].set_ylim(lims[j])
#                     if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)
#
#     plt.show(block=False)
#
#     return fig, ax


def gaussian_distribution(y, mu, sigma):
    gauss_normalization = 1.0 / np.sqrt(2.0 * np.pi)  # normalization factor for Gaussians
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * gauss_normalization


def save(data, file):
    """Saves data to a file."""

    f = open(file, 'wb')
    pickle.dump(data, f)
    f.close()


def load(file):
    """Loads data from file."""

    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data


# Update env parameters in multi environments
def update_env(env, param):
    env.sim.model.geom_friction[22][0] = param
    env.sim.model.geom_friction[23][0] = param


# Update env parameters

def getparam(env):
    return env.sim.model.geom_friction[22][0]


# Do a rollout
def rollout(policy, env, gamma=1., num_rollouts=5, seed=0):
    env.seed(seed)
    running_reward = 0.
    running_discounted = 0.
    all_rewards = []
    for i in range(num_rollouts):
        running_reward = 0
        state = env.reset()
        for j in range(50):
            # Pick an action given the policy
            a = policy.get_actions(state['observation'], state['achieved_goal'], state['desired_goal'])
            # Clip Action to avoid out of bound errors
            state, r, _, info = env.step(a)
            #env.render()
            running_reward += r
            running_discounted = running_discounted + gamma * r
            episode_done = bool(info['is_success'])
            if episode_done:
                break

        all_rewards.append(running_reward)
    return np.mean(all_rewards)

# Plots the robostness of a policy
def plot_robustness(policies, env, env_params, ntests=5, distribution=None, true_param=None, title=""):
    for policy in policies:
        for i in range(len(env_params)):
            # Gets the initial value
            plt.figure(i)

            if isinstance(distribution, dd.Gaussian):
                # Get limits from a Gaussian prior
                minlim = distribution.mean[i] - distribution.std[i]
                maxlim = distribution.mean[i] + distribution.std[i]
            else:
                # Get limits from a Uniform prior
                minlim = distribution.lower[i] - 0.1 * distribution.lower[i]
                maxlim = distribution.upper[i] + 0.1 * distribution.upper[i]

            x_plot = np.arange(minlim, maxlim, 0.05).reshape(-1)
            y_plot = np.zeros((x_plot.size, ntests))
            print("Testing robustness with {} parameters".format(x_plot.size))
            for k in range(ntests):
                print("Running test {} of {}".format(k, ntests))
                for j in range(x_plot.size):
                    update_env(env,  x_plot[j])
                    y_plot[j, k] = rollout(policy=policy, env=env, num_rollouts=5, seed=k)

            mean_y = np.mean(y_plot, 1)
            std_y = np.std(y_plot, 1)
            plt.plot(x_plot, mean_y, '-b', label=r'Accumulated rewards mean')
            plt.fill_between(x_plot, mean_y - std_y, mean_y + std_y, label=r'Accumulated rewards std', alpha=0.2)

    plt.axvline(true_param, c='r', label=r'Real value of ' + env_params[i])
    plt.legend(fontsize=10)
    plt.axis('on')
    plt.xlabel(env_params[i])
    plt.ylabel("reward")
    plt.title(title)
    plt.xlabel("friction")
    plt.show()
    return x_plot, y_plot