import numpy as np
from pathlib import Path
from nautilus import Sampler
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from draw_neural_net import draw_neural_net
from likelihoods.analytic import rosenbrock_likelihood
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import percentileofscore, multivariate_normal

path = Path('figures')


def prior(x):
    return x

# %%


cmap = plt.get_cmap('viridis')

plt.figure(figsize=(2.5, 2.5))
b_1 = Ellipse((0.5, 0.5), 1.3, 0.5, 45, fc=cmap(0.8), ec='black')
b_2 = Ellipse((0.5, 0.5), 0.65, 0.5, 45, fc=cmap(0.5), ec='black')
b_3 = Ellipse((0.5, 0.5), 0.3, 0.6, 45, fc=cmap(0.2), ec='black', alpha=0.8)
plt.gca().add_patch(b_1)
plt.gca().add_patch(b_2)
plt.gca().add_patch(b_3)
plt.text(0.1, 0.9, r'$B_1$', ha='left', va='top')
plt.text(0.9, 0.9, r'$B_2$', ha='right', va='top')
plt.text(0.7, 0.7, r'$B_3$', ha='right', va='top')
plt.text(0.5, 0.5, r'$B_4$', ha='center', va='center')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.3)
plt.savefig(path / 'non-nested.png', dpi=300)
plt.savefig(path / 'non-nested.pdf')
plt.close()

# %%


def likelihood(x):
    return multivariate_normal.logpdf(
        x, mean=[0.5, 0.5], cov=np.array([[1, 0.5], [0.5, 1]]) * 0.05)


fig, axarr = plt.subplots(figsize=(7, 2), ncols=3, nrows=1)

x = np.linspace(0, +1, 2000)
y = np.linspace(0, +1, 2000)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T
in_bound = np.zeros_like(X)
cmap = plt.get_cmap('Blues')
colors = cmap([0.0, 0.3, 0.6])

sampler = Sampler(prior, likelihood, n_dim=2, n_update=4000)

for i, ax in enumerate(axarr):
    sampler.add_bound(verbose=True)
    sampler.fill_bound(verbose=True)
    in_bound += sampler.bounds[-1].contains(points).reshape(
        X.shape).astype(int)
    ax.contourf(X, Y, in_bound, levels=[0.5, 1.5, 2.5, 3.5],
                colors=colors, zorder=i)
    ax.scatter(sampler.points[-1][::10, 0], sampler.points[-1][::10, 1],
               color='black', s=3, lw=0, zorder=101)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.95, 'Iteration {}'.format(i + 1),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes, color='red', zorder=100)
    ax.set_rasterization_zorder(10)

plt.tight_layout(pad=0.3)
plt.savefig(path / 'exploration.pdf')
plt.savefig(path / 'exploration.png', dpi=300)
plt.close()

# %%

fig, axarr = plt.subplots(figsize=(7, 2), ncols=3, nrows=1)
points, log_w, log_l = sampler.posterior()
w = np.exp(log_w)
w = w / np.amax(w)
points = points[np.random.random(len(points)) < w]
i_shell = sampler.shell_association(points)
for i, ax in enumerate(axarr):
    ax.contourf(X, Y, in_bound, levels=[0.5, 1.5, 2.5, 3.5],
                colors=colors, zorder=i)
    mask = i_shell == i
    ax.scatter(points[mask, 0][::10], points[mask, 1][::10], color='black',
               s=3, lw=0, zorder=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(0.5, 0.95, 'Posterior Shell {}'.format(i + 1),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes, color='red', zorder=101)
    ax.set_rasterization_zorder(10)
plt.tight_layout(pad=0.3)
plt.savefig(path / 'posterior.pdf')
plt.savefig(path / 'posterior.png', dpi=300)
plt.close()

# %%

sampler = Sampler(
    prior, rosenbrock_likelihood, n_dim=2, vectorized=True, pass_dict=False,
    random_state=0)
for i in range(6):
    sampler.add_bound(verbose=True)
    sampler.fill_bound(verbose=True)

sampler.add_bound(verbose=True)

# %%

fig, axarr = plt.subplots(figsize=(7, 4), ncols=3, nrows=2)

ax = axarr[0, 0]
ax.text(0.05, 0.95, '1', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
points = np.vstack(sampler.points[:-1]) * 10 - 5
log_l = np.concatenate(sampler.log_l)
log_l_min = sampler.shell_log_l_min[-1]
img = ax.scatter(points[::3, 0], points[::3, 1], s=1, c=log_l[::3], vmin=-1e5,
                 vmax=0, rasterized=True, lw=0, cmap='viridis')
use = log_l > log_l_min
ax.scatter(points[:, 0][use][::3], points[:, 1][use][::3], s=1,
           rasterized=True, lw=0, color='black')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig.colorbar(img, cax=cax, orientation='vertical', ticks=[-1e5, 0])
cbar.ax.set_yticklabels(['-$10^5$', '0'])
ax.set_xlim(-5, +5)
ax.set_ylim(-5, +5)
cbar.set_label(r'$\log \mathcal{L}$', labelpad=-10)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

ax = axarr[0, 1]
ax.text(0.05, 0.95, '2', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
points = np.vstack(sampler.points[:-1])
use = sampler.bounds[-1].neural_bounds[0].outer_bound.contains(points)
points = points[use]
log_l = log_l[use]
points = points * 10 - 5
img = ax.scatter(points[::3, 0], points[::3, 1], s=1, c=log_l[::3], vmin=-1e2,
                 vmax=0, rasterized=True, lw=0, cmap='viridis')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig.colorbar(img, cax=cax, orientation='vertical', ticks=[-1e2, 0])
cbar.ax.set_yticklabels(['-$10^2$', '0'])
ax.set_xlim(-5, +5)
ax.set_ylim(-5, +5)
cbar.set_label(r'$\log \mathcal{L}$', labelpad=-10)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

ax = axarr[0, 2]
ax.text(0.05, 0.95, '3', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
points = (points + 5) / 10
points = sampler.bounds[-1].neural_bounds[0].outer_bound.transform(points)
p_l = np.argsort(np.argsort(log_l)) / float(len(log_l))
p_l_min = percentileofscore(log_l, log_l_min) / 100
s_l = np.where(p_l < p_l_min, p_l / p_l_min / 2,
               0.5 + (p_l - p_l_min) / (1 - p_l_min) / 2)
img = ax.scatter(points[::3, 0], points[::3, 1], s=1.5, c=s_l[::3], vmin=0,
                 vmax=1, rasterized=True, lw=0, cmap='viridis')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig.colorbar(img, cax=cax, orientation='vertical', ticks=[0, 1])
ax.set_xlim(-1, +1)
ax.set_ylim(-1, +1)
cbar.set_label(r'$s_\mathcal{L}$', labelpad=0)
ax.set_xlabel(r'$\tilde{x}_1$')
ax.set_ylabel(r'$\tilde{x}_2$')

ax = axarr[1, 0]
ax.text(0.05, 0.95, '4', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
draw_neural_net(ax, 0.05, 0.95, 0.0, 1.0, [2, 8, 4, 2, 1])
ax.text(0.0, 0.5625, r'$\tilde{x}_1$', horizontalalignment='right',
        verticalalignment='center', transform=ax.transAxes)
ax.text(0.0, 0.4375, r'$\tilde{x}_2$', horizontalalignment='right',
        verticalalignment='center', transform=ax.transAxes)
ax.text(1.0, 0.5, r'$s_\mathcal{L}$', horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
ax.axis('off')

ax = axarr[1, 1]
ax.text(0.05, 0.95, '5', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
s_l_pred = sampler.bounds[-1].neural_bounds[0].emulator.predict(points)
ax.scatter(s_l[::5], s_l_pred[::5], s=1, color='black', rasterized=True, lw=0)
ax.axhline(sampler.bounds[-1].neural_bounds[0].score_predict_min, ls='--',
           color='black')
ax.set_xlabel(r'$s_\mathcal{L}$')
ax.set_ylabel(r'$\hat{s}_\mathcal{L}$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax = axarr[1, 2]
ax.text(0.05, 0.95, '6', horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes, color='red')
x = np.linspace(0, +1, 3000)
y = np.linspace(0, +1, 3000)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T
X = X * 10 - 5
Y = Y * 10 - 5
in_bound = sampler.bounds[-1].contains(points).reshape(X.shape)
ax.contour(X, Y, in_bound, levels=[0.5], colors='black', linewidths=1,
           zorder=-10)
ax.set_rasterization_zorder(0)
ax.set_xlim(-5, +5)
ax.set_ylim(-5, +5)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.tight_layout(pad=0.3)
plt.savefig(path / 'bounds.pdf')
plt.savefig(path / 'bounds.png', dpi=300, transparent=True)
