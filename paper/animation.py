import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nautilus import Sampler
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal

mean = 0.5
sigma = 0.1


def prior(x):
    return x


def likelihood(x):
    return multivariate_normal.logpdf(
        x, mean=[mean, mean],
        cov=np.array([[1.0, 0.5], [0.5, 1.0]]) * sigma**2)


sampler = Sampler(prior, likelihood, 2, random_state=0)

points = []

while sampler.live_evidence_fraction() > 0.1:
    sampler.add_bound(verbose=True)
    sampler.fill_bound(verbose=True)
    points_iteration = sampler.posterior()[0]
    for i in range(len(points)):
        points_iteration = points_iteration[~np.array([np.any(np.all(
            p == points[i], axis=1)) for p in points_iteration])]
    points.append(points_iteration)

x_bound = []
y_bound = []

for bound in sampler.bounds:
    points_sample = bound.sample(1000000)
    hull = ConvexHull(points_sample)
    x_bound.append(points_sample[hull.vertices, 0])
    x_bound[-1] = np.append(x_bound[-1], x_bound[-1][0])
    y_bound.append(points_sample[hull.vertices, 1])
    y_bound[-1] = np.append(y_bound[-1], y_bound[-1][0])


shells = np.arange(len(sampler.bounds)) + 1
plt.plot(shells, sampler.shell_log_v, label=r'$\log V$')
plt.plot(shells, sampler.shell_log_l,
         label=r'$\log \langle \mathcal{L} \rangle$')
plt.plot(shells, sampler.shell_log_v + sampler.shell_log_l,
         label=r'$\log \mathcal{Z}$')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.minorticks_off()
plt.xlabel('Shell')
plt.legend(loc='upper right', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(Path('animation') / 'shells.png', dpi=300)
plt.close()

# %%

np.random.seed(0)

fig = plt.figure(figsize=(8, 6))

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = np.vstack([likelihood(np.stack([x, y]).T) for x, y in zip(X, Y)])
levels = [np.percentile(Z, 100 * (1 - 1.7**-i)) for i in range(10)]
levels.append(np.amax(Z))
levels[0] = -np.inf
plt.contourf(Z, extent=[0, 1, 0, 1], levels=levels)
plt.tick_params(axis='both', which='both', bottom=False, left=False,
                right=False, top=False, labelbottom=False, labelleft=False)
plt.tight_layout(pad=0.1)
plt.savefig(Path('animation') / 'likelihood.png', dpi=300)

for i in range(len(x_bound)):
    plt.plot(x_bound[i], y_bound[i], color='white', zorder=+1)
plt.savefig(Path('animation') / 'explored.png', dpi=300)

points_plot, log_w, log_l = sampler.posterior()
log_g = log_w - log_l
scatter = plt.scatter(points_plot[:, 0], points_plot[:, 1], c=log_g,
                      cmap='inferno', s=1, zorder=+2)
plt.savefig(Path('animation') / 'importance.png', dpi=300)
scatter.remove()

select = np.random.random(len(points_plot)) < np.exp(log_w - np.amax(log_w))
scatter = plt.scatter(points_plot[:, 0][select], points_plot[:, 1][select],
                      s=1, color='black', zorder=+2)
plt.savefig(Path('animation') / 'sample.png', dpi=300)
scatter.remove()

# %%

for i in range(len(x_bound)):
    plt.gca().lines.pop()

# %%

length = 15
frames_points = 50
frames_wait = 20
frames_iteration = (frames_points + frames_wait)
frames = len(points) * frames_iteration

scatter = plt.scatter([], [], marker='o', s=1, color='black')
line = plt.plot(x_bound[0], y_bound[0], color='white')[0]
plt.xlim(0, 1)
plt.ylim(0, 1)

text = plt.text(0.05, 0.95, 'Iteration 1', color='white', ha='left', va='top',
                transform=plt.gca().transAxes, fontsize=25)


def animate(frame):

    print('Frame {}/{}'.format(frame + 1, frames))

    iteration = int(frame // frames_iteration)
    text.set_text('Iteration {}'.format(iteration + 1))

    points_plot = points[frame // frames_iteration]
    points_plot = points_plot[:int(
        len(points_plot) * (frame % frames_iteration) / frames_points)]
    if iteration > 0:
        points_plot = np.vstack(
            [points_plot, np.vstack(points[:frame // frames_iteration])])
    if frame > 0:
        scatter.set_offsets(points_plot)

    # Show and zoom into the next bound.
    if (iteration < len(points) - 1 and
            frame % frames_iteration >= frames_points):
        line.set_data(x_bound[iteration + 1], y_bound[iteration + 1])

    return scatter, text, line


ani = animation.FuncAnimation(
    fig, animate, frames=frames, interval=1000 * length / frames)
ani.save(Path('animation') / 'animation.mp4', fps=frames / length, bitrate=-1,
         extra_args=['-pix_fmt', 'yuv420p'])
