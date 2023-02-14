import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.ticker import MultipleLocator
from nautilus import Sampler
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal

mean = 0.5
sigma = 0.1


def prior(x):
    return x


def likelihood(x):
    return multivariate_normal.logpdf(x, mean=[mean, mean], cov=sigma**2)


sampler = Sampler(prior, likelihood, 2, random_state=0)

points = []

while sampler.live_evidence_fraction() > 0.01:
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
plt.savefig('animation_shells.png', dpi=300)
plt.close()

# %%

fig = plt.figure(figsize=(8, 6))

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = - ((X - mean)**2 + (Y - mean)**2) / sigma**2
plt.contourf(-np.sqrt(-Z), extent=[0, 1, 0, 1],
             levels=-(np.append(np.linspace(0, 5, 16), 10))[::-1], vmin=-5)
plt.tick_params(axis='both', which='both', bottom=False, left=False,
                right=False, top=False, labelbottom=False, labelleft=False)
plt.tight_layout(pad=0.1)
plt.savefig('animation_likelihood.png', dpi=300)

for i in range(len(x_bound)):
    plt.plot(x_bound[i], y_bound[i], color='white')

plt.savefig('animation_likelihood_explored.png', dpi=300)

for i in range(len(x_bound)):
    plt.gca().lines.pop()

# %%

length = 15
frames_points = 50
frames_wait = 25
frames_zoom = 25
frames_iteration = (frames_points + frames_wait + frames_zoom)
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
            frame % frames_iteration >= frames_points + frames_wait):

        line.set_data(x_bound[iteration + 1], y_bound[iteration + 1])

        if iteration == 0:
            x_min_old = 0
            x_max_old = 1
            y_min_old = 0
            y_max_old = 1
        else:
            x_min_old = np.amin(x_bound[iteration])
            x_max_old = np.amax(x_bound[iteration])
            y_min_old = np.amin(y_bound[iteration])
            y_max_old = np.amax(y_bound[iteration])
            x_min_old -= (x_max_old - x_min_old) * 0.1
            x_max_old += (x_max_old - x_min_old) * 0.1
            y_min_old -= (y_max_old - y_min_old) * 0.1
            y_max_old += (y_max_old - y_min_old) * 0.1

        x_min_new = np.amin(x_bound[iteration + 1])
        x_max_new = np.amax(x_bound[iteration + 1])
        y_min_new = np.amin(y_bound[iteration + 1])
        y_max_new = np.amax(y_bound[iteration + 1])
        x_min_new -= (x_max_new - x_min_new) * 0.1
        x_max_new += (x_max_new - x_min_new) * 0.1
        y_min_new -= (y_max_new - y_min_new) * 0.1
        y_max_new += (y_max_new - y_min_new) * 0.1

        f = max(0, (frame % frames_iteration) -
                frames_points - frames_wait + 1) / frames_zoom
        x_min = x_min_old + (x_min_new - x_min_old) * f
        x_max = x_max_old + (x_max_new - x_max_old) * f
        y_min = y_min_old + (y_min_new - y_min_old) * f
        y_max = y_max_old + (y_max_new - y_max_old) * f
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    return scatter, text, line


ani = animation.FuncAnimation(
    fig, animate, frames=frames, interval=1000 * length / frames)
ani.save('animation.mp4', fps=frames / length, bitrate=10000,
         extra_args=['-pix_fmt', 'yuv420p'])
