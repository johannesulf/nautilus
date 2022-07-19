import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack

# %%

summary = []


for folder in os.listdir('benchmarks'):

    directory = os.path.join('benchmarks', folder)

    if not os.path.isdir(directory) or folder == '__pycache__':
        continue

    results = vstack([Table.read(os.path.join(directory, fname)) for fname in
                      os.listdir(directory)])

    for i, sampler in enumerate(np.unique(results['sampler'])):
        print(folder, sampler, np.sum(results['sampler'] == sampler))

        select = results['sampler'] == sampler
        summary_row = {}
        summary_row['sampler'] = sampler
        summary_row['problem'] = folder
        if sampler == 'emcee':
            summary_row['log Z'] = np.nan
            summary_row['log Z error'] = np.nan
        else:
            summary_row['log Z'] = np.mean(results['log Z'][select])
            summary_row['log Z error'] = np.std(results['log Z'][select],
                                                ddof=1)
        summary_row['N_like'] = np.mean(results['N_like'][select])
        summary_row['efficiency'] = np.mean(
            (results['N_eff'] / results['N_like'])[select])
        if sampler == 'emcee':
            summary_row['N_like'] = 50 * 100 / summary_row['efficiency']

        summary.append(summary_row)

    n_dim = np.sum([name[:2] == 'x_' for name in results.colnames])
    ncols = int(np.ceil(np.sqrt(n_dim + 1)))
    nrows = int(np.ceil((n_dim + 1) / ncols))
    if ncols == nrows and nrows > 2:
        ncols = ncols + 1
        nrows = int(np.ceil((n_dim + 1) / ncols))
    fig, axarr = plt.subplots(figsize=(7, 7 * nrows / ncols), nrows=nrows,
                              ncols=ncols)
    axarr = axarr.flatten()
    for i in range(n_dim):
        lines = []
        labels = []
        for sampler in np.unique(results['sampler']):

            if sampler in ['Nautilus-resample', ]:
                continue

            x_bins = np.linspace(0, 1, 1001)
            x = 0.5 * (x_bins[1:] + x_bins[:-1])
            select = results['sampler'] == sampler
            pdf = np.mean(results[select]['x_{}'.format(i)], axis=0)
            pdf_all = np.mean(results['x_{}'.format(i)], axis=0)
            n_bins = np.sum(pdf_all > np.amax(pdf_all) * 1e-3)
            kernel_size = n_bins // 50

            while kernel_size == 0 or 1000 % kernel_size != 0:
                kernel_size += 1

            x = np.mean(x.reshape(1000 // kernel_size, kernel_size),
                        axis=-1)
            pdf = np.mean(pdf.reshape(1000 // kernel_size, kernel_size),
                          axis=-1)

            lines.append(axarr[i].plot(x, pdf, alpha=0.5)[0])
            labels.append(sampler)

        x_bins = np.linspace(0, 1, 1001)
        x = 0.5 * (x_bins[1:] + x_bins[:-1])
        axarr[i].set_xlim(np.amin(x[pdf_all > np.amax(pdf_all) * 1e-3]),
                          np.amax(x[pdf_all > np.amax(pdf_all) * 1e-3]))
        axarr[i].set_ylim(ymin=0)
        if folder == 'galaxy':
            text = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$',
                    r'$\log M_1$', r'$\alpha$', r'$A_{\rm cen}$',
                    r'$A_{\rm sat}$'][i]
        else:
            text = r'$x_{{{}}}$'.format(i + 1)

        axarr[i].text(0.05, 0.95, text, ha='left', va='top',
                      transform=axarr[i].transAxes)
        axarr[i].set_xticks([])
        axarr[i].set_xticks([], minor=True)
        axarr[i].set_yticks([])
        axarr[i].set_yticks([], minor=True)
        ymin, ymax = axarr[i].get_ylim()
        axarr[i].set_ylim(ymax=ymax * 1.1)

    for i in range(n_dim, nrows * ncols):
        axarr[i].axis('off')

    if n_dim < 20:
        axarr[n_dim].legend(lines, labels, loc='center', frameon=False)
    else:
        axarr[-1].legend(lines, labels, loc='center', frameon=False)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join('plots', folder + '_posterior.pdf'))
    plt.savefig(os.path.join('plots', folder + '_posterior.png'), dpi=300)
    plt.close()

# %%

summary = Table(summary)

for problem in np.unique(summary['problem']):
    select = summary['problem'] == problem
    log_z_min = np.nanmin(
        (summary['log Z'] - 4 * summary['log Z error'])[select])
    log_z_max = np.nanmax(
        (summary['log Z'] + 4 * summary['log Z error'])[select])
    log_z = np.linspace(log_z_min, log_z_max, 10000)
    for row in summary[select]:
        if row['sampler'] == 'emcee':
            continue
        plt.plot(log_z, np.exp(
            -0.5 * ((log_z - row['log Z']) / row['log Z error'])**2),
            label=row['sampler'])
    plt.legend(loc='upper center', frameon=False, prop={'size': 9}, ncol=2)
    plt.xlabel(r'Evidence $\log \mathcal{Z}$')
    plt.xlim(log_z_min, log_z_max)
    plt.ylim(0, 1.5)
    plt.gca().set_yticks([])
    plt.gca().set_yticks([], minor=True)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join('plots', problem + '_evidence.pdf'))
    plt.savefig(os.path.join('plots', problem + '_evidence.png'), dpi=300)
    plt.close()

# %%

plt.figure(figsize=(7, 3.0))
ax1 = plt.subplot2grid((8, 2), (1, 0), rowspan=7)
ax2 = plt.subplot2grid((8, 2), (1, 1), rowspan=7)
ax3 = plt.subplot2grid((8, 2), (0, 0), colspan=2)

problem_list = ['rosenbrock-10', 'funnel-20', 'loggamma-30', 'galaxy',
                'exoplanet']
label_list = [r'Rosen$_{10}$', r'Funnel$_{20}$', r'Log$\Gamma_{30}$',
              'Galaxy', 'Exoplanet']
sampler_list = ['Nautilus', 'UltraNest', 'dynesty-unif',
                'dynesty-rwalk', 'dynesty-slice', 'pocoMC']
color_list = ['purple', 'darkblue', 'orange', 'orange', 'orange',
              'royalblue']
label_set = np.zeros(len(sampler_list), dtype=bool)
marker_list = ['o', 'p', 'd', 's', 'v', 'h']

for i, problem in enumerate(problem_list):
    for k, (sampler, color, marker) in enumerate(
            zip(sampler_list, color_list, marker_list)):
        select = ((summary['problem'] == problem) &
                  (summary['sampler'] == sampler))
        if np.any(select):
            label = sampler if not label_set[k] else None
            ax1.scatter(
                [i], summary['N_like'][select], color=color,
                marker=marker, label=label, alpha=0.7, lw=0, s=100)
            label_set[k] = True
            ax2.scatter([i], summary['efficiency'][select], color=color,
                        marker=marker, alpha=0.7, lw=0, s=100)

handles, labels = ax1.get_legend_handles_labels()
ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=6, frameon=False, handletextpad=0, columnspacing=0.8,
           borderpad=0)
ax3.axis('off')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylabel('Likelihood Calls')
ax2.set_ylabel('Efficiency')
ax1.set_xticks(np.arange(len(label_list)))
ax2.set_xticks(np.arange(len(label_list)))
ax1.set_xticklabels(label_list, rotation=45)
ax2.set_xticklabels(label_list, rotation=45)
ax2.set_xticks([], minor=True)
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0)
plt.savefig(os.path.join('plots', 'performance.pdf'))
plt.savefig(os.path.join('plots', 'performance.png'), dpi=300)
plt.close()
