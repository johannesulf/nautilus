import os
import corner
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.table import Table, vstack

# %%

summary = []
posterior = {}


for folder in os.listdir('benchmarks'):

    directory = os.path.join('benchmarks', folder)

    if not os.path.isdir(directory) or folder == '__pycache__':
        continue

    results = vstack([Table.read(os.path.join(directory, fname)) for fname in
                      os.listdir(directory)])
    posterior[folder] = {}

    for i, sampler in enumerate(np.unique(results['sampler'])):
        print(folder, sampler, np.sum(results['sampler'] == sampler))

        select = results['sampler'] == sampler
        summary_row = {}
        summary_row['problem'] = folder
        summary_row['sampler'] = sampler
        if sampler == 'emcee':
            summary_row['log Z'] = np.nan
            summary_row['log Z error'] = np.nan
        else:
            summary_row['log Z'] = np.nanmean(results['log Z'][select])
            if np.sum(select) > 1:
                summary_row['log Z error'] = np.nanstd(
                    results['log Z'][select], ddof=1)
            else:
                summary_row['log Z error'] = np.nan
        summary_row['N_like'] = np.mean(results['N_like'][select])
        summary_row['efficiency'] = np.mean(
            (results['N_eff'] / results['N_like'])[select])

        summary.append(summary_row)
        posterior[folder][sampler] = []

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
            posterior[folder][sampler].append((x, pdf))

        x_bins = np.linspace(0, 1, 1001)
        x = 0.5 * (x_bins[1:] + x_bins[:-1])
        axarr[i].set_xlim(np.amin(x[pdf_all > np.amax(pdf_all) * 1e-3]),
                          np.amax(x[pdf_all > np.amax(pdf_all) * 1e-3]))
        axarr[i].set_ylim(ymin=0)
        if folder == 'cosmology':
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

problem_list = ['cosmology', 'galaxy', 'exoplanet']
sampler_list = ['dynesty-r', 'pocoMC', 'nautilus']
color_list = ['orange', 'royalblue', 'purple']
bar_width = 0.2

for statistic in ['N_like', 'efficiency']:
    for i, problem in enumerate(problem_list):
        for k, (sampler, color) in enumerate(zip(sampler_list, color_list)):
            y = summary[(summary['problem'] == problem) &
                        (summary['sampler'] == sampler)][statistic][0]
            if sampler == 'dynesty-r':
                label = 'dynesty'
            else:
                label = sampler
            plt.bar(i + k * bar_width, y, color=color, width=bar_width,
                    label=label if i == 0 else None, edgecolor='white')

    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(ymax=ymax*1.2)
    plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=2,
               handletextpad=0.4, columnspacing=0.8, borderpad=0,
               labelcolor='white')
    plt.xticks([r + bar_width for r in range(len(problem_list))], problem_list)
    if statistic == 'N_like':
        plt.ylabel(r'$N_{\rm like} / 10^5$', fontsize=14)
        plt.yticks([0, 1e5, 2e5, 3e5, 4e5, 5e5],
                   ['0', '1', '2', '3', '4', '5'])
    else:
        plt.ylabel(r'$N_{\rm eff} / N_{\rm like}$', fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.minorticks_off()
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join('plots', 'spotlight_{}.png'.format(
        statistic.lower())), dpi=300, transparent=True)
    plt.close()

# %%

problem = 'loggamma-30'
sampler_list = ['dynesty-s', 'pocoMC', 'nautilus-r']
color_list = ['orange', 'royalblue', 'purple']

for i, (sampler, color) in enumerate(zip(sampler_list, color_list)):
    k = np.arange(len(summary))[
        (summary['sampler'] == sampler) & (summary['problem'] == problem)][0]
    if sampler == 'nautilus-r':
        label = 'nautilus'
    elif sampler == 'dynesty-s':
        label = 'dynesty'
    else:
        label = sampler
    plotline, caps, barlinecols = plt.errorbar(
        i, summary['log Z'][k], yerr=summary['log Z error'][k], label=label,
        color=color, fmt='o', zorder=1, ms=8, lw=3)
    plt.setp(barlinecols[0], capstyle='round')

plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=3,
           handletextpad=-0.2, columnspacing=0.2, borderpad=0,
           labelcolor='white')
plt.axhline(0, ls='--', color='white', zorder=0)
plt.ylabel(r'Evidence $\ln \mathcal{Z} / \mathcal{Z}_{\rm true}$', fontsize=14)
plt.yticks([-1.5, -1.0, -0.5, 0.0, +0.5])
plt.xlim(-1, len(sampler_list))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.gca().tick_params(axis='y', colors='white', which='both')
plt.gca().yaxis.set_ticks_position('left')
plt.gca().spines['left'].set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.tight_layout(pad=0.0)
plt.savefig(os.path.join('plots', 'spotlight_evidence.png'), dpi=300,
            transparent=True)
plt.close()

# %%

for problem in np.unique(summary['problem']):
    select = summary['problem'] == problem
    if np.all(np.isnan(summary['log Z error'][select])):
        continue
    log_z_min = np.nanmin(
        (summary['log Z'] - 4 * summary['log Z error'])[select])
    log_z_max = np.nanmax(
        (summary['log Z'] + 4 * summary['log Z error'])[select])
    log_z = np.linspace(log_z_min, log_z_max, 100000)
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

problem_list = ['loggamma-30', 'funnel-20', 'rosenbrock-10', 'cosmology',
                'galaxy', 'exoplanet']
label_list = [r'LogGamma$_{30}$', r'Funnel$_{20}$', r'Rosenbrock$_{10}$',
              'Cosmology', 'Galaxy', 'Exoplanet']
sampler_list = ['nautilus', 'UltraNest', 'dynesty-u', 'dynesty-r', 'dynesty-s',
                'pocoMC']
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
            label = sampler if (not label_set[k] and i > 0) else None
            ax1.scatter(
                [i], summary['N_like'][select], color=color,
                marker=marker, label=label, alpha=0.7, lw=0, s=100)
            label_set[k] = i > 0
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
ax1.set_xticks([], minor=True)
ax2.set_xticks([], minor=True)
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0)
plt.savefig(os.path.join('plots', 'performance.pdf'))
plt.savefig(os.path.join('plots', 'performance.png'), dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'dynesty-r', 'pocoMC']
color_list = ['purple', 'orange', 'royalblue']
for sampler, color in zip(sampler_list, color_list):
    plt.plot((posterior['funnel-20'][sampler][0][0] - 0.5) * 20,
             posterior['funnel-20'][sampler][0][1] / 20, color=color,
             label=sampler)
x = np.linspace(-5, +5, 10000)
plt.plot(x, norm.pdf(x), color='black', ls='--', label='analytic')
plt.xlim(-3.5, +3.5)
plt.ylim(ymin=0)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$p(x_1)$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join('plots', 'funnel-20_x1_posterior.pdf'))
plt.savefig(os.path.join('plots', 'funnel-20_x1_posterior.png'), dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'dynesty-r', 'dynesty-s', 'pocoMC', 'emcee']
color_list = ['purple', 'orange', 'orange', 'royalblue', 'grey']
ls_list = ['-', '-', '--', '-', '-', '-']
for sampler, color, ls in zip(sampler_list, color_list, ls_list):
    plt.plot((posterior['rosenbrock-10'][sampler][7][0] - 0.5) * 10,
             posterior['rosenbrock-10'][sampler][7][1] / 10, color=color,
             label=sampler, ls=ls)
plt.xlim(-0.5, +1.75)
plt.ylim(ymin=0)
plt.xlabel(r'$x_8$')
plt.ylabel(r'$p(x_8)$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join('plots', 'rosenbrock-10_x8_posterior.pdf'))
plt.savefig(os.path.join('plots', 'rosenbrock-10_x8_posterior.png'), dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'dynesty-r', 'pocoMC']
color_list = ['purple', 'orange', 'royalblue']
for sampler, color in zip(sampler_list, color_list):
    plt.plot((posterior['funnel-20'][sampler][0][0] - 0.5) * 20,
             posterior['funnel-20'][sampler][0][1] / 20, color=color,
             label=sampler)
x = np.linspace(-5, +5, 10000)
plt.plot(x, norm.pdf(x), color='black', ls='--', label='analytic')
plt.xlim(-3.5, +3.5)
plt.ylim(ymin=0)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$p(x_1)$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join('plots', 'funnel-20_x1_posterior.pdf'))
plt.savefig(os.path.join('plots', 'funnel-20_x1_posterior.png'), dpi=300)
plt.close()

# %%

table_tex = []
problem_list = ['loggamma-30', 'funnel-20', 'rosenbrock-10', 'cosmology',
                'galaxy', 'exoplanet']
template = r'${med:+.{p}f} \pm {err:.{p}f}$'
sampler_list = ['nautilus', 'nautilus-r', 'dynesty-u', 'dynesty-r',
                'dynesty-s', 'pocoMC', 'UltraNest']
for sampler in sampler_list:
    table_tex_row = dict(sampler=sampler)
    for problem in problem_list:
        select = ((summary['problem'] == problem) &
                  (summary['sampler'] == sampler))
        if np.sum(select) == 1:
            log_z = summary[select]['log Z'][0]
            log_z_error = summary[select]['log Z error'][0]
            table_tex_row[problem] = template.format(
                med=log_z, err=log_z_error, p=3)
        else:
            table_tex_row[problem] = r'--'

    table_tex.append(table_tex_row)

table_tex = Table(table_tex)
table_tex.write('evidence.tex', overwrite=True)

# %%

fig, axes = plt.subplots(5, 5, figsize=(7, 7))
table = Table.read(os.path.join(
    'benchmarks', 'rosenbrock-10_emcee_posterior.hdf5'))
table['weights'] /= np.sum(table['weights'])
corner.corner(
    (table['points'][:, 1::2] - 0.5) * 10, weights=table['weights'], bins=70,
    plot_datapoints=False, plot_density=False, no_fill_contours=True,
    levels=(0.68, 0.95, 0.997), range=np.ones(5) * 0.999999, color='grey',
    contour_kwargs=dict(linewidths=1), fig=fig)
table = Table.read(os.path.join(
    'benchmarks', 'rosenbrock-10_nautilus-10000_posterior.hdf5'))
table['weights'] /= np.sum(table['weights'])
corner.corner(
    (table['points'][:, 1::2] - 0.5) * 10, weights=table['weights'], bins=70,
    labels=np.array([r'$x_{{{}}}$'.format(i) for i in range(1, 11)])[1::2],
    plot_datapoints=False, plot_density=False, fill_contours=True,
    levels=(0.68, 0.95, 0.997), range=np.ones(5) * 0.999999, color='purple',
    contour_kwargs=dict(linewidths=0), fig=fig)
axes[0, -1].text(0.0, 0.5, 'emcee', ha='left', va='bottom',
                 transform=axes[0, -1].transAxes, color='grey', fontsize=14)
axes[0, -1].text(0.0, 0.5, 'nautilus', ha='left', va='top',
                 transform=axes[0, -1].transAxes, color='purple', fontsize=14)
# fix ranges not lining up between 1d and 2d histograms, corner bug?
for i in range(4):
    axes[i, i].set_xlim(axes[i + 1, i].get_xlim())
axes[4, 4].set_xlim(axes[4, 0].get_ylim())
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(os.path.join('plots', 'rosenbrock-10_full_posterior.pdf'))
plt.savefig(os.path.join('plots', 'rosenbrock-10_full_posterior.png'), dpi=300)
plt.close()
