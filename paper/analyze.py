import argparse
import corner
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from astropy.table import Table, vstack
from likelihoods.analytic import loggamma_logpdf
from pathlib import Path
from scipy.stats import norm

# %%

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--full')
args = parser.parse_args()

# %%

summary = []
posterior = {}


for path in (Path('.') / 'results').iterdir():

    if not path.is_dir():
        continue

    likelihood = path.name

    results = vstack([Table.read(p) for p in path.glob('*/*')])
    posterior[likelihood] = {}

    for i, sampler in enumerate(np.unique(results['sampler'])):
        print(likelihood, sampler, np.sum(results['sampler'] == sampler))

        select = results['sampler'] == sampler
        if likelihood == 'loggamma-30':
            x_30_mean = [
                np.average(np.linspace(0, 1, 1001)[:-1] + 0.0005,
                           weights=row['x_29']) for row in results[select]]
            print('30 times scatter of mean for x_30: {:.3f}'.format(
                30 * np.std(x_30_mean)))
        summary_row = {}
        summary_row['likelihood'] = likelihood
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

        if (likelihood in ['funnel-20', 'loggamma-30'] and
                sampler == 'nautilus-r100k'):
            print('Mean log Z: {:.5f} +/- {:.5f}'.format(
                summary_row['log Z'], summary_row['log Z error'] /
                np.sqrt(np.sum(select))))

        if 'bmd' in results.colnames:
            summary_row['bmd'] = np.mean(results['bmd'][select])
            if np.sum(select) > 1:
                summary_row['bmd error'] = np.nanstd(
                    results['bmd'][select], ddof=1)
            else:
                summary_row['bmd error'] = np.nan
        else:
            summary_row['bmd'] = np.nan
            summary_row['bmd error'] = np.nan
        summary_row['N_like'] = np.mean(results['N_like'][select])
        summary_row['efficiency'] = np.mean(
            (results['N_eff'] / results['N_like'])[select])
        summary_row['n'] = np.sum(select)

        summary.append(summary_row)
        posterior[likelihood][sampler] = []

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
            kernel_size = n_bins // 70

            while kernel_size == 0 or 1000 % kernel_size != 0:
                kernel_size += 1

            x = np.mean(x.reshape(1000 // kernel_size, kernel_size),
                        axis=-1)
            pdf = np.mean(pdf.reshape(1000 // kernel_size, kernel_size),
                          axis=-1)

            lines.append(axarr[i].plot(x, pdf, alpha=0.5)[0])
            labels.append(sampler)
            posterior[likelihood][sampler].append((x, pdf))

        x_bins = np.linspace(0, 1, 1001)
        x = 0.5 * (x_bins[1:] + x_bins[:-1])
        axarr[i].set_xlim(np.amin(x[pdf_all > np.amax(pdf_all) * 1e-3]),
                          np.amax(x[pdf_all > np.amax(pdf_all) * 1e-3]))
        axarr[i].set_ylim(ymin=0)
        if likelihood == 'cosmology':
            text = [r'$\log M_{\rm min}$', r'$\sigma_{\log M}$', r'$\log M_0$',
                    r'$\log M_1$', r'$\alpha$', r'$A_{\rm cen}$',
                    r'$A_{\rm sat}$'][i]
        else:
            text = r'$\theta_{{{}}}$'.format(i + 1)

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
    plt.savefig(Path('.') / 'figures' / '{}_posterior.pdf'.format(likelihood))
    plt.savefig(Path('.') / 'figures' / '{}_posterior.png'.format(likelihood),
                dpi=300)
    plt.close()

summary = Table(summary)
summary['sampler'] = np.where(summary['sampler'] == 'UltraNest', 'UltraNest-m',
                              summary['sampler'])
path = Path('.') / 'figures'

# %%


summary['log N_like'] = np.log(summary['N_like'])
likelihood_list = ['loggamma-30', 'funnel-20', 'rosenbrock-10', 'cosmology',
                   'exoplanet']

versions = []
for sampler in np.unique(summary['sampler']):
    if re.match(r'^nautilus-[0-9]+\.[0-9]+\.[0-9]+$', sampler) is not None:
        versions.append(sampler)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if len(versions) > 0:

    fig, axarr = plt.subplots(figsize=(7, 7), nrows=3, ncols=2, sharex=True,
                              sharey='row')

    for i, statistic in enumerate(['log N_like', 'log Z', 'bmd']):
        for version, color in zip(versions, colors):
            for k, suffix in enumerate(['', '-r']):
                for x, likelihood in enumerate(likelihood_list):
                    i_publ = np.arange(len(summary))[
                        (summary['sampler'] == 'nautilus' + suffix) &
                        (summary['likelihood'] == likelihood)][0]
                    i_vers = np.arange(len(summary))[
                        (summary['sampler'] == version + suffix) &
                        (summary['likelihood'] == likelihood)][0]
                    y_publ = summary[statistic][i_publ]
                    y_vers = summary[statistic][i_vers]
                    if statistic == 'log N_like':
                        y_publ_err = 0
                        y_vers_err = 0
                    else:
                        y_publ_err = (
                            summary[statistic + ' error'][i_publ] /
                            np.sqrt(summary['n'][i_publ]))
                        y_vers_err = (
                            summary[statistic + ' error'][i_vers] /
                            np.sqrt(summary['n'][i_vers]))
                    plotline, cap, barlinecols = axarr[i, k].errorbar(
                        x, y_vers - y_publ, color=color,
                        yerr=np.sqrt(y_publ_err**2 + y_vers_err**2),
                        marker='o')
                    plt.setp(barlinecols[0], capstyle='round',
                             label=version if x == 0 else None)

    axarr[0, 0].legend(frameon=False)

    for i in range(len(axarr)):
        ymin, ymax = axarr[i, 0].get_ylim()
        ymax = max(np.abs(ymin), np.abs(ymax))
        ymin = - ymax
        axarr[i, 0].set_ylim(ymin, ymax)
        axarr[i, 0].axhline(0, ls='--', color='black')
        axarr[i, 1].axhline(0, ls='--', color='black')

    axarr[0, 0].set_ylabel(r'$\Delta \log N_{\rm like}$')
    axarr[1, 0].set_ylabel(r'$\Delta \log \mathcal{Z}$')
    axarr[2, 0].set_ylabel(r'$\Delta {\rm BMD}$')
    axarr[0, 0].set_title('Without Resampling')
    axarr[0, 1].set_title('With Resampling')
    axarr[2, 0].set_xticks(np.arange(len(likelihood_list)))
    axarr[2, 1].set_xticks(np.arange(len(likelihood_list)))
    axarr[2, 0].set_xticklabels(likelihood_list, rotation=45)
    axarr[2, 1].set_xticklabels(likelihood_list, rotation=45)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path / 'new_versions.pdf')
    plt.savefig(path / 'new_versions.png', dpi=300)
    plt.close()

    if not args.full:
        print("Only performed version comparsion. To run full analysis " +
              "use 'python analyze.py --full'")
        sys.exit()

# %%

key_list = ['log Z', 'bmd']
label_list = [r'Evidence $\log \mathcal{Z}$',
              r'Bayesian Model Dimensionality $d$']
name_list = ['evidence', 'bmd']

for key, label, name in zip(key_list, label_list, name_list):
    for likelihood in np.unique(summary['likelihood']):
        select = summary['likelihood'] == likelihood
        if np.all(np.isnan(summary['{} error'.format(key)][select])):
            continue
        x_min = np.nanmin(
            (summary[key] - 4 * summary['{} error'.format(key)])[select])
        x_max = np.nanmax(
            (summary[key] + 4 * summary['{} error'.format(key)])[select])
        x = np.linspace(x_min, x_max, 100000)
        for row in summary[select]:
            if np.isnan(row['{} error'.format(key)]):
                continue
            plt.plot(x, np.exp(
                -0.5 * ((x - row[key]) / row['{} error'.format(key)])**2),
                label=row['sampler'])
        plt.legend(loc='upper center', frameon=False, prop={'size': 9}, ncol=2)
        plt.xlabel(label)
        plt.xlim(x_min, x_max)
        plt.ylim(0, 1.5)
        plt.gca().set_yticks([])
        plt.gca().set_yticks([], minor=True)
        plt.tight_layout(pad=0.3)
        plt.savefig(path / '{}_{}.pdf'.format(likelihood, name))
        plt.savefig(path / '{}_{}.png'.format(likelihood, name), dpi=300)
        plt.close()

# %%

plt.figure(figsize=(7, 3.0))
ax1 = plt.subplot2grid((8, 2), (1, 0), rowspan=7)
ax2 = plt.subplot2grid((8, 2), (1, 1), rowspan=7)
ax3 = plt.subplot2grid((8, 2), (0, 0), colspan=2)

likelihood_list = ['loggamma-30', 'funnel-20', 'rosenbrock-10', 'cosmology',
                   'galaxy', 'exoplanet']
label_list = [r'LogGamma$_{30}$', r'Funnel$_{20}$', r'Rosenbrock$_{10}$',
              'Cosmology', 'Galaxy', 'Exoplanet']
sampler_list = ['nautilus', 'nautilus-r', 'UltraNest-m', 'dynesty-u',
                'dynesty-r', 'dynesty-s', 'pocoMC']
color_list = ['purple', 'purple', 'darkblue', 'orange', 'orange', 'orange',
              'royalblue']
ls_list = ['-', '--', '-', ':', '-', '--', '-']


for k, (sampler, color, ls) in enumerate(
        zip(sampler_list, color_list, ls_list)):

    x = []
    n_like = []
    eff = []

    for i, likelihood in enumerate(likelihood_list):
        select = ((summary['likelihood'] == likelihood) &
                  (summary['sampler'] == sampler))
        if np.any(select):
            x.append(i)
            n_like.append(summary['N_like'][select][0])
            eff.append(summary['efficiency'][select][0])

    if len(x) > 0:
        ax1.plot(x, n_like, color=color, marker='o', label=sampler, ls=ls,
                 zorder=k)
        ax2.plot(x, eff, color=color, marker='o', ls=ls, zorder=k)

handles, labels = ax1.get_legend_handles_labels()
ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=len(sampler_list), frameon=False, handletextpad=0.3,
           columnspacing=0.8, borderpad=0, markerscale=0, handlelength=1.5)
ax3.axis('off')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_ylabel('Likelihood Evaluations')
ax2.set_ylabel('Sampling Efficiency')
ax2.yaxis.tick_right()
ax2.yaxis.set_ticks_position('both')
ax2.yaxis.set_label_position("right")
ax1.set_xticks(np.arange(len(label_list)))
ax2.set_xticks(np.arange(len(label_list)))
ax1.set_xticklabels(label_list, rotation=45)
ax2.set_xticklabels(label_list, rotation=45)
ax1.set_xticks([], minor=True)
ax2.set_xticks([], minor=True)
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0)
plt.savefig(path / 'performance.pdf')
plt.savefig(path / 'performance.png', dpi=300)
plt.close()

# %%

plt.figure(figsize=(7, 2.5))
ax1 = plt.subplot2grid((8, 2), (1, 0), rowspan=7)
ax2 = plt.subplot2grid((8, 2), (1, 1), rowspan=7)
ax3 = plt.subplot2grid((8, 2), (0, 0), colspan=2)
n_dim_list = [5, 10, 15, 20, 25, 30, 40, 50]


for k, (sampler, color, ls) in enumerate(
        zip(sampler_list, color_list, ls_list)):

    x = []
    n_like = []
    log_z = []
    log_z_err = []

    for i, n_dim in enumerate(n_dim_list):
        select = ((summary['likelihood'] == 'loggamma-{}'.format(n_dim)) &
                  (summary['sampler'] == sampler))
        if np.any(select):
            x.append(n_dim)
            n_like.append(summary['N_like'][select][0])
            log_z.append(summary['log Z'][select][0])
            log_z_err.append(summary['log Z error'][select][0])

    if len(x) > 0:
        ax1.plot(x, n_like, color=color, marker='o', label=sampler, ls=ls,
                 zorder=k)
        plotline, cap, barlinecols = ax2.errorbar(
            np.array(x) + (-(len(sampler_list) - 1) / 2 + k) * 0.25, log_z,
            yerr=log_z_err, color=color, marker='o', ls=ls, zorder=k)
        plt.setp(barlinecols[0], capstyle='round')

handles, labels = ax1.get_legend_handles_labels()
ax3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=len(sampler_list), frameon=False, handletextpad=0.3,
           columnspacing=0.8, borderpad=0, markerscale=0, handlelength=1.5)
ax3.axis('off')
ax2.set_ylim(-20, +20)
ax2.set_yscale('symlog', linthresh=0.1)
ax2.axhline(0, ls='--', color='black')
ax1.set_xlabel(r'Dimensionality $N_{\rm dim}$')
ax2.set_xlabel(r'Dimensionality $N_{\rm dim}$')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax1.set_yscale('log')
ax1.set_ylabel('Likelihood Evaluations')
ax2.set_ylabel(r'Evidence $\log \mathcal{Z}$')
ax1.set_xticks([], minor=True)
ax2.set_xticks([], minor=True)
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0, wspace=0.1)
plt.savefig(path / 'scaling.pdf')
plt.savefig(path / 'scaling.png', dpi=300)
plt.close()

# %%

for k, (sampler, color, ls) in enumerate(
        zip(sampler_list, color_list, ls_list)):

    x = []
    n_like = []

    for i, n_dim in enumerate(n_dim_list):
        select = ((summary['likelihood'] == 'loggamma-{}'.format(n_dim)) &
                  (summary['sampler'] == sampler))
        if np.any(select):
            x.append(n_dim)
            n_like.append(summary['N_like'][select][0])

    if len(x) > 0:
        plt.plot(x, n_like, color=color, marker='o', label=sampler, ls=ls,
                 zorder=k)

plt.legend(handles, labels, loc='upper left', ncol=2, frameon=False,
           handletextpad=0.3, columnspacing=0.8, markerscale=0,
           handlelength=1.5)
plt.xlabel(r'Dimensionality $N_{\rm dim}$')
plt.yscale('log')
plt.ylabel('Likelihood Evaluations')
y_min, y_max = plt.gca().get_ylim()
y_max = y_max * np.exp(0.4 * np.log(y_max / y_min))
plt.ylim(y_min, y_max)
plt.xticks(n_dim_list)
plt.tight_layout(pad=0.1)
plt.savefig(path / 'scaling_only.pdf')
plt.savefig(path / 'scaling_only.png', dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'nautilus-r', 'UltraNest-m', 'dynesty-u',
                'dynesty-r', 'dynesty-s', 'pocoMC']

for likelihood in likelihood_list:
    std = {}
    for sampler in sampler_list:
        try:
            p = posterior[likelihood][sampler]
            n_dim = len(p)
            std[sampler] = np.zeros(n_dim)
            for i in range(n_dim):
                std[sampler][i] = np.sqrt(
                    np.average(p[i][0]**2, weights=p[i][1]) -
                    np.average(p[i][0], weights=p[i][1])**2)
        except KeyError:
            pass
    for sampler in std.keys():
        plt.plot(std[sampler] / std['nautilus-r'], label=sampler)
    plt.xlabel('Parameter')
    plt.ylabel('Relative Uncertainty')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout(pad=0.3)
    plt.savefig(path / '{}_std.pdf'.format(likelihood))
    plt.savefig(path / '{}_std.png'.format(likelihood), dpi=300)
    plt.close()

# %%

x = np.linspace(-5, +5, 10000)
plt.plot(x, norm.pdf(x), color='grey', label='analytic', lw=2.5)
sampler_list = ['nautilus', 'dynesty-r', 'pocoMC']
color_list = ['purple', 'orange', 'royalblue']
ls_list = [(0, (2.5, 2.5)), (2.5, (2.5, 2.5)), '-']
for sampler, color, ls in zip(sampler_list, color_list, ls_list):
    plt.plot((posterior['funnel-20'][sampler][0][0] - 0.5) * 20,
             posterior['funnel-20'][sampler][0][1] / 20, color=color,
             label=sampler, ls=ls, lw=2.0)
plt.xlim(-4, +4)
plt.ylim(ymin=0)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$p(\theta_1)$')
plt.legend(loc='upper left', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(path / 'funnel-20_x1_posterior.pdf')
plt.savefig(path / 'funnel-20_x1_posterior.png', dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'nautilus-r', 'dynesty-r', 'dynesty-s', 'pocoMC']
color_list = ['purple', 'purple', 'orange', 'orange', 'royalblue']
ls_list = ['-', '--', '-', '--', '-']
f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.5, 1]},
                             sharex=True)

for sampler, color, ls in zip(sampler_list, color_list, ls_list):
    x = posterior['loggamma-30'][sampler][9][0]
    y = posterior['loggamma-30'][sampler][9][1]
    ax1.plot(x, y, color=color, label=sampler, ls=ls)
    x_bins = x - np.diff(x)[0] / 2
    x_bins = np.append(x_bins, 1.0)
    y_true = np.zeros_like(y)
    for i in range(len(y_true)):
        x_min = x_bins[i]
        x_max = x_bins[i + 1]
        y_true[i] = np.mean(np.exp(loggamma_logpdf(
            np.linspace(x_min, x_max, 100), 1.0, 2.0 / 3.0, 1.0 / 30.0)))
    mask = y_true > 0
    ax2.plot(x[mask], y[mask] / y_true[mask], color=color, ls=ls)
x = np.linspace(0, 1, 10000)
ax1.plot(x, np.exp(loggamma_logpdf(x, 1.0, 2.0 / 3.0, 1.0 / 30.0)),
         color='black', ls='--', label='analytic')
ax2.plot(x, np.ones_like(x), color='black', ls='--')
plt.xlim(0.4, 0.75)
ax1.set_ylim(ymin=0)
ax2.set_ylim(0, 1.3)
plt.xlabel(r'$\theta_{10}$')
ax1.set_ylabel(r'$p(\theta_{10})$')
ax2.set_ylabel(r'Ratio')
ax1.set_yticks([0, 5, 10])
ax1.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(hspace=0.05)
plt.savefig(path / 'loggamma-30_x10_posterior.pdf')
plt.savefig(path / 'loggamma-30_x10_posterior.png', dpi=300)
plt.close()

# %%

sampler_list = ['nautilus', 'dynesty-r', 'pocoMC']
color_list = ['purple', 'orange', 'royalblue']
for sampler, color in zip(sampler_list, color_list):
    x_bins = np.linspace(1e-6, 1 - 1e-6,
                         len(posterior['exoplanet'][sampler][4][0]) + 1)
    x_bins = norm(loc=np.log(5.05069163), scale=2).isf(1 - x_bins)
    dx = np.diff(x_bins)
    x = 0.5 * (x_bins[1:] + x_bins[:-1])
    y = posterior['exoplanet'][sampler][4][1] / dx
    y /= np.sum(y * dx)

    plt.plot(x, y, color=color, label=sampler)
plt.xlim(0.2, 2.2)
plt.ylim(ymin=0)
plt.xlabel(r'K2-24b Semi-Amplitude $\ln K [\mathrm{m} \, \mathrm{s}^{-1}]$')
plt.ylabel(r'$p(\ln K)$')
plt.legend(loc='best', frameon=False)
_ = plt.tight_layout(pad=0.3)
plt.savefig(path / 'exoplanet_x5_posterior.pdf')
plt.savefig(path / 'exoplanet_x5_posterior.png', dpi=300)
plt.close()


# %%

sampler_list = ['nautilus', 'UltraNest', 'dynesty-r', 'dynesty-s', 'pocoMC',
                'emcee']
color_list = ['purple', 'darkblue', 'orange', 'orange', 'royalblue', 'grey']
ls_list = ['-', '--', '-', '--', '-', '-', '-']
for sampler, color, ls in zip(sampler_list, color_list, ls_list):
    plt.plot((posterior['rosenbrock-10'][sampler][7][0] - 0.5) * 10,
             posterior['rosenbrock-10'][sampler][7][1] / 10, color=color,
             label=sampler if sampler != 'UltraNest' else 'UltraNest-m', ls=ls)
plt.xlim(-0.5, +1.75)
plt.ylim(ymin=0)
plt.xlabel(r'$\theta_8$')
plt.ylabel(r'$p(\theta_8)$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig(path / 'rosenbrock-10_x8_posterior.pdf')
plt.savefig(path / 'rosenbrock-10_x8_posterior.png', dpi=300)
plt.close()

# %%

likelihood_list = ['loggamma-30', 'funnel-20', 'rosenbrock-10', 'cosmology',
                   'galaxy', 'exoplanet']
sampler_list = ['nautilus', 'nautilus-r', 'dynesty-u', 'dynesty-r',
                'dynesty-s', 'pocoMC', 'UltraNest-m']

key_list = ['log Z', 'bmd']
name_list = ['evidence', 'bmd']
template_list = [r'${mean:+.3f} \pm {err:.3f}$', r'${mean:.2f} \pm {err:.2f}$']

for key, name, template in zip(key_list, name_list, template_list):
    table_tex = []
    for sampler in sampler_list:
        table_tex_row = dict(sampler=sampler)
        for likelihood in likelihood_list:
            select = ((summary['likelihood'] == likelihood) &
                      (summary['sampler'] == sampler))
            if np.sum(select) == 1:
                y = summary[select][key][0]
                y_error = summary[select]['{} error'.format(key)][0]
                table_tex_row[likelihood] = template.format(
                    mean=y, err=y_error)
            else:
                table_tex_row[likelihood] = r'--'

        table_tex.append(table_tex_row)

    table_tex = Table(table_tex)
    table_tex.write(Path('.') / 'draft' / '{}.tex'.format(name),
                    overwrite=True)

# %%

fig, axes = plt.subplots(5, 5, figsize=(7, 7))
table = Table.read(Path('.') / 'results' /
                   'rosenbrock-10_emcee_posterior.hdf5')
table['weights'] /= np.sum(table['weights'])
corner_range = 0.999999
corner.corner(
    (table['points'][:, 1::2] - 0.5) * 10, weights=table['weights'], bins=60,
    plot_datapoints=False, plot_density=False, no_fill_contours=True,
    levels=(0.68, 0.95, 0.997), range=np.ones(5) * corner_range, color='grey',
    contour_kwargs=dict(linewidths=1.5, alpha=0.5), fig=fig,
    hist_kwargs=dict(lw=1.5))
corner_range_abs = []
for i in range(5):
    x = (table['points'][:, 1::2][:, i] - 0.5) * 10
    corner_range_abs.append((np.percentile(x, 50 * (1 - corner_range)),
                             np.percentile(x, 100 - 50 * (1 - corner_range))))
table = Table.read(Path('.') / 'results' /
                   'rosenbrock-10_nautilus-10000_posterior.hdf5')
table['weights'] /= np.sum(table['weights'])
corner.corner(
    (table['points'][:, 1::2] - 0.5) * 10, weights=table['weights'], bins=60,
    labels=np.array([r'$\theta_{{{}}}$'.format(i)
                    for i in range(1, 11)])[1::2],
    plot_datapoints=False, plot_density=False, fill_contours=True,
    levels=(0.68, 0.95, 0.997), range=corner_range_abs, color='purple',
    contour_kwargs=dict(linewidths=0), fig=fig, labelpad=-0.1,
    hist_kwargs=dict(alpha=0.5, lw=1.5))
axes[0, -1].text(0.0, 0.5, 'emcee', ha='left', va='bottom',
                 transform=axes[0, -1].transAxes, color='grey', fontsize=14)
axes[0, -1].text(0.0, 0.5, 'nautilus', ha='left', va='top',
                 transform=axes[0, -1].transAxes, color='purple', fontsize=14)
# fix ranges not lining up between 1d and 2d histograms, corner bug?
for i in range(4):
    axes[i, i].set_xlim(axes[i + 1, i].get_xlim())
axes[4, 4].set_xlim(axes[4, 0].get_ylim())
for i in range(5):
    axes[i, i].set_ylim(ymin=0)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(path / 'rosenbrock-10_full_posterior.pdf')
plt.savefig(path / 'rosenbrock-10_full_posterior.png', dpi=300)
plt.close()
