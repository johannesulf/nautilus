import time
from shutil import get_terminal_size

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
console = Console(width=60)


def _key_value_table(rows):
    table = Table.grid()
    table.add_column(width=20)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _status_text(value, target, minimum=True, fmt=".3f"):
    if (value >= target and minimum) or (value < target and not minimum):
        style = "green"
    else:
        style = "yellow"
    text = Text(f"{value:{fmt}}", style=style)
    text.append(f" (Target: {target:{fmt}})", style="dim")
    return text


def _run_properties(sampler):
    log_z = sampler.log_z
    n_eff = sampler.n_eff
    log_z_err = 1.0 / np.sqrt(n_eff) if n_eff > 2 else np.inf
    n_like = sampler.n_like
    f_live_target = sampler.f_live_target
    n_eff_target = sampler.n_eff_target

    rows = []
    rows.append(("Evidence (log Z)", f"{log_z:.3f} +/- {log_z_err:.3f}"))
    rows.append(("Likelihood Calls", f"{n_like}"))
    rows.append(("Eff. Sample Size", _status_text(
        n_eff, n_eff_target, fmt='.0f')))
    if not sampler.explored:
        f_live = sampler.f_live
        rows.append(("Live Fraction", _status_text(
            f_live, f_live_target, minimum=False)))
        rows.append(("Exploration", _progress_bar(
            sampler.exploration_progress)))
    return Group(
        Rule("[bold]Run Properties[/bold]", align="left", style="dim"),
        _key_value_table(rows))


def _current_bound(sampler, empty=False):

    rows = []
    rows.append(("Volume (log V)", f"{sampler.bounds[-1].log_v:.3f}" if not
                 empty else "TBD"))
    rows.append(("Threshold (log L)", f"{sampler.shell_log_l_min[-1]:.3f}" if
                 not empty else "TBD"))
    rows.append(
        ("Replacements", f"{sampler.n_update_iter}/{sampler.n_update}"))
    if empty or sampler.n_like_iter == 0:
        rows.append(("Efficiency", "TBD"))
    elif len(sampler.bounds) == 1:
        rows.append(("Efficiency", "100%"))
    else:
        rows.append(("Efficiency",
                     f"{sampler.n_update_iter / sampler.n_like_iter:.0%}"))

    return Group(
        Rule("[bold]Current Bound[/bold]", align="left", style="dim"),
        _key_value_table(rows))


def _histogram(x, bins=40):
    hist = np.histogram(x, bins=np.linspace(0, 1, bins - 1))[0]
    return Text("|" + ("".join(np.where(hist > 0, "\u2588", " ")) + "|"))


def _live_set(sampler):
    live_set = sampler.live_set
    rows = []
    for i, x in enumerate(live_set.T):
        rows.append((f"Parameter {i + 1}", _histogram(x)))
    return Group(
        Rule("[bold]Live Set Range[/bold]", align="left", style="dim"),
        _key_value_table(rows))


def _progress_bar(f, width=40):

    a = round(f * (width - 2))
    b = width - 2 - a

    return Text("|" + "\u2588" * a + " " * b + '|')


def _create_panel(sampler, status):
    t_start = time.time()  # TODO: remove once finalized
    content = []
    content.append(Text())

    if not sampler.explored:
        i = len(sampler.bounds)
        if status == "Adding Bound":
            i = i + 1
        content.append(Rule(f"[bold]Exploration Phase: Bound {i}[/bold]",
                            characters="═", style="magenta"))
    else:
        content.append(Rule("[bold]Sampling Phase[/bold]", characters="═",
                            style="magenta"))

    content.append(Text())
    content.append(Text(f"Status: {status}"))
    content.append(Text())
    content.append(_run_properties(sampler))

    if not sampler.explored:
        content.append(Text())
        content.append(_current_bound(
            sampler, empty=(status == "Adding Bound")))

    if not sampler.explored:
        content.append(Text())
        content.append(_live_set(sampler))

    t_end = time.time()
    content.append(Text())
    content.append(Text(f"Time for Panel: {(t_end - t_start) * 1000:.2f} ms"))

    return Group(*content)


def _print_status_classic(sampler, status='', header=False):
    """Print current summary statistics.

    Parameters
    ----------
    sampler : nautilus.sampler.Sampler
        Sampler object.
    status: string, optional
        Status of the sampler to be printed. Default is ``''``.
    header : bool, optional
        If ``True``, print a static header. Default is ``False``.

    """
    if status == "Finished Exploration":
        return

    if status == "Adding Bound":
        status = "Bounding"
    else:
        status = status.split(" ")[0]

    if status in ["Aborted", "Finished"]:
        end = '\n'
    else:
        end = '\r'

    if header:
        data = ['Status', 'Bounds', 'Ellipses', 'Networks', 'Calls',
                'f_live', 'N_eff', 'log Z']
    else:
        data = [status, len(sampler.bounds)]
        if len(sampler.bounds) > 1:
            data.extend([sampler.bounds[-1].n_ell, sampler.bounds[-1].n_net])
        else:
            data.extend([0, 0])
        data.extend([sampler.n_like, sampler.f_live, sampler.n_eff,
                     sampler.log_z])

        fmt = ['{}', '{:d}', '{:d}', '{:d}', '{:d}', '{:.4f}', '{:.0f}',
               '{:+.2f}']
        for i in range(len(data)):
            data[i] = 'N/A' if data[i] is None else fmt[i].format(data[i])

    for i, length in enumerate([9, 6, 8, 8, 8, 6, 5, 7]):
        data[i] = '{:<{}}'.format(data[i], length)

    output = ' | '.join(data)
    width = get_terminal_size((80, 24)).columns
    output = output.ljust(width)[:width]
    print(output, end=end, flush=True)


class LivePanel:
    def __init__(self, verbose):
        """Initialize the live panel.

        Parameters
        ----------
        verbose : bool or {'default', 'log', 'full', 'classic'}, optional
            How status updates are written. If ``"default"``, the panel will
            contiously be overwritten, i.e., only the latest status panel will
            be shown. If ``"log"``, some intermediate results will be kept such
            as when a shell during the exploration phase is filled up.
            Furthermore, if ``"full"``, all panels will be shown without
            overwriting them. Finally, ``"classic"`` corresponds to the panel
            used in older versions of `nautilus`. If ``True``, will default to
            ``"default"``. If ``False``, no output will appear. Default is
            ``True``.

        """
        if verbose not in [True, False, "classic", "default", "log", "full"]:
            msg = f"Unkown `verbose` value '{verbose}'."
            raise ValueError(msg)

        if verbose is True:
            verbose = "default"

        self.verbose = verbose
        self.live = None

        if self.verbose == "classic":
            _print_status_classic(None, header=True)

    def update(self, sampler, status, final=False):
        """Update the status of he panel.

        Parameters
        ----------
        sampler : nautilus.sampler.Sampler
            Sampler object.
        status : string
            Status description.
        final : bool, optional
            Whether the panel should be kept. Only used if `verbose` is
            ``"log"``. Default is ``False``.

        """
        if not self.verbose:
            return

        if self.verbose == "classic":
            _print_status_classic(sampler, status)
            return

        panel = _create_panel(sampler, status)

        if self.verbose == "full":
            console.print(panel)
            return

        if self.live is None:
            self.live = Live(panel, console=console, refresh_per_second=10)
            self.live.start(refresh=True)
        else:
            self.live.update(panel)

        if final and self.verbose == "log":
            self.live.stop()
            self.live = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.stop()
            self.live = None
