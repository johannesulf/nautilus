import time

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(width=80)


def _key_value_table(rows):
    table = Table.grid()
    table.add_column(width=20)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _status_text(value, target, minimum=True, fmt=".3f"):
    if (value >= target and minimum) or (value < target and not minimum):
        style = "blue"
    else:
        style = "yellow"
    text = Text(f"{value:{fmt}}", style=style)
    text.append(f" (Target: {target:{fmt}})", style="dim")
    return text


def _global_properties(sampler):
    log_z = sampler.log_z
    n_eff = sampler.n_eff
    log_z_err = 1.0 / np.sqrt(n_eff) if n_eff > 2 else np.inf
    n_like = sampler.n_like
    f_live_target = sampler.f_live_target
    n_eff_target = sampler.n_eff_target

    rows = []
    rows.append(("Evidence log Z", f"{log_z:.3f} +/- {log_z_err:.3f}"))
    rows.append(("Likelihood Calls", f"{n_like}"))
    rows.append(("Eff. Sample Size", _status_text(
        n_eff, n_eff_target, fmt='.0f')))
    if not sampler.explored:
        f_live = sampler.f_live
        rows.append(("Live Fraction", _status_text(
            f_live, f_live_target, minimum=False)))
    return Group(
        Rule("Global Properties", align="left", style="dim"),
        _key_value_table(rows))


def _current_bound(sampler, empty=False):

    rows = []
    rows.append(("Volume log V", f"{sampler.bounds[-1].log_v:.3f}" if not
                 empty else "TBD"))
    rows.append(("Threshold log L", f"{sampler.shell_log_l_min[-1]:.3f}" if not
                 empty else "TBD"))
    rows.append(("Updates", _status_text(
        sampler.n_update_iter if not empty else 0, sampler.n_update,
        fmt="d")))
    if empty or sampler.n_like_iter == 0:
        rows.append(("Efficiency", "TBD"))
    elif len(sampler.bounds) == 1:
        rows.append(("Efficiency", "100%"))
    else:
        rows.append(("Efficiency",
                     f"{sampler.n_update_iter / sampler.n_like_iter:.0%}"))

    return Group(
        Rule("Current Bound", align="left", style="dim"),
        _key_value_table(rows))


def _histogram(x, bins=60):
    hist = np.histogram(x, bins=np.linspace(0, 1, bins + 1))[0]
    return Text(''.join(np.where(hist > 0, "\u2588", "\u2591")))


def _live_set(sampler):
    live_set = sampler.live_set
    rows = []
    #rows.append(("Parameter", "Range"))
    for i, x in enumerate(live_set.T):
        rows.append((f"theta_{i + 1}", _histogram(x)))
    return Group(
        Rule("Live Set Range", align="left", style="dim"),
        _key_value_table(rows))


def _create_panel(sampler, status):
    t_start = time.time()  # TODO: remove once finalized
    content = []
    content.append(Text())

    if not sampler.explored:
        i = len(sampler.bounds)
        if status == "Adding Bound":
            i = i + 1
        content.append(Rule(f"Exploration Phase: Bound {i}", characters="="))
    else:
        content.append(Rule("Sampling Phase", characters="="))

    content.append(Text())
    content.append(Text(f"Status: {status}"))
    content.append(Text())
    content.append(_global_properties(sampler))

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


class LivePanel:
    def __init__(self, verbose):
        self.verbose = verbose
        self.live = None

    def update(self, sampler, status, final=False):
        if not self.verbose:
            return
        panel = _create_panel(sampler, status)
        if self.live is None:
            self.live = Live(panel, console=console, refresh_per_second=10)
            self.live.start(refresh=True)
        else:
            self.live.update(panel)

        if False:
            self.live.stop()
            self.live = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.stop()
            self.live = None
