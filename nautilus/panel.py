from rich.console import Console, Group
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
import numpy as np

console = Console(width=80)

def _key_value_table(rows):
    table = Table.grid()
    table.add_column(width=20)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _status_text(value, target, larger=True, fmt='.3f'):
    if (value > target and larger) or (value < target and not larger):
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
            f_live, f_live_target, larger=False)))
    return Group(
        Rule("Global Properties", align="left", style="dim"),
        _key_value_table(rows))


def _create_panel(sampler, status):
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

        if final:
            self.live.stop()
            self.live = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.stop()
            self.live = None
