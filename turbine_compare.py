#!/usr/bin/env python3
"""
turbine_compare.py: comparison of the eight candidate turbines (Section A6, Tables 4 and 5).

Each turbine sits at its minimum hub height (D/2 + 27 m), with the 10 m Weibull fit of wind.py
(k = 2.037, c = 5.865 m/s) scaled by the power law (alpha = 0.16). Prints Table 5 and saves the
figures and a results table in the folder turbine_results.

Quick start
-----------
    python turbine_compare.py                       # all turbines in TURBINES, each at its own hub height
    python turbine_compare.py --hub-height 145      # force one hub height for every turbine (like-for-like)
    python turbine_compare.py --select "V236" "Haliade"      # only turbines whose name contains these
    python turbine_compare.py --model cp            # Cp-limited power curve instead of datasheet curve
    python turbine_compare.py --no-show             # save the files only, no window
    python turbine_compare.py --turbines my_turbines.json    # use your own list (format below)
    python turbine_compare.py --availability 0.96 --wake 0.10 --electrical 0.02   # net output too


"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

RHO0 = 1.225  # kg/m3, standard air density used for the Cp model


K_10M = 2.037
C_10M = 5.865
Z_REF = 10.0        # anemometer height (m)
ALPHA = 0.16        # Hellmann exponent adopted in report Section A5

# Minimum hub height rule (report Section A5: 145 m gives ~22 m tip clearance above high water for 236 m rotor)
AIR_GAP = 22.0      # required clearance of blade tip above highest water (m)
TIDE_OFFSET = 5.0   # highest water above mean sea level (m)


@dataclass
class Turbine:
    name: str
    rated_mw: float
    rotor_d: float
    cut_in: float
    rated_speed: float
    cut_out: float
    hub_height: Optional[float] = None
    color: Optional[str] = None
    linestyle: str = "-"
    note: str = ""
    curve: Optional[dict] = None   # {"speed": [...], "power_mw": [...]}

    # ---- geometry --------------------------------------------------------
    @property
    def area(self) -> float:
        return math.pi * self.rotor_d ** 2 / 4.0

    @property
    def specific_power(self) -> float:
        """W per m2 of swept area."""
        return self.rated_mw * 1e6 / self.area

    def min_hub_height(self, air_gap: float = AIR_GAP, tide_offset: float = TIDE_OFFSET) -> float:
        return self.rotor_d / 2.0 + air_gap + tide_offset

    def hub(self, override: Optional[float] = None) -> float:
        """Hub height actually used: override > own hub_height > minimum feasible."""
        if override is not None:
            return float(override)
        return float(self.hub_height) if self.hub_height is not None else self.min_hub_height()

    # ---- power curve -----------------------------------------------------
    def power(self, v, model: str = "datasheet", cp: float = 0.45, rho: float = RHO0):
        """Electrical power (MW) at hub-height wind speed v (m/s); v may be scalar or array."""
        v = np.asarray(v, dtype=float)
        if self.curve is not None:
            p = np.interp(v, self.curve["speed"], self.curve["power_mw"], left=0.0, right=0.0)
            return np.where(v > self.cut_out, 0.0, p) if self.cut_out else p
        if model == "cp":
            p = np.minimum(self.rated_mw, 0.5 * rho * self.area * cp * v ** 3 / 1e6)
            p = np.where(v >= self.cut_in, p, 0.0)
            return np.where(v > self.cut_out, 0.0, p)
        if model != "datasheet":
            raise ValueError(f"unknown model {model!r} (use 'datasheet' or 'cp')")
        vin, vr, vo = self.cut_in, self.rated_speed, self.cut_out
        ramp = self.rated_mw * (v ** 3 - vin ** 3) / (vr ** 3 - vin ** 3)
        p = np.where((v >= vin) & (v < vr), ramp, 0.0)
        return np.where((v >= vr) & (v <= vo), self.rated_mw, p)


TURBINES = [
    #        name                 MW     D    vin   vr    vout  hub   colour     style
    Turbine("Vestas V236-15.0",   15.0, 236, 3.0, 11.1, 31.0, None, None, "-"),
    Turbine("GE Haliade-X 14.7",  14.7, 220, 3.0, 11.2, 34.0, None, None, "-"),
    Turbine("SG 14-236 DD",       14.0, 236, 4.0, 12.0, 32.0, None, None, "-"),
    Turbine("SG 11.0-200 DD",     11.0, 200, 4.0, 14.0, 32.0, None, None, "-"),
    Turbine("Vestas V164-10.0",   10.0, 164, 3.0, 12.5, 25.0, None, None, "-"),
    Turbine("SG 8.0-167 DD",       8.0, 167, 3.0, 12.0, 25.0, None, None, "--"),
    Turbine("Vestas V162-6.2",     6.2, 162, 3.0, 10.5, 24.0, None, None, "--",
            note="onshore platform, reference only"),
    Turbine("SG 3.4-132",         3.465, 132, 3.0, 10.3, 25.0, None, None, "--",
            note="onshore platform, reference only"),
]


# ----------------------------------------------------------------------------
# 3. WIND RESOURCE AT HEIGHT
# ----------------------------------------------------------------------------
@dataclass
class WindResource:
    k: float = K_10M
    c_ref: float = C_10M
    z_ref: float = Z_REF
    alpha: float = ALPHA

    def factor(self, z: float) -> float:
        """Speed multiplier from the reference height to height z."""
        return (z / self.z_ref) ** self.alpha

    def weibull_at(self, z: float):
        """(k, c) at height z. k is unchanged by the power law."""
        return self.k, self.c_ref * self.factor(z)

    def mean_at(self, z: float) -> float:
        k, c = self.weibull_at(z)
        return c * math.gamma(1.0 + 1.0 / k)

    def pdf(self, v, z: float):
        k, c = self.weibull_at(z)
        v = np.asarray(v, dtype=float)
        return (k / c) * (v / c) ** (k - 1.0) * np.exp(-((v / c) ** k))

    def prob_below(self, v0: float, z: float) -> float:
        k, c = self.weibull_at(z)
        return 1.0 - math.exp(-((v0 / c) ** k))

    def prob_above(self, v0: float, z: float) -> float:
        k, c = self.weibull_at(z)
        return math.exp(-((v0 / c) ** k))


# ----------------------------------------------------------------------------
# 4. ESTIMATED OUTPUT PER TURBINE
# ----------------------------------------------------------------------------
@dataclass
class Losses:
    """Fractional losses applied to gross energy. All zero = gross output."""
    availability: float = 0.0   # e.g. 0.04 means 96 % available
    wake: float = 0.0
    electrical: float = 0.0
    other: float = 0.0

    def net_factor(self) -> float:
        f = 1.0
        for x in (self.availability, self.wake, self.electrical, self.other):
            f *= (1.0 - x)
        return f


@dataclass
class Output:
    name: str
    rated_mw: float
    rotor_d: float
    hub_height: float
    tip_height: float
    tip_clearance_msl: float
    mean_speed: float
    weibull_k: float
    weibull_c: float
    specific_power: float
    power_at_mean_mw: float
    aep_gross_gwh: float
    cf_gross: float
    aep_net_gwh: float
    cf_net: float
    kwh_per_m2: float
    pct_time_below_cut_in: float
    pct_time_at_rated: float
    pct_energy_at_rated: float
    hours_cut_out: float

    def as_dict(self):
        return asdict(self)


def estimate_output(t: Turbine, wind: WindResource, hub_height: Optional[float] = None,
                    model: str = "datasheet", losses: Optional[Losses] = None,
                    cp: float = 0.45, dv: float = 0.01) -> Output:
    """Annual energy, capacity factor and related stats for ONE turbine at its hub-height wind."""
    z = t.hub(hub_height)
    k, c = wind.weibull_at(z)
    v = np.arange(0.0, 60.0 + dv, dv)
    f = wind.pdf(v, z)
    p = t.power(v, model=model, cp=cp)
    mean_p = float(_trapz(p * f, v))                      # MW, time-averaged
    gross = mean_p * 8760.0 / 1000.0                      # GWh/yr
    net = gross * (losses or Losses()).net_factor()
    at_rated = (v >= t.rated_speed) & (v <= t.cut_out)
    # speed at which Cp model/datasheet reaches rated (for the Cp model use the actual curve)
    if model == "cp" and t.curve is None:
        rated_v = (t.rated_mw * 1e6 / (0.5 * RHO0 * t.area * cp)) ** (1.0 / 3.0)
        at_rated = (v >= rated_v) & (v <= t.cut_out)
    e_rated = float(_trapz((p * f)[at_rated], v[at_rated]) / mean_p) if mean_p > 0 else 0.0
    return Output(
        name=t.name, rated_mw=t.rated_mw, rotor_d=t.rotor_d, hub_height=z,
        tip_height=z + t.rotor_d / 2.0, tip_clearance_msl=z - t.rotor_d / 2.0,
        mean_speed=wind.mean_at(z), weibull_k=k, weibull_c=c, specific_power=t.specific_power,
        power_at_mean_mw=float(t.power(wind.mean_at(z), model=model, cp=cp)),
        aep_gross_gwh=gross, cf_gross=mean_p / t.rated_mw,
        aep_net_gwh=net, cf_net=net * 1000.0 / 8760.0 / t.rated_mw,
        kwh_per_m2=gross * 1e6 / t.area,
        pct_time_below_cut_in=100.0 * wind.prob_below(t.cut_in, z),
        pct_time_at_rated=100.0 * float(_trapz(f[at_rated], v[at_rated])),
        pct_energy_at_rated=100.0 * e_rated,
        hours_cut_out=8760.0 * wind.prob_above(t.cut_out, z),
    )


def estimate_farm(t: Turbine, wind: WindResource, n_turbines: int, **kw) -> dict:
    """Farm-level totals for n identical turbines (use Losses(wake=...) for wake loss)."""
    o = estimate_output(t, wind, **kw)
    return {"turbine": t.name, "n": n_turbines, "capacity_mw": n_turbines * t.rated_mw,
            "aep_gross_gwh": n_turbines * o.aep_gross_gwh, "aep_net_gwh": n_turbines * o.aep_net_gwh,
            "cf_net": o.cf_net}


def turbines_to_match(outputs: Sequence[Output], reference: Output, n_reference: int = 6):
    """Number of each turbine needed to match n_reference of the reference turbine (gross GWh)."""
    target = n_reference * reference.aep_gross_gwh
    return {o.name: target / o.aep_gross_gwh for o in outputs}


# ----------------------------------------------------------------------------
# 5. GRAPH
# ----------------------------------------------------------------------------

_MATLAB_COLOURS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
                   "#4DBEEE", "#A2142F", "#404040"]
FONT_FAMILY = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]   
AX_COL = "#262626"   



def make_figures(turbines: Sequence[Turbine], wind: WindResource, hub_height: Optional[float] = None,
                 model: str = "datasheet", cp: float = 0.45, reference_height: Optional[float] = None,
                 v_max: float = 27.0, stem: str = "turbine", save_pdf: bool = True, show: bool = True):
    """Make SEPARATE figures, each saved as <stem>_<name>.png (and .pdf):
         1_power_curves      power vs hub-height wind speed, with the wind distribution behind
         2_relative_power    power / rated power (shows who reaches rated first)
         3_energy_per_bin    annual energy per 1 m/s wind speed bin
         4_aep_cf            annual energy and capacity factor per turbine (side-by-side bars)
         5_hub_sensitivity   annual energy against hub height (effect of the wind scaling)
         6_wind_by_height    Weibull distribution at 10 m and at the lowest/highest hub height used
    Each turbine's energy uses the Weibull at ITS OWN hub height. The grey wind distribution
    and dotted mean in figures 1-2 are for reference_height (default: first turbine's hub).
    The FIRST turbine in the list is drawn thicker as the focus/selected turbine."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")     # no window needed when only saving files
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ImportError:
        raise SystemExit("matplotlib is not installed. Run:  pip install matplotlib numpy")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "font.size": 10, "text.color": AX_COL,
        "axes.labelsize": 10, "axes.labelcolor": AX_COL, "axes.edgecolor": AX_COL,
        "axes.linewidth": 0.6, "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "grid.color": AX_COL, "grid.alpha": 0.15, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": False,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.color": AX_COL, "ytick.color": AX_COL, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.major.width": .6, "ytick.major.width": .6,
        "lines.linewidth": 1.5,
        "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": AX_COL,
        "legend.framealpha": 1.0, "legend.fontsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    })
    zref = reference_height or turbines[0].hub(hub_height)
    mean_ref = wind.mean_at(zref)
    v = np.arange(0.0, 40.001, 0.01)
    pmax = max(t.rated_mw for t in turbines)
    cols = [t.color or _MATLAB_COLOURS[i % len(_MATLAB_COLOURS)] for i, t in enumerate(turbines)]
    lss = [t.linestyle for t in turbines]
    lws = [2.2 if i == 0 else 1.5 for i in range(len(turbines))]
    hubs_differ = len({round(t.hub(hub_height)) for t in turbines}) > 1
    cutout_hidden = any(t.cut_out > v_max for t in turbines)
    figs = {}

    def label(t):
        txt = f"{t.name} ({t.rated_mw:g} MW)"
        return txt + (f", hub {t.hub(hub_height):.0f} m" if hubs_differ else "")

    def new(name, title, xlabel, ylabel, size=(7.4, 5.4)):
        fig, ax = plt.subplots(figsize=size)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        figs[name] = fig
        return fig, ax

    def legend_below(ax, ncol=2, offset=-0.17):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol,
                  columnspacing=2.0, handlelength=2.4, borderpad=0.6, edgecolor=AX_COL)

    def shade(ax, top_factor):
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
        ax.tick_params(right=False)
        pdf = 8760 * wind.pdf(v, zref)
        ax2.fill_between(v, pdf, color="#d9d9d9", alpha=.55, lw=0)
        ax2.set_ylim(0, pdf.max() * top_factor)
        ax2.set_ylabel(f"Hours per year per 1 m/s at {zref:.0f} m", color="#8a8a8a", fontsize=10, labelpad=6)
        ax2.tick_params(axis="y", colors="#8a8a8a", labelsize=9)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    def mean_line(ax, ypos=0.97, va="top"):
        ax.axvline(mean_ref, color="k", ls="--", lw=1)
        ax.text(mean_ref + .3, ypos, f"Mean {mean_ref:.2f} m/s at {zref:.0f} m", fontsize=9, color="k",
                transform=ax.get_xaxis_transform(), va=va)

    def note_cutout(ax):
        if cutout_hidden:
            ax.text(0.995, 0.985, f"Cut-out speeds above {v_max:g} m/s not shown", fontsize=7.5, style="italic",
                    color="#555", transform=ax.transAxes, ha="right", va="top")

    # 1. power curves
    fig, a = new("1_power_curves", "Power curves against the wind speed distribution",
                 "Hub-height wind speed (m/s)", "Electrical power (MW)")
    for t, col, lw in zip(turbines, cols, lws):
        a.plot(v, t.power(v, model=model, cp=cp), color=col, ls=lss[turbines.index(t)], lw=lw, label=label(t))
    mean_line(a)
    a.set_xlim(0, v_max); a.set_ylim(0, pmax * 1.12)
    a.xaxis.set_major_locator(MultipleLocator(5)); a.xaxis.set_minor_locator(MultipleLocator(1))
    shade(a, 2.4)
    note_cutout(a)
    legend_below(a)

    # 2. relative power
    fig, b = new("2_relative_power", "Power relative to rated power",
                 "Hub-height wind speed (m/s)", "$P\\,/\\,P_{rated}$")
    for t, col, lw in zip(turbines, cols, lws):
        b.plot(v, t.power(v, model=model, cp=cp) / t.rated_mw, color=col, ls=lss[turbines.index(t)], lw=lw, label=label(t))
    mean_line(b, ypos=0.03, va="bottom")
    b.set_xlim(0, v_max); b.set_ylim(0, 1.08)
    b.xaxis.set_major_locator(MultipleLocator(5)); b.xaxis.set_minor_locator(MultipleLocator(1))
    shade(b, 2.4)
    note_cutout(b)
    legend_below(b)

    # 3. energy per bin
    fig, d = new("3_energy_per_bin", "Where the annual energy is produced",
                 "Hub-height wind speed (m/s)", "Annual energy per 1 m/s bin (GWh/yr)")
    for t, col, lw in zip(turbines, cols, lws):
        z = t.hub(hub_height)
        d.plot(v, t.power(v, model=model, cp=cp) * wind.pdf(v, z) * 8760 / 1000,
               color=col, ls=lss[turbines.index(t)], lw=lw, label=label(t))
    mean_line(d)
    d.set_xlim(0, min(v_max, 26)); d.set_ylim(0, d.get_ylim()[1] * 1.1)
    d.xaxis.set_major_locator(MultipleLocator(5)); d.xaxis.set_minor_locator(MultipleLocator(1))
    legend_below(d)

    # 4. AEP and CF summary: two aligned horizontal bar panels, best first
    outs = [estimate_output(t, wind, hub_height=hub_height, model=model, cp=cp) for t in turbines]
    order = sorted(range(len(turbines)), key=lambda i: -outs[i].aep_gross_gwh)
    n = len(order)
    fig, (e, e2) = plt.subplots(1, 2, figsize=(8.4, 0.5 * n + 2.2), sharey=True,
                                gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.06},
                                layout="constrained")
    figs["4_aep_cf"] = fig
    ypos = np.arange(n)
    names = [turbines[i].name + (f"\n({turbines[i].rated_mw:g} MW)") for i in order]
    aep = [outs[i].aep_gross_gwh for i in order]
    cfs = [outs[i].cf_gross * 100 for i in order]
    ccols = ["#0072BD"] * n
    e.barh(ypos, aep, color=ccols, height=.62)
    e2.barh(ypos, cfs, color=ccols, height=.62)
    for yi, val in zip(ypos, aep):
        e.text(val + max(aep) * .015, yi, f"{val:.1f}", va="center", fontsize=9)
    for yi, val in zip(ypos, cfs):
        e2.text(val + 1, yi, f"{val:.0f}%", va="center", fontsize=9)
    e.set_yticks(ypos); e.set_yticklabels(names, fontsize=9)
    e.invert_yaxis()
    e.set_xlim(0, max(aep) * 1.15); e2.set_xlim(0, max(max(cfs) * 1.15, 10))
    e.set_xlabel("Gross annual energy per turbine (GWh/yr)")
    e2.set_xlabel("Capacity factor (%)")
    for ax_ in (e, e2):
        ax_.grid(axis="y", visible=False)

    # 5. hub-height sensitivity
    fig, h = new("5_hub_sensitivity", "Annual energy against hub height",
                 "Hub height (m)", "Gross annual energy per turbine (GWh/yr)")
    zs = np.arange(80, 201, 5.0)
    for t, col, lw in zip(turbines, cols, lws):
        h.plot(zs, [estimate_output(t, wind, hub_height=z, model=model, cp=cp).aep_gross_gwh for z in zs],
               color=col, ls=lss[turbines.index(t)], lw=lw, label=t.name)
        zt = t.hub(hub_height)
        h.plot([zt], [estimate_output(t, wind, hub_height=zt, model=model, cp=cp).aep_gross_gwh],
               "o", color=col, ms=6, mec="k", mew=.6, zorder=5)
    h.set_xlim(80, 200); h.set_ylim(bottom=0)
    h.text(0.99, 0.03, "Markers show the hub height used", transform=h.transAxes, ha="right",
           fontsize=8, color="#555", style="italic")
    legend_below(h, ncol=3, offset=-0.15)

    # 6. wind distribution by height (10 m, lowest and highest hub height used)
    fig, w = new("6_wind_by_height", "Wind speed distribution at 10 m and at hub height",
                 "Wind speed (m/s)", "Hours per year per 1 m/s")
    vv = np.arange(0, 30.001, 0.05)
    hz = sorted({round(t.hub(hub_height)) for t in turbines})
    heights = [wind.z_ref] + ([hz[0], hz[-1]] if len(hz) > 1 else hz)
    pal = _MATLAB_COLOURS[:3]
    for z, col in zip(heights, pal):
        k_, c_ = wind.weibull_at(z)
        w.plot(vv, 8760 * wind.pdf(vv, z), lw=1.8, color=col,
               label=f"{z:.0f} m: mean {wind.mean_at(z):.2f} m/s, c = {c_:.2f} m/s")
        w.axvline(wind.mean_at(z), color=col, ls="--", lw=1)
    w.set_xlim(0, 25); w.set_ylim(bottom=0)
    w.xaxis.set_major_locator(MultipleLocator(5)); w.xaxis.set_minor_locator(MultipleLocator(1))
    w.text(0.99, 0.97, f"k = {wind.k:.2f} at all heights\n(dashed lines = mean speed)", transform=w.transAxes,
           ha="right", va="top", fontsize=8, color="#555", style="italic")
    legend_below(w, ncol=1, offset=-0.15)

    for name, fig in figs.items():
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        fig.savefig(f"{stem}_{name}.png", dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(f"{stem}_{name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        for fig in figs.values():
            plt.close(fig)
    return figs


# ----------------------------------------------------------------------------
# 6. COMMAND LINE
# ----------------------------------------------------------------------------
def load_turbines(path: str):
    with open(path) as fh:
        raw = json.load(fh)
    return [Turbine(
        name=r["name"], rated_mw=r["rated_mw"], rotor_d=r["rotor_d"], cut_in=r["cut_in"],
        rated_speed=r.get("rated_speed", 0.0), cut_out=r["cut_out"], hub_height=r.get("hub_height"),
        color=r.get("color"), linestyle=r.get("linestyle", "-"), note=r.get("note", ""),
        curve=r.get("curve")) for r in raw]


def print_table(outs: Sequence[Output], match: dict, net: bool):
    hdr = (f"{'Turbine':<20}{'MW':>6}{'hub':>5}{'v_mean':>7}{'P@mean':>7}{'AEP':>7}{'CF%':>6}"
           f"{'kWh/m2':>8}{'<vin%':>6}{'>=vr%':>6}{'E@vr%':>6}{'N':>6}")
    print(hdr + ("   AEPnet  CFnet%" if net else ""))
    print("-" * len(hdr))
    for o in outs:
        line = (f"{o.name:<20}{o.rated_mw:>6.1f}{o.hub_height:>5.0f}{o.mean_speed:>7.2f}"
                f"{o.power_at_mean_mw:>7.2f}{o.aep_gross_gwh:>7.1f}{o.cf_gross*100:>6.1f}"
                f"{o.kwh_per_m2:>8.0f}{o.pct_time_below_cut_in:>6.1f}{o.pct_time_at_rated:>6.1f}"
                f"{o.pct_energy_at_rated:>6.1f}{match[o.name]:>6.1f}")
        if net:
            line += f"{o.aep_net_gwh:>9.1f}{o.cf_net*100:>8.1f}"
        print(line)
    print("\nAEP in GWh/yr per turbine (gross unless AEPnet shown); N = turbines needed to match "
          "6 of the first turbine.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turbines", help="JSON file with turbine list (overrides TURBINES)")
    ap.add_argument("--select", nargs="+", help="keep turbines whose name contains any of these")
    ap.add_argument("--hub-height", type=float, help="force this hub height (m) for every turbine")
    ap.add_argument("--model", choices=["datasheet", "cp"], default="datasheet")
    ap.add_argument("--cp", type=float, default=0.45, help="Cp for the cp model")
    ap.add_argument("--k", type=float, default=K_10M, help="Weibull k at 10 m")
    ap.add_argument("--c10", type=float, default=C_10M, help="Weibull c at 10 m (m/s)")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="Hellmann exponent")
    ap.add_argument("--availability", type=float, default=0.0, help="availability LOSS fraction, e.g. 0.04")
    ap.add_argument("--wake", type=float, default=0.0, help="wake loss fraction")
    ap.add_argument("--electrical", type=float, default=0.0, help="electrical loss fraction")
    ap.add_argument("--other", type=float, default=0.0, help="other loss fraction")
    ap.add_argument("--out", default=os.path.join("turbine_results", "turbine"), help="output file stem (no extension)")
    ap.add_argument("--no-show", action="store_true", help="save files only, do not open a window")
    args, _unknown = ap.parse_known_args(argv)   # ignores extra args added by Jupyter/Spyder
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    turbines = load_turbines(args.turbines) if args.turbines else TURBINES
    if args.select:
        turbines = [t for t in turbines if any(s.lower() in t.name.lower() for s in args.select)]
    if not turbines:
        raise SystemExit("No turbines selected.")

    wind = WindResource(k=args.k, c_ref=args.c10, alpha=args.alpha)
    losses = Losses(args.availability, args.wake, args.electrical, args.other)
    net = any(x > 0 for x in (args.availability, args.wake, args.electrical, args.other))

    outs = [estimate_output(t, wind, hub_height=args.hub_height, model=args.model,
                            losses=losses, cp=args.cp) for t in turbines]
    match = turbines_to_match(outs, outs[0])

    print(f"Wind at 10 m: k={wind.k}, c={wind.c_ref} m/s | power-law alpha={wind.alpha} | model={args.model}\n")
    print_table(outs, match, net)

    with open(args.out + "_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(outs[0].as_dict().keys()))
        w.writeheader()
        for o in outs:
            w.writerow(o.as_dict())
    make_figures(turbines, wind, hub_height=args.hub_height, model=args.model, cp=args.cp,
                 stem=args.out, show=not args.no_show)
    folder = os.path.abspath(os.path.dirname(args.out) or ".")
    print(f"\nSaved 6 separate figures ({os.path.basename(args.out)}_1_...png to _6_...png, plus .pdf) "
          f"and {os.path.basename(args.out)}_results.csv in:\n  {folder}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # save the results next to this script
    main()