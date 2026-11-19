"""
sensitivity.py: sensitivity analysis of the 270 MW design and the 180 MW farm (Section B4, Appendix B.4).

    python sensitivity.py      (needs grid_results/grid_layouts.csv from layout_grid.py, and zones.npz)

The layouts stay fixed (no re-optimising) and one input changes at a time from its base value in
wind.py or farm.py. Cases that change the wind or the hub height rerun the PyWake wake model,
because the energy changes; all other cases only re-cost the farm.

Prints, in the order of the report:
    Table 20   every case of Figure 12, both farms (B4)
    B4.1-B4.3  the LCOE at the published discount rate, turbine prices and cable costs
    Table 21   LCOE against hub height, with and without the taller tower (B4.4)
    Table 11   LCOE against operating life (B4.5)
Shows and saves, in grid_results:
    sens_tornado.png        Figure 12
    sens_discount_rate.png  Figure 13
    sens_turbine_price.png  Figure 14
    sens_cables.png         Figure 15
    sens_hub_height.png     Figure 16
"""

import os
import warnings
from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformSite
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

import farm
import wind
from wind import load, KNOT_TO_MS

warnings.filterwarnings("ignore")

# ================================================================ settings
OUT = "grid_results"
FARMS = {"270 MW": (18, "#2a78d6"), "180 MW": (12, "#eb6834")}       # turbines, colour
LONG_TERM_FACTOR = 5.42 / 5.20     # Camborne 1991-2020 mean 10.53 kn = 5.42 m/s; 2006 mean 5.20 m/s
HUBS = [145, 155, 165, 175, 185]   # m; 145 m is the lowest with 22 m tip clearance (Section A5)
TOWER_COST = 2.3e6                 # GBP per 15 MW tower (BVG Associates floating wind guide, 2025)
TOWER_BASE = 25.0                  # m above sea where the tower starts (transition piece)
TOWER_EXP = 1.67                   # tower mass and cost ~ height^1.67 (WindPACT)
LIVES = [20, 30, 35]               # operating lives for Table 11, years
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def crf(rate, life):
    return rate / (1 - (1 + rate) ** -life)


def found_scale(h):
    """Foundation cost against hub height x rotor area (Dicorato et al. 2011), relative to 145 m."""
    f = lambda hh: 1 + 0.8e-6 * (hh * (farm.D / 2) ** 2 - 1e5)
    return f(h) / f(farm.TURBINE["hub"])


def tower_extra_per_kw(h):
    """Extra tower cost against the 145 m base, GBP/kW (Appendix B.4)."""
    ratio = ((h - TOWER_BASE) / (farm.TURBINE["hub"] - TOWER_BASE)) ** TOWER_EXP
    return TOWER_COST * (ratio - 1) / farm.KW


def half_up(x, nd=1):
    """Round to 2 decimals as in the saved tables, then halves up to nd (112.449 -> 112.45 -> 112.5),
    which is how the report tables were rounded."""
    return float(Decimal(f"{x:.2f}").quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


# ================================================================ cases of Table 20 and Figure 12
# (group, label, {(module, name): value}, {energy settings: alpha, speed, hub})
F = farm
CASES = [
    ("Wind", "Long-term mean wind (2006 was 4% below 1991-2020)", {}, {"speed": LONG_TERM_FACTOR}),
    ("Wind", "Hellmann alpha 0.11 (unstable air, flat open coast)", {}, {"alpha": 0.11}),
    ("Wind", "Hellmann alpha 0.14", {}, {"alpha": 0.14}),
    ("Wind", "Hellmann alpha 0.20", {}, {"alpha": 0.20}),
    ("Energy", "Availability 92% (measured UK fleet 2022-25)", {(F, "LOSSES"): 0.92 * 0.97 * 0.98}, {}),
    ("Capex", "Turbine 783 GBP/kW (Nordex average selling price Q3 2026)", {(F, "TURBINE_PER_KW"): 783}, {}),
    ("Capex", "Turbine 1,040 GBP/kW (East Anglia TWO contract, 2024)", {(F, "TURBINE_PER_KW"): 1040}, {}),
    ("Capex", "Turbine 1,316 GBP/kW (Arup 2025)", {(F, "TURBINE_PER_KW"): 1316}, {}),
    ("Capex", "Foundation 585 GBP/kW (BVGA, 1 GW farm)", {(F, "FOUND_PER_KW"): 585}, {}),
    ("Capex", "Foundation 902 GBP/kW (Arup 2025 Table 23)", {(F, "FOUND_PER_KW"): 902}, {}),
    ("Capex", "DESNZ peer review: Arup lines -10%, development 162",
     {(F, "INSTALL_PER_KW"): 130 * 2540 / 2823, (F, "OTHER_PER_KW"): 649 * 2540 / 2823, (F, "DEV_PER_KW"): 162}, {}),
    ("Capex", "No 15% small farm uplift", {(F, "INSTALL_PER_KW"): 130 / 1.15, (F, "OTHER_PER_KW"): 649 / 1.15}, {}),
    ("Capex", "Depth slope 0 %/m", {(F, "FOUND_PER_M"): 0.0}, {}),
    ("Capex", "Depth slope 2.3 %/m (Danish Energy Agency)", {(F, "FOUND_PER_M"): 0.023}, {}),
    ("Capex", "Array cable 0.45M GBP/km (Danish Energy Agency)", {(F, "ARRAY_PER_KM"): 0.45e6}, {}),
    ("Capex", "Array cable 1.1M GBP/km (Inch Cape contract, 2024)", {(F, "ARRAY_PER_KM"): 1.1e6}, {}),
    ("Capex", "Export cable 0.84M GBP/km (Seagreen, Ofgem)", {(F, "EXPORT_PER_KM"): 0.84e6}, {}),
    ("Capex", "Export cable 2.0M GBP/km", {(F, "EXPORT_PER_KM"): 2.0e6}, {}),
    ("Capex", "Landfall 1.74M GBP per cable (low case)", {(F, "LANDFALL"): 1.74e6}, {}),
    ("Capex", "Landfall 4.73M GBP per cable (high case)", {(F, "LANDFALL"): 4.73e6}, {}),
    ("Opex", "O&M +15% small farm uplift", {(F, "OPEX_PER_KW"): 46.5 * 1.15 + 8.6 + 0.8}, {}),
    ("Opex", "Decommissioning 86 GBP/kW (Arup)", {(F, "DECOM_PER_KW"): 86}, {}),
    ("Opex", "Decommissioning 426 GBP/kW (BVGA)", {(F, "DECOM_PER_KW"): 426}, {}),
    ("Finance", "Arup finance basis: 6.2% over 35 years", {(F, "RATE"): 0.062, (F, "LIFE"): 35, (F, "CRF"): crf(0.062, 35)}, {}),
]
for h in HUBS[1:4]:
    CASES.append(("Hub height", f"Hub height {h} m (foundation scaled, tower not costed)",
                  {(F, "FOUND_PER_KW"): farm.FOUND_PER_KW * found_scale(h)}, {"hub": h}))


def figure_label(label):
    """Shorter wording of a case for Figure 12."""
    for a, b in [("GBP", "£"), ("Hellmann alpha", "Shear exponent α"), ("1991-2020", "1991 to 2020"),
                 ("2022-25", "2022 to 2025"), ("-10%", "−10%")]:
        label = label.replace(a, b)
    return label


@contextmanager
def changed(overrides):
    """Set some module constants for the duration of one case, then put them back."""
    old = {k: getattr(*k) for k in overrides}
    for (mod, name), val in overrides.items():
        setattr(mod, name, val)
    try:
        yield
    finally:
        for (mod, name), val in old.items():
            setattr(mod, name, val)


# ================================================================ model
def setup():
    """Layouts, sites and an energy function (PyWake, cached) for both farms."""
    df = load(farm.DATA_FILE)
    v10 = df["knots"].to_numpy(float) * KNOT_TO_MS
    wd = df["dir"].to_numpy(float)
    zones = np.load("zones.npz")
    depth_at, seabed_at = farm.depth_function(), farm.seabed_function(zones["all_buildable"])
    G = pd.read_csv(os.path.join(OUT, "grid_layouts.csv"))
    ws = np.arange(0, 35.01, 0.05)
    cache = {}

    def energy(n, xy, alpha=wind.ALPHA, speed=1.0, hub=None):
        """Annual energy before losses, MWh/yr."""
        hub = hub or farm.TURBINE["hub"]
        key = (n, alpha, speed, hub)
        if key not in cache:
            wt = WindTurbine(farm.TURBINE["name"], diameter=farm.D, hub_height=hub,
                             powerCtFunction=PowerCtTabular(ws, farm.power(ws), "w", farm.thrust(ws)))
            wfm = Bastankhah_PorteAgel_2014(UniformSite(ti=0.06), wt, k=farm.WAKE_K, use_effective_ws=False)
            cache[key] = float(np.sum(wfm.aep(xy[:, 0], xy[:, 1], wd=wd, ws=speed * v10 * (hub / 10) ** alpha,
                                              time=True))) * 1e3
        return cache[key]

    farms = {}
    for name, (n, col) in FARMS.items():
        xy = G[G.farm_turbines == n][["x_m", "y_m"]].to_numpy(float)
        farms[name] = dict(n=n, col=col, xy=xy, depth=depth_at(xy), seabed=seabed_at(xy))

    def lcoe(f, overrides=None, **e):
        """LCOE (GBP/MWh) of one farm with some inputs changed, Equation (8)."""
        with changed(overrides or {}):
            e_net = energy(f["n"], f["xy"], **e) * farm.LOSSES
            return farm.annual_cost(f["xy"], f["depth"], f["seabed"]) / e_net + farm.RENT

    return farms, lcoe, energy


# ================================================================ figures
def style(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("LCOE (£/MWh)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def sweep(ax, farms, lcoe, xs, make, marks, scale=1.0):
    """LCOE of both farms against one input; marks = (x, label, is_base, offset, ha) on the design curve.
    Returns the design LCOE at each marked value."""
    for name, f in farms.items():
        ax.plot(np.asarray(xs) * scale, [lcoe(f, make(x)) for x in xs], color=f["col"], lw=2, label=name)
    f = farms["270 MW"]
    out = {}
    for x, lab, base, off, ha in marks:
        y = lcoe(f, make(x))
        out[lab] = y
        ax.plot(x * scale, y, "o", ms=8 if base else 6.5, mfc=f["col"] if base else "white", mec=f["col"], mew=2, zorder=5)
        ax.annotate(f"{lab} ({y:.1f})" if base else lab, (x * scale, y), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=7.5, color=INK, fontweight="bold" if base else "normal")
    ax.set_xlim(xs[0] * scale, xs[-1] * scale)
    return out


def tornado(table, base):
    plot = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 10))
    y = np.arange(len(plot))
    for k, off in [("270 MW", -0.2), ("180 MW", 0.2)]:
        n = FARMS[k][0]
        ax.barh(y + off, plot[f"change {k}"], height=0.38, color=FARMS[k][1], edgecolor="white", lw=0.5,
                label=f"{k}, {n} turbines, base {half_up(base[k]):.1f} £/MWh")
    ax.set_yticks(y)
    ax.set_yticklabels([figure_label(c) for c in plot["case"]], fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Change in LCOE (£/MWh)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sens_tornado.png"), dpi=250)


# ================================================================ main
def main():
    farms, lcoe, energy = setup()
    names = list(FARMS)
    base = {k: lcoe(f) for k, f in farms.items()}
    for k, f in farms.items():
        print(f"Base, {k}: net energy {energy(f['n'], f['xy']) * farm.LOSSES / 1e3:.1f} GWh/yr, LCOE {base[k]:.2f} GBP/MWh")

    # ------------------------------------------------ Table 20 and Figure 12
    rows = [["Base", "Base settings"] + sum(([base[k], 0.0] for k in names), [])]
    for group, label, overrides, e in CASES:
        vals = {k: lcoe(f, overrides, **e) for k, f in farms.items()}
        rows.append([group, label] + sum(([vals[k], vals[k] - base[k]] for k in names), []))
    table = pd.DataFrame(rows, columns=["group", "case"] + sum(([f"LCOE {k}", f"change {k}"] for k in names), []))
    print("\nTable 20: sensitivity cases, LCOE and change from the base (GBP/MWh)")
    print(f"  {'Group':11s} {'Case':58s} {'270 MW':>7s} {'Change':>7s} {'180 MW':>7s} {'Change':>7s}")
    for _, r in table.iterrows():
        print(f"  {r.group:11s} {r.case:58s} {half_up(r['LCOE 270 MW']):7.1f} {half_up(r['change 270 MW']):+7.1f} "
              f"{half_up(r['LCOE 180 MW']):7.1f} {half_up(r['change 180 MW']):+7.1f}")
    table.round(2).to_csv(os.path.join(OUT, "sensitivity_results.csv"), index=False)
    tornado(table[table.group != "Base"], base)

    # ------------------------------------------------ Figure 13: discount rate
    print("\nSections B4.1 to B4.3 (270 MW design)")
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    make = lambda r: {(farm, "RATE"): r, (farm, "CRF"): crf(r, farm.LIFE)}
    m = sweep(ax, farms, lcoe, np.linspace(0.03, 0.12, 37), make,
              [(0.08, "Base, 8%", True, (-8, 6), "right"), (0.062, "Arup hurdle rate, 6.2%", False, (-6, 6), "right")],
              scale=100)
    print(f"  Discount rate 6.2% (Arup): {m['Arup hurdle rate, 6.2%']:.1f} GBP/MWh")
    style(ax, "Real discount rate (%), 20 year life")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sens_discount_rate.png"), dpi=300)

    # ------------------------------------------------ Figure 14: turbine price
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    m = sweep(ax, farms, lcoe, np.linspace(650, 1400, 31), lambda p: {(farm, "TURBINE_PER_KW"): p},
              [(783, "Nordex, Q3 2026", False, (-6, 6), "right"), (870, "Base: Vestas, Q2 2026", True, (-8, 8), "right"),
               (1040, "East Anglia TWO", False, (-6, 6), "right"), (1316, "Arup 2025", False, (-6, 6), "right")])
    print(f"  Turbine 783 GBP/kW (Nordex): {m['Nordex, Q3 2026']:.1f}; 1,040 (East Anglia TWO): "
          f"{m['East Anglia TWO']:.1f}; 1,316 (Arup): {m['Arup 2025']:.1f} GBP/MWh")
    style(ax, "Turbine price including tower (£/kW)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sens_turbine_price.png"), dpi=300)

    # ------------------------------------------------ Figure 15: array and export cables, same y scale
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True)
    m1 = sweep(a1, farms, lcoe, np.linspace(0.3e6, 2.2e6, 39), lambda c: {(farm, "ARRAY_PER_KM"): c},
               [(0.45e6, "DEA", False, (0, 9), "center"), (0.88e6, "Base", True, (-2, 10), "right"),
                (1.1e6, "Inch Cape", False, (8, -14), "left")], scale=1e-6)
    m2 = sweep(a2, farms, lcoe, np.linspace(0.6e6, 2.2e6, 33), lambda c: {(farm, "EXPORT_PER_KM"): c},
               [(0.84e6, "Seagreen", False, (-6, 6), "right"), (1.2e6, "Base", True, (-6, 6), "right"),
                (2.0e6, "High case", False, (-6, 6), "right")], scale=1e-6)
    print(f"  Array cable 0.45 to 1.1 M GBP/km: {m1['DEA']:.1f} to {m1['Inch Cape']:.1f} GBP/MWh")
    print(f"  Export cable 0.84 M GBP/km (Seagreen): {m2['Seagreen']:.1f}; 2.0 M GBP/km: {m2['High case']:.1f} GBP/MWh")
    style(a1, "(a) Array cable cost (M£/km)")
    style(a2, "(b) Export cable cost (M£/km)")
    a2.set_ylabel("")
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sens_cables.png"), dpi=300)

    # ------------------------------------------------ Table 21 and Figure 16: hub height
    print("\nTable 21: LCOE against hub height, without and with the cost of the taller tower (GBP/MWh)")
    print(f"  {'Hub':>4s} {'Tower':>6s} | {'270 MW: GWh':>11s} {'no tower':>8s} {'tower':>7s} | "
          f"{'180 MW: GWh':>11s} {'no tower':>8s} {'tower':>7s}")
    hub = {k: ([], []) for k in names}
    for h in HUBS:
        found = {(farm, "FOUND_PER_KW"): farm.FOUND_PER_KW * found_scale(h)}
        tower = {**found, (farm, "TURBINE_PER_KW"): farm.TURBINE_PER_KW + tower_extra_per_kw(h)}
        line = f"  {h:4d} {tower_extra_per_kw(h):6.1f}"
        for k, f in farms.items():
            a, b = lcoe(f, found, hub=h), lcoe(f, tower, hub=h)
            hub[k][0].append(a)
            hub[k][1].append(b)
            line += f" | {energy(f['n'], f['xy'], hub=h) * farm.LOSSES / 1e3:11.1f} {a:8.2f} {b:7.2f}"
        print(line)
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for k, f in farms.items():
        ax.plot(HUBS, hub[k][1], "o-", color=f["col"], lw=2, ms=6, label=f"{k}, tower costed")
        ax.plot(HUBS, hub[k][0], "o--", color=f["col"], lw=1.2, ms=4, mfc="white", label=f"{k}, tower not costed")
    ax.axvline(farm.TURBINE["hub"], color=MUTED, lw=0.8, ls=":")
    ax.text(farm.TURBINE["hub"] + 0.6, ax.get_ylim()[1], "design, 145 m\n(lowest with tip clearance)", va="top",
            fontsize=7.5, color=MUTED)
    ax.set_xticks(HUBS)
    style(ax, "Hub height (m)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sens_hub_height.png"), dpi=300)

    # ------------------------------------------------ Table 11: operating life
    print("\nTable 11: LCOE against operating life, at an 8% discount rate (GBP/MWh)")
    for n in LIVES:
        vals = [lcoe(f, {(farm, "LIFE"): n, (farm, "CRF"): crf(farm.RATE, n)}) for f in farms.values()]
        print(f"  {n:2d} years: CRF {crf(farm.RATE, n):.4f}, 270 MW {half_up(vals[0]):.1f}, 180 MW {half_up(vals[1]):.1f}")

    print(f"\nSaved {OUT}/sensitivity_results.csv and the five figures (sens_*.png)")
    plt.show()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
