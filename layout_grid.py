"""
layout_grid.py: layout and number of turbines (Section A7.3 and A7.4, Appendix A.7).
Every farm size from 2 to 33 turbines is laid out on a REGULAR grid (two lines of orientation,
MGN 654), inside the ideal + good area of site_maps.py, for the lowest LCOE.

    pip install py_wake
    python layout_grid.py                      print the results of the report from the saved files
                                               and draw Figures 10 and 11 (no optimising, seconds)
    python layout_grid.py --optimise           search every size again, 2 to 33 turbines (several hours)
    python layout_grid.py --optimise 6 12 18   search only these sizes (minutes each)
    python layout_grid.py --rescore            re-cost the saved layouts after a cost change in farm.py
    python layout_grid.py --map 12             detailed map of another farm size

Needs zones.npz (python site_maps.py), wind.py and farm.py (wind, power curve and every cost).

The turbines sit on a parallelogram grid of nr lines with nc turbines each:
    p(i, j) = centre + (i - (nr-1)/2) * a * u + (j - (nc-1)/2) * b * v
    u = (sin t, cos t), v = (sin(t+phi), cos(t+phi))
If nr x nc is more than N, the last line is left part full (still on the grid, so both lines of
orientation are kept). The optimiser (differential evolution) chooses the bearing t, the angle
phi (60 to 120 deg), the spacings a and b (5 to 12 D, and every pair at least 5 D apart) and the
centre, for every shape; every turbine must be inside the ideal + good area. Search on every 4th
hour, then the best shape is re-scored on all hours.

Saves four files, all in the folder grid_results:
    grid_lcoe_table.csv   one row per farm size: LCOE, energy, wake loss, grid shape, bearings and spacings;
                          best_in_band = yes marks the best farm for each number of export cables
    grid_layouts.csv      every turbine of every farm size (column farm_turbines says which farm);
                          e.g. the 270 MW design is the rows with farm_turbines = 18
    grid_result.png       LCOE and wake loss against size, and the best layout of each band
    grid_layout_18.png    detailed map of the design (turbines, lines of orientation, cables, export route,
                          depth, coast, 15 km limit), and its numbers printed for the report
"""

import os
import sys
import math
import time
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import differential_evolution
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import farm
import site_maps
from wind import load, hub_speed, KNOT_TO_MS
from py_wake.site import UniformSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014

ZONE = "ideal_good"             # design zone from Section A4: ideal + good area (54 km2)
MAX_MW = 500                    # largest farm studied
KW = farm.KW                    # kW per turbine
SIZES = list(range(2, int(MAX_MW * 1e3 // KW) + 1))                       # 2 to 33 turbines
BANDS = [(0, 90), (90, 180), (180, 270), (270, 300), (300, 500)]          # MW: 1 to 4 export cables of 90 MW, then
                                                                          # offshore substation (Table 6)
MIN_D = 5                       # minimum spacing, rotor diameters
SEARCH_EVERY = 4                # use every 4th hour while searching (the result is re-scored on every hour)
D = farm.D
DESIGN = 18                                            # farm drawn on the detailed map (18 x 15 MW = 270 MW)
OUT = "grid_results"                                   # every output goes in this folder
TABLE = os.path.join(OUT, "grid_lcoe_table.csv")
LAYOUTS = os.path.join(OUT, "grid_layouts.csv")
FIGURE = os.path.join(OUT, "grid_result.png")


def shapes(n):
    """(lines, turbines per line) for n turbines: at least 2 lines (1 line only for 2 or 3 turbines),
    the last line may be part full."""
    out = [(1, n)] if n <= 3 else []
    for r in range(2, n + 1):
        c = math.ceil(n / r)
        if c < r:
            break
        if r * c - n < c:
            out.append((r, c))
    return out


def grid_points(p, nr, nc, n):
    t, phi, a, b, cx, cy = p
    t, phi = math.radians(t), math.radians(phi)
    u = np.array([math.sin(t), math.cos(t)])
    v = np.array([math.sin(t + phi), math.cos(t + phi)])
    i, j = np.meshgrid(np.arange(nr) - (nr - 1) / 2, np.arange(nc) - (nc - 1) / 2, indexing="ij")
    xy = np.array([cx, cy]) + (i.ravel()[:, None] * a * D) * u + (j.ravel()[:, None] * b * D) * v
    return xy[:n]                                   # drop the end of the last line if it is part full


def band_of(n):
    mw = n * KW / 1e3
    return next(((lo, hi) for lo, hi in BANDS if lo < mw <= hi), None)


def half_up(x, nd):
    """Round halves up (125.25 -> 125.3), as in the report tables; Python's round() would give 125.2."""
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def capacity(zone):
    """How many turbines fit in the zone on the best hexagonal grid at the minimum spacing (upper bound)."""
    P, sp, best = Path(zone), MIN_D * D * 1.02, 0
    for ang in np.radians(range(0, 60, 15)):
        for ox in np.linspace(0, sp, 3):
            for oy in np.linspace(0, sp, 3):
                i, j = np.meshgrid(np.arange(-80, 81), np.arange(-80, 81))
                gx = (i + 0.5 * (j % 2)) * sp + ox
                gy = j * sp * math.sqrt(3) / 2 + oy
                g = np.c_[(math.cos(ang) * gx - math.sin(ang) * gy).ravel(),
                          (math.sin(ang) * gx + math.cos(ang) * gy).ravel()] + zone.mean(axis=0)
                best = max(best, P.contains_points(g).sum())
    return best


# ================================================================ optimisation
def optimise(sizes):
    t0 = time.time()
    zones = np.load("zones.npz")
    poly = zones[ZONE]
    path = Path(poly)
    depth_at = farm.depth_function()
    seabed_at = farm.seabed_function(zones["all_buildable"])
    seabed_raw = farm.seabed_function(zones["all_buildable"], smooth=False)
    ROCK = farm.SEABED_FACTOR["Rock or other hard substrata"]

    df = load(farm.DATA_FILE)
    ws_h = hub_speed(df["knots"].to_numpy(float) * KNOT_TO_MS, farm.TURBINE["hub"])
    wd_h = df["dir"].to_numpy(float)
    ws = np.arange(0, 30.01, 0.05)
    wt = WindTurbine(farm.TURBINE["name"], diameter=farm.D, hub_height=farm.TURBINE["hub"],
                     powerCtFunction=PowerCtTabular(ws, farm.power(ws), "w", farm.thrust(ws)))
    wfm = Bastankhah_PorteAgel_2014(UniformSite(ti=0.06), wt, k=farm.WAKE_K, use_effective_ws=False)
    sub = slice(None, None, SEARCH_EVERY)
    alone = wfm.aep([0.0], [0.0], wd=wd_h, ws=ws_h, time=True) * 1e3            # MWh/yr, one turbine, no wakes

    def lcoe(xy, full=False):
        s = slice(None) if full else sub
        E = wfm.aep(xy[:, 0], xy[:, 1], wd=wd_h[s], ws=ws_h[s], time=True) * 1e3   # annual figure for any number of hours
        C = farm.annual_cost(xy, depth_at(xy), seabed_at(xy))
        return C / (E * farm.LOSSES) + farm.RENT, E

    xmin, ymin = poly.min(0)
    xmax, ymax = poly.max(0)
    bounds = [(0, 180), (60, 120), (MIN_D, 12), (MIN_D, 12), (xmin, xmax), (ymin, ymax)]
    rows, layouts = [], []
    print(f"{'N':>3} {'MW':>5} {'shape':>6} {'LCOE':>7} {'wake %':>7}   (GBP/MWh, regular grid, zone {ZONE})", flush=True)
    for n in sizes:
        if capacity(poly) < n:
            print(f"{n:3d} {n * KW / 1e3:5.0f}  does not fit", flush=True)
            continue
        best = None
        for nr, nc in shapes(n):
            def obj(p):
                xy = grid_points(p, nr, nc, n)
                outside = ~path.contains_points(xy)
                d = cdist(xy, xy) + np.eye(n) * 1e9
                pen = outside.sum() * 50 + max(0.0, MIN_D * D - d.min()) / D * 50
                return 200 + pen if pen > 0 else lcoe(xy)[0]
            res = differential_evolution(obj, bounds, seed=1, maxiter=60, popsize=12, tol=1e-6,
                                         polish=False, init="sobol")
            if res.fun >= 200:
                continue                                        # this shape does not fit
            if best is None or res.fun < best[0]:
                best = (res.fun, (nr, nc), res.x)
        if best is None:
            print(f"{n:3d} {n * KW / 1e3:5.0f}  no grid fits", flush=True)
            continue
        _, (nr, nc), p = best
        xy = grid_points(p, nr, nc, n)
        c, E = lcoe(xy, full=True)
        P = wfm(xy[:, 0], xy[:, 1], wd=wd_h, ws=ws_h, time=True).Power.values
        e = P.sum(axis=1) / 1e6 * 8760 / P.shape[1]                                 # MWh/yr per turbine
        depth, seabed = depth_at(xy), seabed_at(xy)
        on_rock = seabed_raw(xy) >= ROCK - 1e-6
        d = cdist(xy, xy) + np.eye(n) * 1e9
        wake = 100 * (1 - E / (n * alone))
        rows.append(dict(turbines=n, MW=n * KW / 1e3, shape=f"{nr}x{nc}", lcoe=round(c, 2),
                         energy_GWh=round(E * farm.LOSSES / 1e3, 1), wake_pct=round(wake, 2),
                         bearing_deg=round(p[0], 1), angle_deg=round(p[1], 1), a_D=round(p[2], 2), b_D=round(p[3], 2),
                         min_spacing_D=round(d.min() / D, 2), on_rock=int(on_rock.sum()),
                         array_km=round(farm.array_cable_km(xy), 1), export_km=round(farm.export_km(xy.mean(axis=0)), 1)))
        la, lo = farm.to_latlon(xy)
        layouts.append(pd.DataFrame(dict(
            farm_turbines=n, turbine=np.arange(1, n + 1), latitude=la.round(5), longitude=lo.round(5),
            x_m=xy[:, 0].round(0), y_m=xy[:, 1].round(0), depth_m=depth.round(1),
            seabed_factor=seabed.round(2), on_mapped_rock=np.where(on_rock, "yes", "no"),
            energy_MWh_per_yr=e.round(0), wake_loss_pct=(100 * (1 - e / alone)).round(2))))
        print(f"{n:3d} {n * KW / 1e3:5.0f} {nr:>3d}x{nc:<2d} {c:7.1f} {wake:7.2f}   [{(time.time() - t0) / 60:.0f} min]",
              flush=True)

    T = pd.DataFrame(rows)
    Lall = pd.concat(layouts) if layouts else pd.DataFrame()
    rerun = [r["turbines"] for r in rows]
    if os.path.exists(TABLE) and os.path.exists(LAYOUTS):   # keep sizes from an earlier run that were not rerun now
        oldT, oldL = pd.read_csv(TABLE), pd.read_csv(LAYOUTS)
        T = pd.concat([oldT[~oldT.turbines.isin(rerun)], T]).sort_values("turbines")
        Lall = pd.concat([oldL[~oldL.farm_turbines.isin(rerun)], Lall]).sort_values(["farm_turbines", "turbine"])
    os.makedirs(OUT, exist_ok=True)
    T.drop(columns="best_in_band", errors="ignore").to_csv(TABLE, index=False)
    Lall.to_csv(LAYOUTS, index=False)
    print(f"Saved {TABLE} and {LAYOUTS} ({(time.time() - t0) / 60:.0f} min)")


def rescore():
    """Re-cost the saved grid layouts with the current farm.py inputs, without optimising.
    Valid when the change does not depend on where the turbines are (e.g. the TNUoS credit, a cost per kW)."""
    zones = np.load("zones.npz")
    depth_at = farm.depth_function()
    seabed_at = farm.seabed_function(zones["all_buildable"])
    df = load(farm.DATA_FILE)
    ws_h = hub_speed(df["knots"].to_numpy(float) * KNOT_TO_MS, farm.TURBINE["hub"])
    wd_h = df["dir"].to_numpy(float)
    ws = np.arange(0, 30.01, 0.05)
    wt = WindTurbine(farm.TURBINE["name"], diameter=farm.D, hub_height=farm.TURBINE["hub"],
                     powerCtFunction=PowerCtTabular(ws, farm.power(ws), "w", farm.thrust(ws)))
    wfm = Bastankhah_PorteAgel_2014(UniformSite(ti=0.06), wt, k=farm.WAKE_K, use_effective_ws=False)
    T, L = pd.read_csv(TABLE), pd.read_csv(LAYOUTS)
    for i, n in T.turbines.items():
        xy = L.loc[L.farm_turbines == n, ["x_m", "y_m"]].to_numpy(float)
        E = wfm.aep(xy[:, 0], xy[:, 1], wd=wd_h, ws=ws_h, time=True) * 1e3
        C = farm.annual_cost(xy, depth_at(xy), seabed_at(xy))
        T.loc[i, "lcoe"] = round(C / (E * farm.LOSSES) + farm.RENT, 2)
        print(f"{n:3d} {n * KW / 1e3:5.0f}  LCOE {T.loc[i, 'lcoe']:7.2f}", flush=True)
    T.drop(columns="best_in_band", errors="ignore").to_csv(TABLE, index=False)
    print(f"Saved {TABLE}")


# ================================================================ results and figure (from the saved files)
def lines_along(xy, bearing_deg):
    """Turbines on the same straight line along bearing_deg (lines are at least 5 D apart)."""
    t = math.radians(bearing_deg)
    w = np.array([math.sin(t), math.cos(t)])
    n = np.array([-w[1], w[0]])
    across = xy @ n
    order = np.argsort(across)
    groups = np.split(order, np.where(np.diff(across[order]) > D)[0] + 1)
    return [g[np.argsort(xy[g] @ w)] for g in groups if len(g) >= 2]


def plot_results():
    T = pd.read_csv(TABLE).sort_values("turbines")
    G = pd.read_csv(LAYOUTS)
    zones = np.load("zones.npz")
    best = {}
    for _, r in T.iterrows():
        b = band_of(int(r.turbines))
        if b and (b not in best or r.lcoe < best[b].lcoe):
            best[b] = r
    # the three cable band is represented by the design, which fills all three cables (Section A7.3)
    design_band = band_of(DESIGN)
    cheapest = best[design_band]
    best[design_band] = T.set_index("turbines").loc[DESIGN].rename(DESIGN)
    best[design_band]["turbines"] = DESIGN
    names = {(0, 90): "1 cable, 66 kV", (90, 180): "2 cables", (180, 270): "3 cables (design)",
             (270, 300): "4 cables", (300, 500): "Offshore substation, 220 kV"}
    shown = []
    print("\nTable 6: best regular grid for each export arrangement (8 %, 20 years)")
    print(f"  {'Export system':28s} {'Turbines':>8s} {'MW':>4s} {'Grid':>5s} {'GBP/MWh':>8s} {'p/kWh':>6s} "
          f"{'GWh/yr':>7s} {'Load f.':>8s} {'Wake':>6s}")
    for (lo, hi), r in sorted(best.items()):
        n = int(r.turbines)
        L = G[G.farm_turbines == n]
        cf = r.energy_GWh * 1e3 / (r.MW * 8760) * 100
        print(f"  {names[(lo, hi)]:28s} {n:8d} {r.MW:4.0f} {r['shape']:>5s} {half_up(r.lcoe, 1):8.1f} "
              f"{half_up(r.lcoe / 10, 2):6.2f} {half_up(r.energy_GWh, 0):7,.0f} {cf:7.1f}% {half_up(r.wake_pct, 1):5.1f}%")
        shown.append(((lo, hi), r, L, cf))
    if int(cheapest.turbines) != DESIGN:
        print(f"  (cheapest in the three cable band: {int(cheapest.turbines)} turbines, {cheapest.lcoe:.1f} GBP/MWh, "
              f"which leaves the third cable part used)")
    T["best_in_band"] = np.where(T.turbines.isin([int(s[1].turbines) for s in shown]), "yes", "")
    T.to_csv(TABLE, index=False)
    low = min(shown, key=lambda s: s[1].lcoe)[1]
    print(f"Lowest LCOE overall: {int(low.turbines)} turbines ({low.MW:.0f} MW), {low.lcoe:.1f} GBP/MWh")

    nb = len(shown)
    fig = plt.figure(figsize=(max(4.6 * nb, 10), 9.5))
    gs = fig.add_gridspec(2, nb, height_ratios=[1, 1.6], hspace=0.32, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    for k, (lo, hi) in enumerate(BANDS):
        ax.axvspan(lo, hi, color="grey", alpha=0.07 if k % 2 == 0 else 0.0, lw=0)
        if lo > 0:
            ax.axvline(lo, ls="--", color="grey", lw=0.8)
    ax.plot(T.MW, T.lcoe, "o-", color="#1a9850", ms=4, label="LCOE, regular grid (MGN 654), ideal + good zone")
    for (lo, hi), r, L, cf in shown:
        ax.plot(r.MW, r.lcoe, "*", color="#d73027", ms=16, zorder=4)
        ax.annotate(f"{int(r.turbines)} turbines\n{half_up(r.lcoe, 1):.1f}", (r.MW, r.lcoe), textcoords="offset points",
                    xytext=(0, -30), ha="right" if r.MW > 0.9 * MAX_MW else "center", fontsize=8, color="#d73027")
    ax.plot([], [], "*", color="#d73027", ms=12, label="best grid for each number of export cables (design for three cables)")
    axw = ax.twinx()
    axw.plot(T.MW, T.wake_pct, "s--", color="#2166ac", ms=3, lw=1)
    axw.set_ylabel("Farm wake loss (%)", color="#2166ac")
    axw.tick_params(axis="y", colors="#2166ac")
    axw.set_ylim(0, T.wake_pct.max() * 1.4)
    ax.plot([], [], "s--", color="#2166ac", ms=3, lw=1, label="wake loss of the grid (right axis)")
    for x0, lab in [(farm.EXPORT_CABLE_MW, "2nd export cable"), (2 * farm.EXPORT_CABLE_MW, "3rd cable"),
                    (3 * farm.EXPORT_CABLE_MW, "4th cable"), (farm.DIRECT_MAX_MW, "offshore substation + 220 kV")]:
        ax.text(x0, ax.get_ylim()[0], " " + lab, va="bottom", ha="left", fontsize=7, color="grey", rotation=90)
    ax.set_xlim(0, MAX_MW * 1.02)
    ax.set_xlabel("Farm capacity (MW)")
    ax.set_ylabel("LCOE (£/MWh)")
    ax.set_title("LCOE and wake loss against farm size: regular grid layouts (every size optimised)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ext = zones[ZONE] / 1000
    xlim = (ext[:, 0].min() - 0.6, ext[:, 0].max() + 0.6)
    ylim = (ext[:, 1].min() - 0.6, ext[:, 1].max() + 0.6)
    vmax = max(L.wake_loss_pct.max() for _, _, L, _ in shown)
    maps = []
    for i, ((lo, hi), r, L, cf) in enumerate(shown):
        m = fig.add_subplot(gs[1, i])
        maps.append(m)
        for zn, col, lab in [("all_buildable", "#e3e3e3", "other buildable sea"),
                             ("ideal_good", "#91cf60", "good"), ("ideal", "#1a9850", "ideal")]:
            z = zones[zn] / 1000
            m.fill(z[:, 0], z[:, 1], color=col, label=lab)
        xy = L[["x_m", "y_m"]].to_numpy(float)
        for bearing, col in [(r.bearing_deg, "#3366cc"), (r.bearing_deg + r.angle_deg, "#ff9900")]:
            for g in lines_along(xy, bearing):
                m.plot(xy[g, 0] / 1000, xy[g, 1] / 1000, color=col, lw=4, alpha=0.35, solid_capstyle="round")
        nodes = np.vstack([xy, xy.mean(axis=0)])
        tree = minimum_spanning_tree(cdist(nodes, nodes)).tocoo()
        for p, q in zip(tree.row, tree.col):
            m.plot(nodes[[p, q], 0] / 1000, nodes[[p, q], 1] / 1000, color="black", lw=0.7)
        m.plot(*(nodes[-1] / 1000), "ks", ms=5)
        sc = m.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, c=L.wake_loss_pct, cmap="Reds", vmin=0, vmax=vmax,
                       s=45 if len(L) > 15 else 70, edgecolors="black", lw=0.6, zorder=3)
        m.set_xlim(*xlim)
        m.set_ylim(*ylim)
        m.set_aspect("equal")
        m.set_xlabel("km east")
        if i == 0:
            m.set_ylabel("km north")
            m.legend(loc="upper left", fontsize=7)
        else:
            m.set_yticklabels([])
        m.set_title(f"{lo} to {hi} MW: {int(r.turbines)} turbines, {r['shape']} grid\n"
                    f"{half_up(r.lcoe, 1):.1f} £/MWh, wake {half_up(r.wake_pct, 1):.1f}%, CF {cf:.0f}%", fontsize=9)
    cb = fig.colorbar(sc, ax=maps, shrink=0.6, pad=0.01)
    cb.set_label("Wake loss per turbine (%)", fontsize=8)
    fig.text(0.01, 0.005, "Zones: site_maps.py (EMODnet bathymetry DTM 2024, EUSeaMap 2025, EMODnet vessel density). "
             "Wind: Met Office Camborne 2006. Shaded bands: the two lines of orientation; square: centre of the farm; "
             "black lines: array cable network as costed.", fontsize=7)
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURE}")


def plot_map(N):
    """Detailed map of one farm size, and its numbers for the report."""
    G = pd.read_csv(LAYOUTS)
    L = G[G.farm_turbines == N].reset_index(drop=True)
    T = pd.read_csv(TABLE).set_index("turbines")
    if N not in T.index:
        print(f"No {N}-turbine farm in {TABLE}: run python layout_grid.py {N} first")
        return
    row = T.loc[N]
    xy = L[["x_m", "y_m"]].to_numpy(float)

    # ---------------------------------------------------------- numbers for the text
    d = cdist(xy, xy) + np.eye(len(xy)) * 1e9
    _, _, land_xy = farm.load_depth(farm.DEPTH_FILE)
    coast_km = cKDTree(land_xy).query(xy)[0] / 1000
    station_km = np.hypot(*(xy - farm.to_xy(*farm.STATION)).T) / 1000
    b_rows, b_cols = row.bearing_deg, row.bearing_deg + row.angle_deg
    print()
    print(f"Section A7.4: {N} turbines, {row['shape']} grid: LCOE {row.lcoe:.2f} GBP/MWh, {row.energy_GWh:.1f} GWh/yr, "
          f"wake loss {row.wake_pct:.2f} %")
    print(f"Lines of orientation: bearing {b_rows % 180:.0f} deg (spacing {row.a_D:.1f} D apart) and "
          f"{b_cols % 180:.0f} deg (spacing {row.b_D:.1f} D); angle between them {row.angle_deg:.0f} deg")
    rock = ", ".join(str(t) for t in L.turbine[L.on_mapped_rock == "yes"]) or "none"
    print(f"Closest pair {d.min() / D:.2f} D; array cable {row.array_km:.1f} km; export route to Hayle "
          f"{row.export_km:.1f} km; on mapped rock: turbine {rock}")
    print(f"Depth {L.depth_m.min():.1f} to {L.depth_m.max():.1f} m; coast {coast_km.min():.1f} to {coast_km.max():.1f} km; "
          f"Camborne station {station_km.min():.1f} to {station_km.max():.1f} km (limit 15 km)")

    # ---------------------------------------------------------- map
    lat, lon, Z = site_maps.load_grid()
    LO, LA = np.meshgrid(lon, lat)
    land = np.isnan(Z) | (Z >= 0)
    zones = np.load("zones.npz")

    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    ax.set_facecolor("#f4f8fb")
    cs = ax.contour(LO, LA, np.where(land, np.nan, -Z), levels=[15, 30], colors="#2f5d8a",
                    linewidths=0.6, alpha=0.6, zorder=1)
    ax.clabel(cs, fmt="%d m", fontsize=7)
    for zn, col in [("all_buildable", "#d9d9d9"), ("ideal_good", "#91cf60"), ("ideal", "#1a9850")]:
        la, lo = farm.to_latlon(zones[zn])
        ax.fill(lo, la, color=col, alpha=0.9, lw=0, zorder=2)
    site_maps.base_map(ax, LO, LA, land, "", fs=8)

    # export route to the Hayle landfall (farm.WAYPOINTS, as costed in farm.export_km)
    centre = xy.mean(axis=0)
    cla, clo = farm.to_latlon(centre[None])
    rl = [cla[0]] + [w[0] for w in farm.WAYPOINTS]
    ro = [clo[0]] + [w[1] for w in farm.WAYPOINTS]
    ax.plot(ro, rl, color="#7b3294", lw=2.2, ls=(0, (5, 2)), zorder=6)
    ax.plot(ro[-1], rl[-1], "^", color="#7b3294", ms=8, mec="black", mew=0.5, zorder=7)

    # the two lines of orientation (MGN 654 section 6.2)
    la, lo = farm.to_latlon(xy)
    for bearing, col in [(b_rows, "#3366cc"), (b_cols, "#ff9900")]:
        for g in lines_along(xy, bearing):
            ax.plot(lo[g], la[g], color=col, lw=5, alpha=0.35, solid_capstyle="round", zorder=6)

    # array cables as costed (farm.array_cable_km: shortest network to the collection point)
    nodes = np.vstack([xy, centre])
    tree = minimum_spanning_tree(cdist(nodes, nodes)).tocoo()
    nla, nlo = farm.to_latlon(nodes)
    for p, q in zip(tree.row, tree.col):
        ax.plot(nlo[[p, q]], nla[[p, q]], color="black", lw=0.8, zorder=8)
    ax.plot(nlo[-1], nla[-1], "s", color="black", ms=6, zorder=8)

    sc = ax.scatter(L.longitude, L.latitude, c=L.wake_loss_pct, cmap="Reds", vmin=0, vmax=max(8.0, L.wake_loss_pct.max()), s=85,
                    edgecolors="black", linewidths=0.7, zorder=9)
    for _, r in L.iterrows():
        ax.text(r.longitude + 0.0035, r.latitude + 0.0022, str(int(r.turbine)), fontsize=6.5, zorder=10)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Wake loss per turbine (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # prevailing wind and a scale bar
    ax.annotate("", xy=(-5.458, 50.356), xytext=(-5.474, 50.343),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6), zorder=10)
    ax.text(-5.476, 50.341, "strongest winds\nfrom 180-240°", fontsize=7.5, color="#333", va="top")
    x0, y0 = -5.262, 50.356
    km = 2 / (111.32 * math.cos(math.radians(y0)))
    ax.plot([x0, x0 + km], [y0, y0], color="black", lw=2, zorder=10)
    ax.text(x0 + km / 2, y0 + 0.002, "2 km", ha="center", fontsize=7.5, zorder=10)

    ax.set_xlim(-5.48, -5.215)
    ax.set_ylim(50.18, 50.362)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{N} x V236-15 MW on a regular {row['shape']} grid: LCOE {half_up(row.lcoe, 1):.1f} £/MWh, "
                 f"wake loss {row.wake_pct:.1f} %", fontsize=10, loc="left")
    handles = [Patch(fc="#1a9850", label="ideal area"),
               Patch(fc="#91cf60", label="good area (design zone: ideal + good)"),
               Patch(fc="#d9d9d9", label="other buildable sea"),
               Line2D([], [], color="black", marker="o", mfc="#fb6a4a", ls="", ms=7, label="V236-15 MW turbine (number), coloured by wake loss"),
               Line2D([], [], color="#3366cc", lw=5, alpha=0.35, label=f"line of orientation, {b_rows % 180:.0f}°"),
               Line2D([], [], color="#ff9900", lw=5, alpha=0.35, label=f"line of orientation, {b_cols % 180:.0f}°"),
               Line2D([], [], color="black", lw=0.8, marker="s", ms=5, label="66 kV array cables (as costed)"),
               Line2D([], [], color="#7b3294", lw=2, ls=(0, (5, 2)), marker="^", ms=6,
                      label="export route to the Hayle landfall"),
               Line2D([], [], color="black", ls="--", lw=0.8, label="15 km from Camborne station")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=7.5, framealpha=0.95)
    fig.text(0.01, 0.005, "Data: EMODnet Bathymetry DTM 2024 (depth, coastline); EMODnet EUSeaMap 2025 (seabed, in zones); "
             "Met Office Camborne hourly wind 2006 (energy). Layout and figure: layout_grid.py; zones: site_maps.py.",
             fontsize=6.5, color="#444", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(OUT, f"grid_layout_{N}.png")
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from the folder this file is in
    args = sys.argv[1:]
    if "--map" in args:
        plot_map(int(args[args.index("--map") + 1]))
    else:
        if "--optimise" in args:
            sizes = [int(a) for a in args if a.isdigit()]
            optimise(sizes or SIZES)
        elif "--rescore" in args:
            rescore()
        plot_results()
        plot_map(DESIGN)
    plt.show()
