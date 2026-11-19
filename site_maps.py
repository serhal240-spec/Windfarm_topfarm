"""
site_maps.py: offshore site selection maps (Part A).

    python site_maps.py

Needs, in the same folder: farm.py and the three EMODnet files (bathymetry, seabed
habitats, vessel density). Prints the numbers of Section A4 and Appendix A.4, shows
and saves two figures, and writes zones.npz (the ideal and good area used by layout_grid.py):
  site_criteria_maps.png       Figure 5: (a) seabed (b) depth (c) shipping
  site_suitability_map.png     Figure 6: the weighted score of all criteria

Method
1. Hard limits: within 15 km of the station, 15 to 50 m deep, at least 2 km from
   the coast. Anything else is impossible.
2. Every remaining spot is scored 1 to 5 on eight criteria, averaged over the
   1.75 km around it (about the area of a six turbine farm), with the rules
   and weights of Table 2.
"""

import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from scipy.spatial import cKDTree

import farm

DEPTH_FILE = "emodnet_bathymetry_dtm2024_cornwall.csv"
SEABED_FILE = "emodnet_euseamap2025_eunis2019_cornwall.geojson"
SHIP_FILE = "emodnet_vessel_density_allavg_cornwall.csv"
FOOTPRINT_M = 1750            # radius of the area scored around each spot

WEIGHTS = dict(seabed=0.30, wind=0.20, room=0.10, depth=0.10, ship=0.10, coast=0.10, sac=0.05, export=0.05)
BANDS = [4.5, 4.0, 3.5]       # ideal, good, fair; below 3.5 is poor
ROCK = "Rock or other hard substrata"

SOURCES_DATA = ("EMODnet Seabed Habitats, EUSeaMap 2025 broad scale seabed habitat map (EUNIS 2019), layer eusm2025_eunis2019_full, accessed 7 Nov 2026. "
                "EMODnet Bathymetry Consortium, EMODnet Digital Bathymetry (DTM 2024), accessed 7 Nov 2026.\n"
                "EMODnet Human Activities, vessel density, all vessel types, annual average (layer vesseldensity_allavg; AIS from CLS, ORBCOMM, vesseltracker.com), "
                "accessed 11 Nov 2026. Met Office, Camborne (WMO 03808) hourly wind, 2006.")
PLACES = [("Portreath", 50.262, -5.293), ("Porthtowan", 50.286, -5.240), ("Godrevy Pt", 50.243, -5.400),
          ("St Ives", 50.214, -5.480), ("Hayle", 50.186, -5.420), ("Camborne station", 50.2179, -5.3273)]


# ================================================================ data
def load_grid():
    d = pd.read_csv(DEPTH_FILE, skiprows=[1])
    lat, lon = np.sort(d.latitude.unique()), np.sort(d.longitude.unique())
    Z = d.pivot_table(index="latitude", columns="longitude", values="elevation").reindex(index=lat, columns=lon).to_numpy()
    return lat, lon, Z


def seabed_polygons():
    """EUSeaMap polygons as matplotlib paths (for drawing), grouped by substrate.
    Outer rings run anticlockwise and holes clockwise, so the holes are left unfilled."""
    def oriented(ring, anticlockwise):
        r = np.array(ring)
        area = np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])
        return r if (area > 0) == anticlockwise else r[::-1]

    g = json.load(open(SEABED_FILE))
    out = {}
    for f in g["features"]:
        for poly in f["geometry"]["coordinates"]:
            rings = [oriented(poly[0], True)] + [oriented(h, False) for h in poly[1:]]
            codes = []
            for r in rings:
                codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
            out.setdefault(f["properties"]["substrate"], []).append(Path(np.vstack(rings), codes))
    return out


def seabed_at(lon, lat):
    """Substrate name at each point: inside a polygon's outer ring and not in one of its holes.
    Polygons are read in file order."""
    g = json.load(open(SEABED_FILE))
    pts = np.c_[lon, lat]
    bed = np.array(["Not classified"] * len(pts), dtype=object)
    for f in g["features"]:
        b = f["bbox"]
        m = (pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) & (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3])
        if not m.any():
            continue
        idx = np.where(m)[0]
        inside = np.zeros(len(idx), bool)
        for poly in f["geometry"]["coordinates"]:
            i = Path(np.array(poly[0])).contains_points(pts[idx])
            for hole in poly[1:]:
                i &= ~Path(np.array(hole)).contains_points(pts[idx])
            inside |= i
        bed[idx[inside]] = f["properties"]["substrate"]
    return bed


def export_route_km(pts):
    """Cable length to the Hayle landfall, joining the route round Godrevy Point at its nearest waypoint."""
    wp = [farm.to_xy(*w) for w in farm.WAYPOINTS]
    tail = [sum(np.hypot(*(wp[j + 1] - wp[j])) for j in range(k, len(wp) - 1)) for k in range(len(wp))]
    return np.min([np.hypot(*(pts - wp[k]).T) + tail[k] for k in range(len(wp))], axis=0) / 1000 * farm.ROUTE_FACTOR


# ================================================================ scoring rules (group zone study)
def score_seabed(rock):  return np.select([rock < 0.05, rock <= 0.25], [5, 3], 1)
def score_room(usable):  return np.select([usable >= 0.9, usable >= 0.75], [5, 3], 1)
def score_ship(v):       return np.select([v < 1, v <= 5], [5, 3], 1)
def score_coast(km):     return np.select([km >= 7, km >= 5, km >= 3], [5, 4, 3], 2)
def score_export(km):    return np.select([km < 8, km < 14, km < 20], [5, 4, 3], 2)


def score_depth(z):
    return np.select([(z >= 20) & (z <= 30), (z > 30) & (z <= 35), (z > 35) & (z <= 40), (z >= 15) & (z < 20)],
                     [5, 4, 3, 3], 2)



# ================================================================ map helpers
def circle():
    th = np.linspace(0, 2 * np.pi, 400)
    return (farm.STATION[1] + 15 / (111.32 * math.cos(math.radians(farm.STATION[0]))) * np.sin(th),
            farm.STATION[0] + 15 / 110.54 * np.cos(th))


def base_map(ax, LO, LA, land, title, fs=7):
    ax.contourf(LO, LA, np.where(land, 1.0, np.nan), levels=[0.5, 1.5], colors=["#ece6d6"], zorder=3)
    ax.contour(LO, LA, np.where(land, 1.0, 0.0), levels=[0.5], colors="#555", linewidths=0.6, zorder=4)
    ax.plot(*circle(), color="black", lw=0.8, ls="--", zorder=5)
    for n, a, b in PLACES:
        ax.plot(b, a, "o", ms=2.5, color="black", zorder=6)
        ax.text(b + 0.004, a + 0.002, n, fontsize=fs, zorder=6)
    ax.set_xlim(-5.56, -5.12); ax.set_ylim(50.178, 50.36)
    ax.set_aspect(1 / math.cos(math.radians(50.28)))
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.tick_params(labelsize=7)


def save_zones(cls, ok, xy):
    """Outline of each nested zone (ideal; ideal + good; ... ; all buildable) in metres, for layout_grid.py."""
    from scipy import ndimage
    out = {}
    names = ["ideal", "ideal_good", "ideal_good_fair", "all_buildable"]
    for k, name in enumerate(names):
        m = ok & (cls <= k)
        lab, n = ndimage.label(m)
        big = lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))
        cs = plt.figure().add_subplot().contour(xy[..., 0], xy[..., 1], big.astype(float), levels=[0.5])
        poly = max(cs.allsegs[0], key=len)[::3]
        plt.close()
        keep = [poly[0]]
        for q in poly[1:]:                       # drop repeated points
            if np.hypot(*(q - keep[-1])) > 1e-6:
                keep.append(q)
        poly = np.array(keep)
        if np.hypot(*(poly[0] - poly[-1])) < 1e-6:
            poly = poly[:-1]
        area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
        out[name] = poly if area > 0 else poly[::-1]   # anticlockwise
    np.savez("zones.npz", **out)


# ================================================================ main
def main():
    lat, lon, Z = load_grid()
    LO, LA = np.meshgrid(lon, lat)
    land = np.isnan(Z) | (Z >= 0)
    depth = np.where(land, np.nan, -Z)
    xy = farm.to_xy(LA, LO)
    station = farm.to_xy(*farm.STATION)
    from_station = np.hypot(xy[..., 0] - station[0], xy[..., 1] - station[1]) / 1000
    land_tree = cKDTree(xy[land])
    coast = np.full(Z.shape, np.nan)
    coast[~land] = land_tree.query(xy[~land])[0] / 1000

    # 1. hard limits
    ok = (~land & (from_station <= farm.LIMIT_KM) & (depth >= farm.DEPTH_MIN) & (depth <= farm.DEPTH_MAX)
          & (coast >= farm.COAST_MIN_KM))
    P = xy[ok]
    cell_km2 = (lat[1] - lat[0]) * 110.54 * (lon[1] - lon[0]) * 111.32 * math.cos(math.radians(50.3))
    sea = ~land & (from_station <= farm.LIMIT_KM)
    shallow = sea & (depth < farm.DEPTH_MIN)
    near_coast = sea & (depth >= farm.DEPTH_MIN) & (coast < farm.COAST_MIN_KM)
    print("Hard limits (Section A4.2)")
    print(f"  Sea within {farm.LIMIT_KM:.0f} km of the station   {sea.sum() * cell_km2:6.1f} km2 "
          f"(deepest {np.nanmax(depth[sea]):.0f} m)")
    print(f"  Too shallow (under {farm.DEPTH_MIN:.0f} m)         {shallow.sum() * cell_km2:6.1f} km2")
    print(f"  Too close to coast (under {farm.COAST_MIN_KM:.0f} km)    {near_coast.sum() * cell_km2:6.1f} km2")
    print(f"  Buildable                          {ok.sum() * cell_km2:6.1f} km2 ({ok.sum()} grid points)")

    # 2. criteria at every buildable point
    polys = seabed_polygons()
    bed = seabed_at(LO[ok], LA[ok])
    route = export_route_km(P)
    ship_df = pd.read_csv(SHIP_FILE).dropna()
    ship_tree = cKDTree(np.c_[ship_df.latitude, ship_df.longitude * math.cos(math.radians(50.3))])
    ship = ship_df.density.to_numpy()[ship_tree.query(np.c_[LA[ok], LO[ok] * math.cos(math.radians(50.3))])[1]]

    # averaged over the footprint around each point
    near = cKDTree(P).query_ball_point(P, FOOTPRINT_M)
    n_total = cKDTree(xy.reshape(-1, 2)).query_ball_point(P, FOOTPRINT_M, return_length=True)
    usable = np.array([len(n) for n in near]) / n_total
    rock = np.array([np.mean(bed[n] == ROCK) for n in near])
    S = dict(seabed=score_seabed(rock),
             wind=np.full(len(P), 5),     # power law: the same wind everywhere, so it cannot separate sites
             room=score_room(usable),
             depth=score_depth(np.array([depth[ok][n].mean() for n in near])),
             ship=score_ship(np.array([ship[n].mean() for n in near])),
             coast=score_coast(coast[ok]),
             sac=np.full(len(P), 2),       # Bristol Channel Approaches SAC covers all buildable sea
             export=score_export(route))

    def total(w):
        return sum(w[k] * S[k] for k in w) / sum(w.values())

    score = total(WEIGHTS)

    # ------------------------------------------------ figure 1: criteria
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.6))
    outside = np.where(~land & (from_station > farm.LIMIT_KM), 1.0, np.nan)

    ax = axs[0]
    colours = {ROCK: "#8c5a3c", "Coarse substrate": "#e0b878", "Mixed sediment": "#b9a07a",
               "Sand": "#f4e38c", "Sandy mud": "#a7b08f", "Seabed": "#d7dde3"}
    names = {ROCK: "Rock", "Coarse substrate": "Coarse sediment", "Mixed sediment": "Mixed sediment",
             "Sand": "Sand", "Sandy mud": "Sandy mud", "Seabed": "Not classified"}
    for name, paths in polys.items():
        for p in paths:
            ax.add_patch(PathPatch(p, fc=colours.get(name, "#d7dde3"), ec="none", zorder=1))
    ax.legend(handles=[Patch(fc=colours[k], label=names[k]) for k in colours if k in polys],
              loc="lower right", fontsize=7)
    fig.colorbar(plt.cm.ScalarMappable(cmap="Greys"), ax=ax, shrink=0.75).ax.set_visible(False)
    base_map(ax, LO, LA, land, "(a) Seabed type")

    ax = axs[1]
    im = ax.contourf(LO, LA, depth, levels=[0, 10, 15, 20, 25, 30, 35, 40, 50, 60], cmap="Blues", zorder=1)
    ax.contour(LO, LA, depth, levels=[15], colors="#c0392b", linewidths=0.9, zorder=2)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Depth below LAT (m)")
    base_map(ax, LO, LA, land, "(b) Water depth (red line: 15 m)")

    ax = axs[2]
    grid = ship_df.pivot_table(index="latitude", columns="longitude", values="density")
    im = ax.pcolormesh(grid.columns, grid.index, np.maximum(grid.to_numpy(), 0.01), norm=LogNorm(0.05, 100),
                       cmap="magma_r", shading="nearest", zorder=1)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Vessel hours per km² per month")
    base_map(ax, LO, LA, land, "(c) Shipping, all vessels, annual average")

    for ax in axs.flat:
        ax.contourf(LO, LA, outside, levels=[0.5, 1.5], colors=["white"], alpha=0.65, zorder=2)
    fig.text(0.01, 0.005, "Sources: " + SOURCES_DATA + "\nCoastline: 0 m contour of the DTM. Dashed: 15 km from the station. Maps drawn by the authors in Python (site_maps.py).",
             fontsize=7, color="#444")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig("site_criteria_maps.png", dpi=200)

    # ------------------------------------------------ figure 2: suitability
    cls = np.full(Z.shape, 4.0)
    cls[ok] = np.select([score >= BANDS[0], score >= BANDS[1], score >= BANDS[2]], [0, 1, 2], 3)
    cls[land] = 5
    area = [np.sum(cls[ok] == i) * cell_km2 for i in range(4)]
    top = score >= score.max() - 1e-9
    blat, blon = LA[ok][top].mean(), LO[ok][top].mean()

    col = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#c9c9c9", "#ece6d6"]
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.pcolormesh(lon, lat, cls, cmap=ListedColormap(col), vmin=-0.5, vmax=5.5, shading="nearest", rasterized=True)
    cs = ax.contour(LO, LA, depth, levels=[15, 30], colors="#2f5d8a", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, fmt="%d m", fontsize=7)
    base_map(ax, LO, LA, land, "", fs=8.5)
    ax.set_ylim(50.17, 50.365)
    ax.plot(blon, blat, marker="*", ms=17, color="#00441b", mec="white", zorder=7)
    ax.annotate(f"highest score\n{score.max():.2f} / 5", (blon, blat), xytext=(12, -24), textcoords="offset points",
                fontsize=9, color="#00441b", fontweight="bold")
    labels = [f"Ideal: score 4.5 or more ({area[0]:.0f} km²)", f"Good: 4.0 to 4.5 ({area[1]:.0f} km²)",
              f"Fair: 3.5 to 4.0 ({area[2]:.0f} km²)", f"Poor: under 3.5 ({area[3]:.0f} km²)",
              "Impossible: shallower than 15 m, within 2 km of the coast, or beyond 15 km"]
    handles = [Patch(fc=col[i], ec="#888", label=labels[i]) for i in range(5)]
    handles += [Line2D([], [], color="black", ls="--", lw=0.8, label="15 km from Camborne station"),
                Line2D([], [], color="#2f5d8a", lw=0.8, label="15 m and 30 m depth")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.3, framealpha=0.95)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Offshore technical suitability within 15 km of Camborne", fontsize=12)
    weights = ", ".join(f"{k} {int(round(v * 100))} %" for k, v in WEIGHTS.items())
    fig.text(0.01, 0.005, f"Score out of 5 averaged over the {FOOTPRINT_M / 1000:.2f} km around each spot, group zone study rules and weights ({weights}).\n"
             "Sources: " + SOURCES_DATA + "\nMap drawn by the authors in Python (site_maps.py).", fontsize=7, color="#444")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig("site_suitability_map.png", dpi=200)
    save_zones(cls, ok, xy)
    print("\nScoring result (Section A4.3, Appendix A.4)")
    print(f"  Highest score {score.max():.2f} / 5 at {blat:.3f} N, {-blon:.3f} W, "
          f"{farm.to_xy(blat, blon)[1] / 1000 - station[1] / 1000:.0f} km north of the station")
    i = np.where(top)[0]
    print(f"  There: mapped rock {100 * rock[i].mean():.0f} %, mean depth {np.mean([depth[ok][near[k]].mean() for k in i]):.0f} m, "
          f"coast {coast[ok][i].mean():.1f} km,")
    print(f"         shipping {np.mean([ship[near[k]].mean() for k in i]):.2f} vessel h/km2/month, "
          f"export route to Hayle {route[i].mean():.0f} km")
    print(f"  Ideal (4.5 or more) {area[0]:.0f} km2, good (4.0 to 4.5) {area[1]:.0f} km2, "
          f"fair {area[2]:.0f} km2, poor {area[3]:.0f} km2")
    print("\nOther weights (Table 15): where is the highest score?")
    for label, w in [("Study weights", WEIGHTS), ("Seabed halved (15 %)", dict(WEIGHTS, seabed=0.15)),
                     ("All criteria equal", {k: 1 for k in WEIGHTS}), ("Seabed ignored", dict(WEIGHTS, seabed=0))]:
        sc = total(w)
        top = sc >= sc.max() - 1e-9
        print(f"  {label:22s} top score {sc.max():.2f} at {LA[ok][top].mean():.3f} N, {-LO[ok][top].mean():.3f} W")
    print("\nSaved site_criteria_maps.png, site_suitability_map.png and zones.npz")
    plt.show()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # find the data files next to this script
    main()
