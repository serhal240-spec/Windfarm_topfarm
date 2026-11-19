"""
farm.py: turbine, wake, energy and cost inputs of the offshore wind farm north of Portreath.

    python farm.py

Every other script imports its settings from here: the V236 power and thrust
curves (report Section A7, Appendix A.7), the losses, and the unit costs of
Part B (Sections B1 to B3, Appendix B.1). Run on its own, it prints the
derived values used in the report and plots the power and thrust curves.
"""

import os
import json
import math

import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

# ================================================================ files
DATA_FILE = "1 Camborne 2006.xls"
DEPTH_FILE = "emodnet_bathymetry_dtm2024_cornwall.csv"
SEABED_FILE = "emodnet_euseamap2025_eunis2019_cornwall.geojson"
SEABED_FACTOR = {"Rock or other hard substrata": 1.40, "Coarse substrate": 1.05, "Mixed sediment": 1.05,
                 "Sand": 1.00, "Sandy mud": 1.00}       # foundation cost multiplier by seabed type (Equation 4)
SEABED_OTHER = 1.05           # not classified
SEABED_SMOOTH_M = 300         # class edges blurred over 300 m

# ================================================================ turbine and energy
TURBINE = dict(name="Vestas V236-15 MW", P=15e6, D=236, hub=145)   # 145 m: blade tip 22 m above MHWS (A5)
CP, CT0 = 0.43, 0.80          # power and thrust coefficients (Appendix A.7)
CUT_IN, CUT_OUT = 3.0, 31.0   # m/s (cut-out: Vestas V236-15.0 MW brochure)
WAKE_K = 0.0325               # Gaussian wake growth rate, k* = 0.38 I + 0.004 with I = 7.5 %
LOSSES = 0.95 * 0.97 * 0.98   # availability, electrical, other

# ================================================================ site
STATION = (50.2179, -5.3273)                  # Camborne Met Office station (lat, lon)
BOX = (50.315, 50.355, -5.375, -5.310)        # its centre is the origin of the local x, y coordinates (m)
LIMIT_KM = 15                                 # brief: within 15 km of the station
DEPTH_MIN, DEPTH_MAX = 15, 50                 # m, fixed bottom monopiles
COAST_MIN_KM = 2
# export route to the Hayle landfall, round Godrevy Point
WAYPOINTS = [(50.258, -5.410), (50.220, -5.430), (50.196, -5.418)]
ROUTE_FACTOR = 14.5 / 13.33                   # straight lines to the length of the group's cable route

# ================================================================ costs (sources in report Section B1 and Appendix B.1)
DEV_PER_KW = 216              # development, incl. consenting (Arup 2025, medium case)
TURBINE_PER_KW = 870          # Vestas average selling price Q2 2026, EUR 1.0m/MW x 0.87 GBP/EUR
INSTALL_PER_KW = 130          # installation } Arup 2025 Table 23 (3 % and 15 % of 3,760 GBP/kW)
OTHER_PER_KW = 649            # other        } x 1.15 small farm uplift (our assumption)
FOUND_PER_KW = 769            # monopile at 30 m on sand: NREL Cost of Wind Energy Review 2024
FOUND_PER_M = 0.014           # +1.4 % per metre deeper than 30 m (Dicorato et al. 2011)
ARRAY_PER_KM = 0.88e6         # 66 kV array cable, supplied and installed (Kikuchi and Ishihara 2023)
EXPORT_PER_KM = 1.2e6         # 66 kV export cable per km, per cable (Kikuchi and Ishihara 2023)
EXPORT_CABLE_MW = 90          # one 66 kV export cable per 90 MW (Prysmian)
LANDFALL = 3.27e6             # per cable: drilled landfall under the Hayle dunes
ONSHORE_CABLE = 1.7 * 0.5e6   # per cable: 1.7 km at 0.5M GBP/km
NETWORK = 5.0e6               # grid works per export cable
SUBSTATION = 12.0e6           # onshore substation, +50 % per extra export cable
DCO = 0.75e6                  # Development Consent Order fees above 100 MW (Planning Inspectorate 2025/26)
OPEX_PER_KW = 46.5 + 8.6 + 0.8    # GBP/kW/yr: O&M and insurance (Arup 2025), distribution network charges (NGED)
RENT = 0.02 * 87.8            # seabed rent, GBP/MWh: Crown Estate 2 % of revenue at the AR7 price of 87.8 GBP/MWh
DECOM_PER_KW = 255            # paid at year 20 (DESNZ peer review 2025; Arup 86, BVGA 426)
RATE, LIFE = 0.08, 20         # brief
CRF = RATE / (1 - (1 + RATE) ** -LIFE)       # capital recovery factor, 0.1019

# Large farms: offshore substation and 220 kV export above 300 MW (Section A7.3)
DIRECT_MAX_MW = 300
OSS_1GW = 282e6               # 1 GW HVAC offshore substation (BVG Associates guide)
ONSHORE_SUB_1GW = 30e6        # onshore substation, electrical (BVG Associates guide)
SCALE_EXP = 0.7               # substation cost ~ size^0.7 (our assumption)
HV_CABLE_MW = 300             # per 220 kV export cable (BVG Associates guide)
HV_SUBSEA_PER_KM = 2.0e6      # 220 kV subsea cable, installed (our assumption)
HV_ONSHORE_PER_KM = 1.5e6     # 220 kV onshore cable (our assumption)

# TNUoS (NESO Final TNUoS Tariffs 2026/27, zone 27 West Devon and Cornwall), Equation (6)
TNUOS_YRS = -13.295891        # Year Round Shared tariff, GBP/kW/yr (multiplied by the annual load factor)
TNUOS_ADJ = -2.476760         # adjustment tariff, GBP/kW/yr
TNUOS_ALF = 0.38              # net load factor of the design (Section A7.4)
TNUOS_PER_KW = TNUOS_YRS * TNUOS_ALF + TNUOS_ADJ   # -7.5 GBP/kW/yr; negative means the farm is paid
TNUOS_MIN_MW = 100            # farms below 100 MW are embedded and pay no TNUoS

RHO = 1.225
D = TURBINE["D"]
AREA = math.pi * D ** 2 / 4
V_RATED = (2 * TURBINE["P"] / (RHO * AREA * CP)) ** (1 / 3)

# ================================================================ geometry
CENTRE = ((BOX[0] + BOX[1]) / 2, (BOX[2] + BOX[3]) / 2)


def to_xy(lat, lon):
    """Metres east (x) and north (y) of CENTRE."""
    x = (np.asarray(lon) - CENTRE[1]) * 111320 * math.cos(math.radians(CENTRE[0]))
    y = (np.asarray(lat) - CENTRE[0]) * 110540
    return np.stack([x, y], axis=-1)


def load_depth(path=DEPTH_FILE):
    """Sea points with their depth, and land points (for the coast)."""
    d = pd.read_csv(path, skiprows=[1])
    xy = to_xy(d["latitude"].to_numpy(), d["longitude"].to_numpy())
    elev = d["elevation"].to_numpy()
    land = np.isnan(elev) | (elev >= 0)
    return xy[~land], -elev[~land], xy[land]


def to_latlon(pts):
    """Latitude and longitude of points given in metres from CENTRE (inverse of to_xy)."""
    la = CENTRE[0] + pts[:, 1] / 110540
    lo = CENTRE[1] + pts[:, 0] / (111320 * math.cos(math.radians(CENTRE[0])))
    return la, lo


def depth_function():
    """Water depth (m) at any point, interpolated from the EMODnet grid."""
    d = pd.read_csv(DEPTH_FILE, skiprows=[1])
    lat, lon = np.sort(d.latitude.unique()), np.sort(d.longitude.unique())
    Z = d.pivot_table(index="latitude", columns="longitude", values="elevation").reindex(index=lat, columns=lon).to_numpy()
    grid = RegularGridInterpolator((lat, lon), np.nan_to_num(-Z, nan=0.0), bounds_error=False, fill_value=0.0)
    return lambda pts: grid(np.c_[to_latlon(pts)])


def seabed_function(zone, step=150.0, smooth=True):
    """Foundation multiplier for the seabed at any point (EUSeaMap 2025, Equation 4).
    smooth=True blurs the class edges over SEABED_SMOOTH_M (map uncertainty, and smooth for the optimiser);
    smooth=False gives the raw map class, used to check which turbines sit on mapped rock."""
    x = np.arange(zone[:, 0].min() - 1000, zone[:, 0].max() + 1000, step)
    y = np.arange(zone[:, 1].min() - 1000, zone[:, 1].max() + 1000, step)
    X, Y = np.meshgrid(x, y)
    la, lo = to_latlon(np.c_[X.ravel(), Y.ravel()])
    pts = np.c_[lo, la]
    f = np.full(len(pts), SEABED_OTHER)
    for feat in json.load(open(SEABED_FILE))["features"]:
        b = feat["bbox"]
        m = (pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) & (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3])
        if not m.any():
            continue
        idx = np.where(m)[0]
        inside = np.zeros(len(idx), bool)
        for poly in feat["geometry"]["coordinates"]:
            i = Path(np.array(poly[0])).contains_points(pts[idx])
            for hole in poly[1:]:
                i &= ~Path(np.array(hole)).contains_points(pts[idx])
            inside |= i
        f[idx[inside]] = SEABED_FACTOR.get(feat["properties"]["substrate"], SEABED_OTHER)
    F = gaussian_filter(f.reshape(X.shape), SEABED_SMOOTH_M / step) if smooth else f.reshape(X.shape)
    grid = RegularGridInterpolator((y, x), F, bounds_error=False, fill_value=SEABED_OTHER,
                                   method="linear" if smooth else "nearest")
    return lambda p: grid(np.c_[p[:, 1], p[:, 0]])


def export_km(start_xy):
    """Length of the export route from a point at sea to the Hayle landfall, km."""
    pts = [start_xy] + [to_xy(*w) for w in WAYPOINTS]
    return ROUTE_FACTOR * sum(np.hypot(*(b - a)) for a, b in zip(pts[:-1], pts[1:])) / 1000


# ================================================================ turbine and wakes
def power(v):
    """V236 power curve, W: constant Cp up to 15 MW, between cut-in and cut-out."""
    p = np.minimum(0.5 * RHO * AREA * CP * v ** 3, TURBINE["P"])
    return np.where((v >= CUT_IN) & (v <= CUT_OUT), p, 0.0)


def thrust(v):
    """Thrust coefficient: CT0 below rated speed, CT0 (v_r / v)^3 above it."""
    ct = np.where(v <= V_RATED, CT0, CT0 * (V_RATED / np.maximum(v, 1e-6)) ** 3)
    return np.where((v >= CUT_IN) & (v <= CUT_OUT), ct, 0.0)


# ================================================================ costs
KW = TURBINE["P"] / 1e3       # kW per turbine


def export_cost(mw, km):
    """Equation (5): 66 kV export straight to shore, no offshore substation (up to 300 MW)."""
    cables = math.ceil(mw / EXPORT_CABLE_MW)
    cost = cables * (EXPORT_PER_KM * km + LANDFALL + ONSHORE_CABLE + NETWORK)
    cost += SUBSTATION * (1 + 0.5 * (cables - 1))
    if mw > 100:
        cost += DCO
    return cost


def export_cost_any(mw, km):
    """Export system of any size: Equation (5) up to 300 MW, offshore substation and 220 kV above."""
    if mw <= DIRECT_MAX_MW:
        return export_cost(mw, km)
    cables = math.ceil(mw / HV_CABLE_MW)
    size = (mw / 1000) ** SCALE_EXP
    return ((OSS_1GW + ONSHORE_SUB_1GW) * size + DCO
            + cables * (HV_SUBSEA_PER_KM * km + LANDFALL + 1.7 * HV_ONSHORE_PER_KM + NETWORK))


def array_cable_km(pts):
    """Shortest cable network joining every turbine and the collection point (middle of the farm), km."""
    nodes = np.vstack([pts, pts.mean(axis=0)])
    return minimum_spanning_tree(cdist(nodes, nodes)).sum() / 1000


def capex(pts, depth, seabed):
    """Capital cost breakdown, GBP (Table of Section B1). Foundations: Equation (4), one per turbine."""
    kw = KW * len(pts)
    return {
        "Development": DEV_PER_KW * kw,
        "Turbines": TURBINE_PER_KW * kw,
        "Installation and other": (INSTALL_PER_KW + OTHER_PER_KW) * kw,
        "Foundations": (FOUND_PER_KW * KW * (1 + FOUND_PER_M * (np.asarray(depth) - 30)) * np.asarray(seabed)).sum(),
        "Array cables": ARRAY_PER_KM * array_cable_km(pts),
        "Export and substations": export_cost_any(kw / 1e3, export_km(pts.mean(axis=0))),
    }


def opex(n):
    """Yearly costs before seabed rent, GBP: O&M, insurance, network charges and the TNUoS credit."""
    kw = KW * n
    tnuos = TNUOS_PER_KW * kw if kw >= TNUOS_MIN_MW * 1e3 else 0.0
    return OPEX_PER_KW * kw + tnuos


def annual_cost(pts, depth, seabed):
    """Numerator of Equation (8), GBP/yr: annualised capex and decommissioning plus net opex."""
    kw = KW * len(pts)
    decom = DECOM_PER_KW * kw * (1 + RATE) ** -LIFE * CRF
    return CRF * sum(capex(pts, depth, seabed).values()) + opex(len(pts)) + decom


def lcoe(pts, depth, seabed, e_net):
    """Equation (8), GBP/MWh."""
    return annual_cost(pts, depth, seabed) / e_net + RENT


# ================================================================ main
STRIKE = 113.0                # administrative strike price, GBP/MWh (Section B3)
TO_STRIKE_BASIS = 0.9241      # GDP deflator ratio used in Section B3 to put the LCOE on the strike price basis
LAYOUTS = os.path.join("grid_results", "grid_layouts.csv")   # written by layout_grid.py


def design(n):
    g = pd.read_csv(LAYOUTS)
    g = g[g.farm_turbines == n]
    pts = g[["x_m", "y_m"]].to_numpy(float)
    e_net = g["energy_MWh_per_yr"].sum() * LOSSES
    return pts, g["depth_m"].to_numpy(), g["seabed_factor"].to_numpy(), e_net


def main():
    if not os.path.exists(LAYOUTS):
        raise SystemExit("Run layout_grid.py first: it writes grid_results/grid_layouts.csv")
    n = 18
    pts, depth, seabed, e_net = design(n)
    mw = KW * n / 1e3
    print(f"Design: {n} x {TURBINE['name']} = {mw:.0f} MW, hub {TURBINE['hub']} m")
    print(f"Net AEP {e_net / 1e3:.1f} GWh/yr, load factor {e_net / (mw * 8760) * 100:.1f}%")

    parts = capex(pts, depth, seabed)
    total = sum(parts.values())
    print("\nCapital cost (Section B1)")
    for k, val in parts.items():
        print(f"  {k:26s} {val / 1e6:7.1f} M GBP {val / (mw * 1e3):7.0f} GBP/kW {val / total * 100:6.1f}%")
    print(f"  {'Total':26s} {total / 1e6:7.1f} M GBP {total / (mw * 1e3):7.0f} GBP/kW")
    print(f"  Array cable {array_cable_km(pts):.1f} km, export route {export_km(pts.mean(axis=0)):.1f} km, "
          f"{math.ceil(mw / EXPORT_CABLE_MW)} export cables")

    kw = mw * 1e3
    print("\nYearly costs (Section B2)")
    print(f"  O&M                        {46.5 * kw / 1e6:6.1f} M GBP")
    print(f"  Insurance                  {8.6 * kw / 1e6:6.1f} M GBP")
    print(f"  Network charges            {0.8 * kw / 1e6:6.1f} M GBP")
    print(f"  TNUoS credit ({TNUOS_PER_KW:.2f} GBP/kW) {TNUOS_PER_KW * kw / 1e6:5.1f} M GBP")
    print(f"  Net opex                   {opex(n) / 1e6:6.2f} M GBP")
    print(f"  Seabed rent ({RENT:.2f} GBP/MWh)  {RENT * e_net / 1e6:5.1f} M GBP")
    print(f"  Total                      {(opex(n) + RENT * e_net) / 1e6:6.1f} M GBP")
    decom = DECOM_PER_KW * kw
    print(f"  Decommissioning, year {LIFE}   {decom / 1e6:6.2f} M GBP (once)")

    df = (1 + RATE) ** -LIFE
    a_capex, a_decom = CRF * total, CRF * decom * df
    cost = a_capex + opex(n) + a_decom
    L = cost / e_net + RENT
    print("\nLevelised cost (Section B3, Equation 8)")
    print(f"  CRF {CRF:.5f}, discount factor at year {LIFE}: {df:.4f}")
    print(f"  Annualised capex {a_capex / 1e6:.2f}, opex {opex(n) / 1e6:.2f}, "
          f"decommissioning {a_decom / 1e6:.2f}: total {cost / 1e6:.2f} M GBP/yr")
    print(f"  Cost per MWh {cost / e_net:.2f} + rent {RENT:.2f} = LCOE {L:.2f} GBP/MWh = {L / 10:.2f} p/kWh")

    print(f"\nNPV at the strike price of {STRIKE:.0f} GBP/MWh (Section B3, Equation 10)")
    for m in (18, 12):
        p, d, s, e = design(m)
        l = round(lcoe(p, d, s, e), 1)                  # LCOE to one decimal, as quoted in Section B3
        l_b = round(l * TO_STRIKE_BASIS, 2)
        print(f"  {m} turbines ({KW * m / 1e3:.0f} MW): LCOE {l:.1f} x {TO_STRIKE_BASIS} = {l_b:.2f} GBP/MWh, "
              f"AEP {e / 1e3:.1f} GWh, NPV {(STRIKE - l_b) * e / CRF / 1e6:+.1f} M GBP")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # find the data files next to this script
    main()
