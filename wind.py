"""
wind.py: the wind resource at Camborne (report sections A1 to A3 and A5).

    python wind.py

Reads the 2006 hourly record, removes the missing hours, checks the rest,
fits the Weibull distribution by the method of moments, draws the wind rose
and scales the wind to hub height with the power law.
farm.py and the layout scripts import load() and hub_speed() from here.
"""

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "1 Camborne 2006.xls"
KNOT_TO_MS = 0.5144       # 1 kn = 0.5144 m/s (used by every script, so Parts A and B share one wind record)
ERROR_CODE = 32768        # the Met Office marks missing hours with this number
RHO = 1.225               # air density, kg/m3

# Wind at hub height: power law (Hellmann)
H_REF = 10.0              # m, height of the Camborne anemometer
ALPHA = 0.16              # Hellmann exponent, neutral air over a flat open coast (Section A5)
HUB = 145.0               # m, hub height of the V236 (Section A5)


# ---------------------------------------------------------------- data
def read_raw(path=DATA_FILE):
    raw = pd.read_excel(path).iloc[:, :6]
    raw.columns = ["day", "month", "year", "hour", "knots", "dir"]
    raw["knots"] = pd.to_numeric(raw["knots"], errors="coerce")
    raw["dir"] = pd.to_numeric(raw["dir"], errors="coerce")
    return raw


def load(path=DATA_FILE):
    """Read the hourly data and remove the missing hours."""
    df = read_raw(path)
    bad = (df["knots"] == ERROR_CODE) | df["knots"].isna() | df["dir"].isna()
    print(f"{len(df)} hours in the file, {bad.sum()} missing, {(~bad).sum()} kept")
    df = df[~bad].reset_index(drop=True)
    df["v10"] = df["knots"] * KNOT_TO_MS
    return df


def quality_checks(path=DATA_FILE):
    """The checks of Section A1 and Appendix A.1."""
    raw = read_raw(path)
    missing = (raw["knots"].isna() | (raw["knots"] == ERROR_CODE) | raw["dir"].isna()).to_numpy()
    blocks, start = [], None                      # runs of consecutive missing hours
    for i, m in enumerate(missing):
        if m and start is None:
            start = i
        if not m and start is not None:
            blocks.append((start, i - start))
            start = None
    if start is not None:
        blocks.append((start, len(missing) - start))
    time = pd.Timestamp("2006-01-01") + pd.to_timedelta(np.arange(len(raw)), unit="h")  # one row per hour
    print("Missing hours, by block:")
    for i, n in blocks:
        print(f"  {time[i]:%d %b %H:00}: {n} h")
    good = raw.loc[~missing, "knots"]
    print(f"Range: {good.min():.0f} to {good.max():.0f} kn; highest {good.max() * KNOT_TO_MS:.1f} m/s "
          f"at {time[good.idxmax()]:%H:00 on %d %b}")
    print(f"Largest hour to hour change: {good.diff().abs().max():.0f} kn")
    runs = (good != good.shift()).cumsum()
    print(f"Longest run of an identical speed: {good.groupby(runs).size().max()} h")
    print(f"Calm hours: {(good == 0).sum()}")


# ---------------------------------------------------------------- Weibull
def weibull_fit(v):
    """Method of moments, empirical form (Justus et al. 1978). Returns shape k and scale c."""
    mu, sigma = v.mean(), v.std(ddof=1)
    k = (sigma / mu) ** -1.086
    c = mu / math.gamma(1 + 1 / k)
    return k, c


def weibull_pdf(x, k, c):
    return (k / c) * (x / c) ** (k - 1) * np.exp(-(x / c) ** k)


# ---------------------------------------------------------------- hub height
def hub_speed(v10, hub):
    """Power law: v_hub = v_10 * (h_hub / h_ref) ^ alpha."""
    return v10 * (hub / H_REF) ** ALPHA


# ---------------------------------------------------------------- wind rose
def sectors(df, n=12):
    """Hours with a direction (calm hours left out) and their 30 degree sector, 0 = north."""
    d = df[df["knots"] > 0]
    s = (np.round(d["dir"] / (360 / n)) % n).astype(int)
    return d, s


def rose_table(df, n=12):
    d, s = sectors(df, n)
    above9 = d["v10"] > 9
    t = pd.DataFrame({
        "sector_deg": np.arange(n) * (360 // n),
        "share_hours_%": [(s == i).mean() * 100 for i in range(n)],
        "mean_speed": [d["v10"][s == i].mean() for i in range(n)],
        "share_above_9ms_%": [(above9 & (s == i)).sum() / above9.sum() * 100 for i in range(n)],
    })
    return t.round(2), len(d)


# ---------------------------------------------------------------- plots
def plot_weibull(ax, v, k, c):
    bins = np.arange(0, v.max() + 1, 1.0)
    ax.hist(v, bins=bins, density=True, color="#9ecae1", edgecolor="white", label="2006 data")
    x = np.linspace(0.01, v.max(), 200)
    ax.plot(x, weibull_pdf(x, k, c), color="#d1495b", lw=2,
            label=f"Weibull k = {k:.3f}, c = {c:.3f} m/s")
    ax.set_xlabel("Wind speed at 10 m (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Wind speed distribution, Camborne 2006")
    ax.legend()


def plot_rose(ax, df):
    n = 12
    width = 2 * np.pi / n
    d, s = sectors(df, n)
    bands = [(0, 4), (4, 8), (8, 12), (12, 100)]
    colours = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    bottom = np.zeros(n)
    for (lo, hi), col in zip(bands, colours):
        share = np.array([((s == i) & (d["v10"] >= lo) & (d["v10"] < hi)).sum()
                          for i in range(n)]) / len(d) * 100
        ax.bar(np.arange(n) * width, share, width=width * 0.95,
               bottom=bottom, color=col, edgecolor="white",
               label=f"{lo} to {hi} m/s" if hi < 100 else f"over {lo} m/s")
        bottom += share
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Wind rose, Camborne 2006 (% of hours)")
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=8)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # find the data files next to this script
    quality_checks()
    df = load()
    v = df["v10"].to_numpy()
    k, c = weibull_fit(v)
    print(f"Mean speed at 10 m: {v.mean():.3f} m/s, standard deviation {v.std(ddof=1):.3f} m/s")
    print(f"Weibull fit (method of moments): k = {k:.3f}, c = {c:.3f} m/s")
    print(f"Mean power density at 10 m: {0.5 * RHO * np.mean(v ** 3):.0f} W/m2")

    table, n_dir = rose_table(df)
    print(f"\nWind rose: {n_dir} hours with a direction")
    print(table.to_string(index=False))
    top = table[table["sector_deg"].isin([180, 210, 240])]
    print(f"180 to 240 deg: {top['share_hours_%'].sum():.1f}% of hours, "
          f"{top['share_above_9ms_%'].sum():.1f}% of hours above 9 m/s")

    f = hub_speed(1, HUB)
    print(f"\nHub height {HUB:.0f} m, alpha {ALPHA}: factor {f:.3f}, "
          f"mean {v.mean() * f:.2f} m/s, c = {c * f:.2f} m/s (k unchanged)")

    fig = plt.figure(figsize=(13, 5.5))
    plot_weibull(fig.add_subplot(1, 2, 1), v, k, c)
    plot_rose(fig.add_subplot(1, 2, 2, projection="polar"), df)
    fig.tight_layout()
    plt.show()
