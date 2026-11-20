# SEF01 Wind Energy, Group 10: code

Python code for the 270 MW offshore wind farm north of Portreath. Run every script from this folder.

## Install

```
pip install numpy pandas scipy matplotlib xlrd py_wake
```

## Getting the results

The main results come from `layout_grid.py`. It needs these to have run first:

1. `python wind.py` reads and checks the Camborne 2006 data, then fits the Weibull distribution (Sections A1 to A3, A5).
2. `python site_maps.py` scores the sea within 15 km and writes `zones.npz`, the area the turbines can use (Section A4).
3. `python layout_grid.py` gives the layout and number of turbines, with Table 6 and Figures 10 and 11 (Section A7).

`zones.npz` and the `grid_results` folder are already included. You can therefore run `layout_grid.py` straight away. It reads the saved layouts and takes a few seconds. To search all the layouts again, run `python layout_grid.py --optimise`, which takes several hours, so it's normal if you didn't get results right away.

After `layout_grid.py`:

4. `python farm.py` gives the costs, LCOE and NPV of the design (Sections B1 to B3).
5. `python sensitivity.py` gives the sensitivity analysis, Tables 11, 20 and 21 and Figures 12 to 16 (Section B4).

