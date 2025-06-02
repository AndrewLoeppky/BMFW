#!/usr/bin/env python
# coding: utf-8

# # bmwflib
# 
# library of functions common to all bmwf plotters

from herbie import Herbie
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "heavy"
from toolbox import EasyMap, pc
import cartopy.crs as ccrs
import cartopy.feature as feature
import pandas as pd

from matplotlib.patches import Rectangle
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pytz
import os, shutil
import json

import warnings
# supress because removal of old datafiles is handled outside herbie
warnings.filterwarnings('ignore')


def clear_directory(path):
    """
    clears all files in a directory
    """
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
    return None


def get_var(run, model, fxx, var, lev):
    """
    helper function to initialize herbie objects in a list comp
    """
    ds = Herbie(
        run,
        model=model,
        fxx=fxx,
        variable=var,
        level=lev,
    ).xarray()

    # if the variable name is unknown, assign it
    if "unknown" in ds:
        ds = ds.rename({"unknown": var})

    return ds


def plot_cities(ax):
    """
    plots and labels cities on map based on JSON file path input
    """
    with open("../config/cities.json", "r") as f:
        cities = pd.DataFrame.from_dict(
            json.load(f), orient="index", columns=["Latitude", "Longitude"]
        )

    ax.scatter(
        x=cities.Longitude,
        y=cities.Latitude,
        color="yellow",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
    )
    for index, row in cities.iterrows():
        ax.text(
            row.Longitude,
            row.Latitude + 0.1,
            index,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            alpha=0.7,
            color="white",
        )

    return None


def make_basemap(ax):
    """
    generates a basemap matching ACMWF style
    """
    ax.add_feature(feature.LAND, color="#C3A97C")
    ax.add_feature(feature.STATES)
    ax.add_feature(feature.OCEAN, color="lightslategrey")
    ax.add_feature(feature.COASTLINE)
    ax.add_feature(feature.LAKES, color="lightslategrey")
    ax.add_feature(feature.RIVERS, color="lightslategrey")

    return None

