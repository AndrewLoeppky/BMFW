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


def find_pressure_centres(pres, radius, delrad):
    """
    Andrew's experimental function to locate and label pressure
    centres.

    inputs:
    -------
    pres: (np.array) the pressure field to operate on.

    radius: (pixels)    the characteristic length scale of the centre.
                     suggest 90pi for high pressure on GDPS data,
                     50pi for low pressure on GDPS.
    delrad: (pixels) step to take when searching past false centre
                     candidates.
                     suggest 50pi for high pressure GPDS,
                     30pi for low pressure GDPS.
    returns:
    --------
    centres (list of tuples) the pressure centres
    """
    # fill lists with H and L coordinates (pixels)
    centres = []

    # get the aspect ratio
    aspect = pres.shape[1] / pres.shape[0]

    # get the index of the max value and save it
    ymax, xmax = np.unravel_index(np.nanargmax(pres), pres.shape)
    centres.append((int(xmax), int(ymax)))

    # mask out all values within radius of the max
    x = np.arange(0, np.shape(pres)[1])
    y = np.arange(0, np.shape(pres)[0])
    mask = ((x[np.newaxis, :] - xmax) / aspect) ** 2 + (
        (y[:, np.newaxis] - ymax)
    ) ** 2 < radius**2
    pres[mask] = np.nan

    n = 0  # number of centres marked
    m = 0  # number of false centres located
    while (n < 10) and (m < 2):
        # get the index of the new max value
        ymaxp, xmaxp = np.unravel_index(np.nanargmax(pres), pres.shape)

        # check to see if the new max is on the border of the masked area (+ 5 pixels)
        falseextrema = bool(
            ((xmaxp - xmax) / aspect) ** 2 + ((ymax - ymax)) ** 2 < (radius + 5) ** 2
        )

        # if the new centre is next to the old one, ignore it, increase radius by
        # delrad, and try again until a true centre is found
        if falseextrema:
            m += 1
            mask = ((x[np.newaxis, :] - xmaxp) / aspect) ** 2 + (
                (y[:, np.newaxis] - ymaxp)
            ) ** 2 < (radius + delrad) ** 2
            pres[mask] = np.nan
            ymaxp, xmaxp = np.unravel_index(np.nanargmax(pres), pres.shape)

        # once a true centre is found, save it and iterate n
        centres.append((int(xmaxp), int(ymaxp)))
        n += 1

        # mask the new centre
        mask = ((x[np.newaxis, :] - xmaxp) / aspect) ** 2 + (
            (y[:, np.newaxis] - ymaxp)
        ) ** 2 < radius**2
        pres[mask] = np.nan

        # reset the max indices
        xmax = xmaxp
        ymax = xmaxp

    return centres


def plot_pressure(da, ax, extent, radius=100, delrad=50, levels=range(900, 1100, 2)):
    """
    plots pressure contours, highs, lows on an existing axis

    da (xarray dataarray (e.g. ds.prmsl)) the pressure field

    ax (matplotlib axis) the axis on which to plot

    extent (str) one of "MWF", "Alberta Rockies", "South Coast",
                 "South Interior", "BC-AB"

    levels (range) levels to contour plot
    """
    ht = ax.contour(
        da.longitude,
        da.latitude,
        da,
        colors="k",
        transform=pc,
        levels=levels,
    )
    ax.clabel(ht, inline=True, fontsize=9)

    # find high and low pressures
    highs = find_pressure_centres(np.array(da), radius, delrad)
    lows = find_pressure_centres(np.array(da) * -1, radius/2, delrad/2)

    # truncate pressure center to map extent and plot
    with open("../config/map_extents.json", "r") as f:
        map_extents = json.load(f)
    ext = map_extents[extent]
    for low in lows:
        lat = da[low[1], low[0]].latitude
        lon = da[low[1], low[0]].longitude
        if (ext[0] < lon) & (lon < ext[1]) & (ext[2] < lat) & (lat < ext[3]):
            ax.text(lon, lat, "L", size=40, transform=pc, ha="center", va="center")

    for high in highs:
        lat = da[high[1], high[0]].latitude
        lon = da[high[1], high[0]].longitude
        if (ext[0] < lon) & (lon < ext[1]) & (ext[2] < lat) & (lat < ext[3]):
            ax.text(lon, lat, "H", size=40, transform=pc, ha="center", va="center")

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
        
    # slice out only the domain we care about
    if model == "gdps":
        ds = ds.sel(longitude=slice(-195, -97), latitude=slice(30, 73))
    elif model == "rdps":
        ds = ds.sel(x=slice(220, 500), y=slice(200, 520))
    elif model == "hrdps":
        ds = ds.sel(x=slice(0, 900), y=slice(100, 1150))

    return ds


def make_figure(extent="MWF"):
    """
    Generates a figure using cartopy. Extent is default 
    the wide MWF view. set to any key in map_extents.json.
    """
    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()
    ax = fig.add_subplot(
        1,
        1,
        1,
        position=(0, 0, 1, 1),
        projection=ccrs.LambertConformal(
            central_longitude=-123,
            central_latitude=54.0,
        ),
    )
    
    # add a basemap
    ax.add_feature(feature.LAND, color="#C3A97C")
    ax.add_feature(feature.STATES)
    ax.add_feature(feature.OCEAN, color="lightslategrey")
    ax.add_feature(feature.COASTLINE)
    ax.add_feature(feature.LAKES, color="lightslategrey")
    ax.add_feature(feature.RIVERS, color="lightslategrey")
    
    # set the extent from map_extents.json
    with open("../config/map_extents.json", "r") as f:
        map_extents = json.load(f)
    ax.set_extent(map_extents[extent], ccrs.PlateCarree())
    
    # lock the aspect ratio at 1:1
    ext =  ax.get_extent()
    ax.set_aspect(abs((ext[1] - ext[0]) / (ext[3] - ext[2])))

    return fig, ax


def plot_cities(ax, extent="MWF"):
    """
    plots and labels cities on map based on JSON file path input
    """
    with open("../config/cities.json", "r") as f:
        cities = pd.DataFrame.from_dict(
            json.load(f), orient="index", columns=["Latitude", "Longitude"]
        )

    # clip cities to map extent
    with open("../config/map_extents.json", "r") as f:
        map_extents = json.load(f)
    ext = map_extents[extent]
    cities = cities[
        (ext[0] < cities.Longitude)
        & (cities.Longitude < ext[1])
        & (ext[2] < cities.Latitude)
        & (cities.Latitude < ext[3])
    ]

    # do the plot
    ax.scatter(
        x=cities.Longitude,
        y=cities.Latitude,
        color="yellow",
        edgecolor="black",
        s=75,
        transform=ccrs.PlateCarree(),
    )
    for index, row in cities.iterrows():
        ax.text(
            row.Longitude,
            row.Latitude + 0.15,
            index,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="bottom",
            alpha=0.8,
            color="white",
            weight="light",
            size=12,
        )

    return None

def make_title(fig, ax, ds, title, offset=0):
    """
    Creates title, date, model run, shaded regions of plot.
    offset shifts position of model run (eg 00 Z) left or right.
    """
    # shade out top and left regions
    ax.add_patch(
        Rectangle(
            (0, 0),
            0.12,
            0.88,
            transform=ax.transAxes,
            color="slategray",
            alpha=0.8,
        )
    )
    ax.add_patch(
        Rectangle((0, 0.88), 1, 1, transform=ax.transAxes, color="slategray", alpha=0.8)
    )

    # title
    fig.text(
        0.403,
        0.938,
        title,
        color="black",
        size=20,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    fig.text(
        0.4,
        0.94,
        title,
        color="white",
        size=20,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    # timestamps
    the_date = (
        pd.Timestamp(ds.valid_time.values).tz_localize("UTC").tz_convert("America/Vancouver")
    )
    fmt_date = the_date.strftime("%a. %d %b. %y \n%H:%M %p %Z")
    fig.text(0.703, 0.938, fmt_date, color="black", size=20, ha="center", va="center", transform=ax.transAxes,)
    fig.text(0.7, 0.94, fmt_date, color="white", size=20, ha="center", va="center", transform=ax.transAxes,)

    # model run identifier in bottom left
    fig.text(
        0.062,
        0.013,
        f"{ds.model.upper()} {pd.Timestamp(ds.time.values).strftime("%H")} Z",
        color="black",
        size=12,
        ha="center",
        transform=ax.transAxes,
    )

    fig.text(
        0.06,
        0.015,
        f"{ds.model.upper()} {pd.Timestamp(ds.time.values).strftime("%H")} Z",
        color="white",
        size=12,
        ha="center",
        transform=ax.transAxes,
    )

    return None

