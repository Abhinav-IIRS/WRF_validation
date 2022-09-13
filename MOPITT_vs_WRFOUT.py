# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:25:29 2022

@author: Abhinav

@email: abhinavsharma@iirs.gov.in
"""

#%% Objectives and Workflow

'''
Objectives:
    
    1. Read model output and interpolate to MOPITT retrieval grids    [x]
    2. Pre-process the WRFOUT with a-priori and averaging kernel from MOPITT    [x]
    3. Read MOPITT CO mixing ratio      [x]
    4. Compare various spatial features and plotting    [ ]
    5. (Optional) Run a comparision line along Gangetic Plain   [ ]
    
Workflow:
    
    1. Load modules
    2. Load model output and extract necessary variables (lat, lon, time, CO)
    3. Load satellite output and extract necessary variables (apriori, averaging kernal, retrieved CO surface mixing ratio)
    4. Interpolate wrfout to standard MOPITT level
    5. Process wrfout using apriori and Averaging kernel
    6. comparision of wrfout with MOPITT CO
    
'''

#%% Modules load

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
import matplotlib.dates as mdates
import scipy.stats as stats
import netCDF4 as nc
import xarray as xr
import wrf
import h5py
import os
import scipy as sc
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator, FixedLocator, FixedFormatter
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy
import numpy.ma as ma
#import rioxarray
import geopandas
#from shapely.geometry import mapping
import regionmask
import xskillscore as xs
import plotly.express as px

#%% Matplotlib unviversal params

#~~> Define universal matplotlib params

from matplotlib import rcParams
rcParams['figure.figsize'] = (10.0, 6.0) #predefine the size of the figure window
rcParams['figure.dpi'] = 600 # set DPI
rcParams.update({'font.size': 14}) # setting the default fontsize for the figure
rcParams['axes.labelweight'] = 'bold' #Bold font style for axes labels
rcParams['lines.linewidth'] = 1.5
rcParams["image.cmap"] = 'viridis'
# rcParams['font.family'] = 'Arial'
#rcParams['timezone'] = "Asia/Kolkata"
from matplotlib import style
style.use('ggplot')

#%% Model output loading

# Opening model file

# using xarray
#model_fn = "wrfout_V3.8_sinked_d01_2019-12-06.nc"
model_fn = "Wrfout_V3.8_sinked_average_d01_30Nov_07Dec_2019.nc"
#model_fn = "Sample.nc"                                # may use ncks -v sel in.nc out.nc to create a subset with interested variables
#model_fn = "wrfout_v4_d01_2019-12-06.nc"
model_fl = xr.open_dataset(model_fn)

# Getting model history run time steps
model_time = model_fl.XTIME.sel()
model_time = model_time.values
model_time = pd.to_datetime(model_time)               # Makes a datimeIndex with Time Zone naive information

model_time = model_time.tz_localize("UTC")            # Gives a Time Zone aware time index
local_model_time = model_time.tz_convert("Asia/Kolkata")                    # TimeZone conversion
local_model_time = local_model_time.tz_localize(None)                       # Strip TZ awareness returns to tz_naive information but in local time
model_time = model_time.tz_localize(None)

xlat = model_fl.XLAT
xlat = xlat.values
xlat = xlat[:,0]
xlong = model_fl.XLONG
xlong = xlong.values
xlong = xlong[0,:]
'''
# wrf out coordinates:
  * XTIME    (XTIME) datetime64[ns] 2019-11-30
    XLONG    (south_north, west_east) float32 66.86 67.16 67.45 ... 98.12 98.42
    XLAT     (south_north, west_east) float32 4.442 4.442 4.442 ... 38.55 38.55
XTIME: 1, bottom_top: 36, south_north: 129, west_east: 109
'''

wrfout_CO_raw = model_fl['co'].values
wrfout_CO_raw = wrfout_CO_raw * 1000                  # converting PPMV to PPBV
#wrfout_CO_raw = wrfout_CO_raw[:,:,:,:]
#plt.pcolormesh(wrfout_CO_raw[0,0,:,:])

#%% MOPITT file loading and extraction

# --> NOTE: MOPITT CO mixing ratio is in PPBV <--
# --> NOTE: MOPITT CO total column is in molecules/cm^2 <--

# Using xarray to read HDF5 file, require explicit mentioning of groups
MOP_fn = 'MOP03JM-201912-L3V95.9.3_MonthlyAvg_Dec2019.he5'
MOP_fl = xr.open_dataset(MOP_fn, group = "HDFEOS/GRIDS/MOP03/Data Fields", engine="h5netcdf")

MOP_flh5 = h5py.File(MOP_fn)
mlon = MOP_fl['XDim'].values
mlat = MOP_fl['YDim'].values
#mpress1 = MOP_fl['Prs1'].values * 100
#mpress2 = MOP_fl['Prs2'].values * 100
# Reading groups within HDF file
# From [https://hdfeos.org/zoo/LaRC/MOP03T-20131129-L3V5.9.1.he5.py]
#[https://hdfeos.org/zoo/index_openLaRC_Examples.php#MOPITT]
#[https://asdc.larc.nasa.gov/project/MOPITT/MOP02T_6]

with h5py.File(MOP_fn, mode='r') as f:
    # Apriori CO - surface mixing ratio day
    group = f['HDFEOS/GRIDS/MOP03/Data Fields']
    dsname = 'APrioriCOSurfaceMixingRatioDay'
    apriori_sfc = group[dsname][:].T
    longname = group[dsname].attrs['long_name'].decode()
    units = group[dsname].attrs['units'].decode()
    fillvalue = group[dsname].attrs['_FillValue']

    apriori_sfc[apriori_sfc == fillvalue] = np.nan
    apriori_sfc = np.ma.masked_array(apriori_sfc, np.isnan(apriori_sfc))
    
    dsname = 'APrioriCOMixingRatioProfileDay'
    apriori_lev = group[dsname][:].T
    longname = group[dsname].attrs['long_name'].decode()
    units = group[dsname].attrs['units'].decode()
    fillvalue = group[dsname].attrs['_FillValue']

    apriori_lev[apriori_lev == fillvalue] = np.nan
    apriori_lev = np.ma.masked_array(apriori_lev, np.isnan(apriori_lev))
    
    
    # RetrievedCOSurfaceMixingRatioDay
    group = f['HDFEOS/GRIDS/MOP03/Data Fields']
    dsname = 'RetrievedCOSurfaceMixingRatioDay'
    MOP_CO_sfc = group[dsname][:].T
    longname = group[dsname].attrs['long_name'].decode()
    units = group[dsname].attrs['units'].decode()
    fillvalue = group[dsname].attrs['_FillValue']

    MOP_CO_sfc[MOP_CO_sfc == fillvalue] = np.nan
    MOP_CO_sfc = np.ma.masked_array(MOP_CO_sfc, np.isnan(MOP_CO_sfc))
    
    dsname = 'RetrievedCOMixingRatioProfileDay'
    MOP_CO_lev = group[dsname][:].T
    longname = group[dsname].attrs['long_name'].decode()
    units = group[dsname].attrs['units'].decode()
    fillvalue = group[dsname].attrs['_FillValue']

    MOP_CO_lev[MOP_CO_lev == fillvalue] = np.nan
    MOP_CO_lev = np.ma.masked_array(MOP_CO_lev, np.isnan(MOP_CO_lev))

    # RetrievalAveragingKernelMatrixDay
    
    dsname = 'RetrievalAveragingKernelMatrixDay'
    Avg_K = group[dsname][:].T
    longname = group[dsname].attrs['long_name'].decode()
    units = group[dsname].attrs['units'].decode()
    fillvalue = group[dsname].attrs['_FillValue']

    Avg_K[Avg_K == fillvalue] = np.nan
    Avg_K = np.ma.masked_array(Avg_K, np.isnan(Avg_K))
    
    # extracting pressure values
    dsname = 'Pressure'
    pressure = group[dsname][:].T
    pressure[pressure == fillvalue] = np.nan
    pressure = np.ma.masked_array(pressure, np.isnan(pressure))
    
# making full profile of apriori and Retrieved CO mixing ratio
apriori = np.empty([10,180,360])
apriori[0,:,:] = apriori_sfc
apriori[1::,:,:] = apriori_lev

MOP_CO = np.empty([10,180,360])
MOP_CO[0,:,:] = MOP_CO_sfc
MOP_CO[1::,:,:] = MOP_CO_lev


# quick look
# plt.pcolormesh(apriori), plt.colorbar()
# plt.pcolormesh(MOP_CO), plt.colorbar()

#%% Processing WRFOUT for MOPITT grid

mlon = mlon-180
mlat = mlat-90
test = mlat-xlat.min()

lat_bound_low = np.argmin(np.abs(mlat-xlat.min()))
lat_bound_high = np.argmin(np.abs(mlat-xlat.max()))

lon_bound_low = np.argmin(np.abs(mlon-xlong.min()))
lon_bound_high = np.argmin(np.abs(mlon-xlong.max()))


lat = mlat[lat_bound_low:lat_bound_high+1]
lon = mlon[lon_bound_low:lon_bound_high+1]

#%% Spatial interpolation
# interpolation on 2-d lat-lon level
# # If wrfout_CO is on 1 pressure level:
# f = sc.interpolate.interp2d(xlong,xlat,wrfout_CO_raw[0,:,:], kind = 'linear')
# wrfout_new = f(lon,lat)



# If wrfout_CO_raw is a 4-D array with shape time=1, pressure=36, lat = 129, lon = 109:
wrfout_CO_1deg = np.empty([wrfout_CO_raw.shape[0],wrfout_CO_raw.shape[1],lat.shape[0],lon.shape[0]])
for i in range(0,wrfout_CO_raw.shape[1]):
    f = sc.interpolate.interp2d(xlong,xlat,wrfout_CO_raw[0,i,:,:])
    wrfout_CO_1deg[0,i,:,:] = f(lon,lat)

# old(raw) wrfout CO on dx (30km) resolution
plt.pcolormesh(wrfout_CO_raw[0,3,:,:])
# new wrfout CO on 1-deg resolution
plt.pcolormesh(wrfout_CO_1deg[0,3,:,:])


#%% Vertical interpolation

# Interpolating WRFOUT on standard pressure levels

desired_levels = np.arange(1000,0,-100)

basepress = model_fl['PB']
pertpress = model_fl['P']

totpress = (basepress.values + pertpress.values)/100
totpress_1deg = np.empty([totpress.shape[0],totpress.shape[1],lat.shape[0],lon.shape[0]])
for i in range(0,totpress.shape[1]):
    f = sc.interpolate.interp2d(xlong,xlat,totpress[0,i,:,:])
    totpress_1deg[0,i,:,:] = f(lon,lat)

# Using WRF-Python generates unreliable result
#wrfout_CO_vert = wrf.interplevel(wrfout_CO_1deg[0,:,:,:],totpress_1deg[0,:,:,:],desired_levels, meta = False)

wrfout_CO_vert = np.empty([1,10,36,32])
for j in range(0,totpress_1deg.shape[2]):
    for i in range(0,totpress_1deg.shape[3]):
        interp = sc.interpolate.interp1d(totpress_1deg[0,:,j,i], wrfout_CO_1deg[0,:,j,i], 'nearest', fill_value="extrapolate")
        wrfout_CO_vert[0,:,j,i] = interp(desired_levels)


#%% Subsetting satellite variables
# quick look
# i = 5
# plt.pcolormesh(wrfout_CO_vert[i,:,:]), plt.title("pressure level = "+str(desired_levels[i])+" hPa")
# plt.colorbar()

# subsetting apriori for wrfout bounds
apriori_ind = apriori[:,lat_bound_low:lat_bound_high+1, lon_bound_low:lon_bound_high+1]

#plt.pcolormesh(apriori)
#plt.pcolormesh(apriori_ind)


Avg_K_ind = Avg_K[:,:,lat_bound_low:lat_bound_high+1, lon_bound_low:lon_bound_high+1]

# Avg_K_ind_avg = np.nanmean(Avg_K_ind, axis = -1)
# Avg_K_ind_avg = np.nanmean(Avg_K_ind_avg, axis = -1)

#plotting averaging kernels for all pressure at all levels (10*10)

# plt.figure(figsize = (16,12))
# for i in range(0,len(desired_levels)):    
#     plt.plot(Avg_K_ind_avg[i,:],desired_levels, label = str(desired_levels[i]))
    
# plt.gca().invert_yaxis()
# plt.legend(loc = 'upper right')
# plt.title("Averaging kernel averaged for entire India")
# plt.ylabel("Pressure \n $hPa$")
# plt.xlabel("Averaging kernel")

# # Plotting for an individual level - for all levels => 1 x 10
# plt.figure(figsize = (16,12))
# i = 0                                                 # select index for level of interest from desired_levels
# plt.plot(Avg_K_ind_avg[i,:],desired_levels, label = str(desired_levels[i]))
# plt.gca().invert_yaxis()
# plt.legend(loc = 'upper right')
# plt.title("Averaging kernel averaged for entire India for level = "+str(desired_levels[i])+' hPa')
# plt.ylabel("Pressure \n $hPa$")
# plt.xlabel("Averaging kernel")



MOP_CO_ind = MOP_CO[:,lat_bound_low:lat_bound_high+1, lon_bound_low:lon_bound_high+1]

#%% Smoothing of wrfout

def NANmatmul(A,B):
    '''
    Calculates the matrix multiplication product for arrays with NAN.
    Retruns the product as an array for i*[j x j]*k as i*k.

    Parameters
    ----------
    A : array_like
        2-D array.
    B : array_like
        2-D array.

    Returns
    -------
    Matrix multiplication product.

    '''
    sza = A.shape
    szb = B.shape
    a,b = A.shape[0],B.shape[1]
    c=np.zeros([a,b])
    for i in range(0, sza[0]):
      for k in range(0, szb[1]):
        c[i, k]=np.nansum(A[i, ]*B[:, k])
    
    return c

# Xret = Xapriori + Avg_K * (Xwrf - Xapriori)
WRF_Smooth_lev = np.empty(wrfout_CO_vert.shape)
for j in range(0, apriori_ind.shape[1]):
    for i in range(0, apriori_ind.shape[2]):
        diff = wrfout_CO_vert[0,:,j,i] - apriori_ind[:,j,i]
        diff = diff.reshape(-1,1)
        diff2 = NANmatmul(Avg_K_ind[:,:,j,i],diff)
        #diff2 = Avg_K_ind[:,:,j,i]*diff
        #diff2 = np.nansum(diff2, axis = 1)       # the dot product at end drops nans [x1y1 + x2y2 + x3*nan]
        # other options - np.ma.dot, np.einsum, np.matmul
        #diff2 = np.matmul(Avg_K_ind[:,:,j,i], diff)
        #diff2 = np.einsum('ij,j',Avg_K_ind[:,:,j,i], diff)
        diff2 = diff2[:,0]
        WRF_Smooth = apriori_ind[:,j,i] + diff2
        WRF_Smooth_lev[0,:,j,i] = WRF_Smooth #np.diagonal(WRF_Smooth, offset = 0)






fig, axes = plt.subplots(nrows = 1,ncols = 3,
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         figsize = (20,10), dpi = 300)

axes = axes.flatten()

#ax1 = fig.add_subplot(1,2,2)
norm = mcolors.TwoSlopeNorm(vmin=0.0, vmax = 300.0, vcenter=150.0)
cmap = plt.cm.magma_r
cmap_r = plt.cm.get_cmap('magma_r')
# axes[0] = plt.axes(projection=ccrs.PlateCarree())
axes[0].coastlines(facecolor = 'None')
axes[0].add_feature(cfeature.BORDERS, linestyle='dotted')
axes[0].add_geometries(shpreader.Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth = 1.0, linestyle='-.')
axes[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
levels = MaxNLocator(nbins = 9).tick_values(vmin_all, vmax_all)
cf = axes[0].contourf(xx,yy,WRF_Smooth_lev[0,PL,:,:], levels = levels, cmap=cmap_r, norm = norm, extend = 'both')
c = axes[0].contour(xx,yy,WRF_Smooth_lev[0,PL,:,:], levels = levels, colors = 'black' )
axes[0].clabel(c, inline = True, fontsize = 10, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)
cbar = plt.colorbar(cf, shrink = 0.95, pad= 0.10, ax = axes[0])
cbar.set_label(label = '$PPBV$',size = 'large')
axes[0].set_title("Smoothed WRF output CO concentration\n at "+str(desired_levels[PL])+" hPa", fontsize = 20)

#ax2 = fig.add_subplot(1,2,2)
norm = mcolors.TwoSlopeNorm(vmin=0.0, vmax = 300.0, vcenter=150.0)
cmap = plt.cm.magma_r
cmap_r = plt.cm.get_cmap('magma_r')
#axes[1] = plt.axes(projection=ccrs.PlateCarree())
axes[1].coastlines(facecolor = 'None')
axes[1].add_feature(cfeature.BORDERS, linestyle='dotted')
axes[1].add_geometries(shpreader.Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth = 1.0, linestyle='-.')
axes[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
levels = MaxNLocator(nbins = 9).tick_values(vmin_all, vmax_all)
cf = axes[1].contourf(xx,yy,MOP_CO_ind[PL,:,:], levels = levels, cmap=cmap_r, norm = norm, extend = 'both')
c = axes[1].contour(xx,yy,MOP_CO_ind[PL,:,:],  levels = levels, colors = 'black' )
axes[1].clabel(c, inline = True, fontsize= 10, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)
cbar = plt.colorbar(cf, shrink = 0.95, pad= 0.10, ax = axes[1])
cbar.set_label(label = '$PPBV$',size = 'large')

axes[1].set_title("MOPITT retrieved CO concentration\n at "+str(desired_levels[PL])+" hPa", fontsize = 20)

#ax1 = fig.add_subplot(1,2,2)
norm = mcolors.TwoSlopeNorm(vmin= 0.0, vmax = 300.0, vcenter=150.0)
cmap = plt.cm.magma_r
cmap_r = plt.cm.get_cmap('magma_r')
# axes[0] = plt.axes(projection=ccrs.PlateCarree())
axes[2].coastlines(facecolor = 'None')
axes[2].add_feature(cfeature.BORDERS, linestyle='dotted')
axes[2].add_geometries(shpreader.Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='None', edgecolor='black', linewidth = 1.0, linestyle='-.')
axes[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
levels = MaxNLocator(nbins = 9).tick_values(vmin_all, vmax_all)
cf = axes[2].contourf(xx,yy,wrfout_CO_vert[0,PL,:,:], levels = levels, cmap=cmap_r, norm = norm, extend = 'both')
c = axes[2].contour(xx,yy,wrfout_CO_vert[0,PL,:,:], levels = levels, colors = 'black' )
axes[2].clabel(c, inline = True, fontsize = 10, inline_spacing=10, fmt='%i', rightside_up=True, use_clabeltext=True)
cbar = plt.colorbar(cf, shrink = 0.95, pad= 0.10, ax = axes[2])
cbar.set_label(label = '$PPBV$',size = 'large')
axes[2].set_title("Original WRF output CO concentration\n at "+str(desired_levels[PL])+" hPa", fontsize = 20)


fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95,
                    wspace=0.10, hspace=0.1)
plt.show()

       





