# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:52:37 2022

@author: Abhinav

@email: abhinavsharma@iirs.gov.in
"""
#%% Introduction : Objectives, Workflow
'''
OBJECTIVES:
    1. Read Excel files from CPCB stations and parse them for desired stations and dates
    2. Load model output for selected region, date
    3. Compare the model output and observation 
    4. Calculate basic statistical values (bias, error etc.)
    5. Plot the timeseries
    
Optional objs:
    1. Use of Dask for quick loading and processing
    2. Automation of reading script
'''
'''
WORKFLOW:
    1. Load important modules
    
    >>--|*:*|--[NOTE : Model data is in UTC, CPCB obs are in IST]--|*:*|--<<

    2. Open model dataset and obtain run time information for excel parsing
    3. Open excel files for multiple stations and get them for ONE GAS - Multiple Stations Average format
    4. Conversion of Observed values to PPBV
    5. Preparation of 4-d mask using shapefile
    6. Get model output for multiple species in individual np.arrays and conversion to PPBV
    7. Plotting 
    
    
'''


###############################################################################
#%% Import Required modules

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
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader
#import cartopy
import numpy.ma as ma
#import rioxarray
import geopandas
#from shapely.geometry import mapping
import regionmask
import xskillscore as xs
import plotly.express as px

#%% Matplotlib unviversal params
# #'''
# #~~> Define universal matplotlib params

from matplotlib import rcParams
rcParams['figure.figsize'] = (10.0, 6.0) #predefine the size of the figure window
rcParams['figure.dpi'] = 600 # set DPI
rcParams.update({'font.size': 14}) # setting the default fontsize for the figure
rcParams['axes.labelweight'] = 'bold' #Bold font style for axes labels
rcParams['lines.linewidth'] = 1.5
#rcParams['timezone'] = "Asia/Kolkata"
from matplotlib import style
style.use('ggplot')
# #'''

#%% Model output loading
###############################################################################
# Opening model file

# using xarray
#model_fn = "wrfout_V3.8_sinked_d01_2019-12-06.nc"
model_fn = "Wrfout_V3.8_sinked_timeseries_d01_30Nov_07Dec_2019.nc"
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

#%% Alternative way to define array (require model start time)

#Model_start_time = model_time[0]                     #~~> takes value from model output
Model_start_time = datetime(2019,11,30,00,00,00)      #~~> Require  manual feeding
Model_run_time = (np.array([Model_start_time + timedelta(hours=i) for i in range(len(model_time))])).astype(datetime)

Model_start_time_IST = datetime(2019,11,30,5,30,00)      #~~> Require  manual feeding
Model_run_time_IST = (np.array([Model_start_time_IST + timedelta(hours=i) for i in range(len(model_time))])).astype(datetime)


#%% Model load : alter method
# using netCDF4 and wrf-python
#nc1 = nc.Dataset(model_fn)
# op_NO = wrf.getvar(nc1,'no',timeidx=wrf.ALL_TIMES)     # extract variable of choice for all time steps
# #op_NO[3,0,:,:].plot()                                 # visualize sample
# latws, lonws = wrf.latlon_coords(op_NO)
# cart_proj = wrf.get_cartopy(op_NO)
# latw = wrf.to_np(latws)
# lonw = wrf.to_np(lonws)
# # Finding indices for lat-lon of interest using wrf-python.ll_to_xy
# # From [https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.ll_to_xy.html#wrf.ll_to_xy]
#(latW,LonW) = wrf.ll_to_xy(nc1,28.64,77.21)


#%% Basic Looping Params

Species = ["PM2.5","PM10","NO","NO2","NOx","CO","O3","SO2"]

Stations = ["ITO","IGI_AirportT3","RKPuram","Anand_Vihar"]
xlsx_suffix = "_DELHI_CPCB_2019hourly.xlsx"

# Sample load to assign From Date
Obs_all = pd.read_excel("ITO_DELHI_CPCB_2019hourly.xlsx", header = 16, usecols="A:J",na_values=['None'])
Obs_all['From Date'] = pd.to_datetime(Obs_all['From Date'],format='%d-%m-%Y %H:%M')
Obs_all['To Date'] = pd.to_datetime(Obs_all['To Date'],format='%d-%m-%Y %H:%M')

# Creating a IST time average of From Date to To Date information in observation
IST_time = []
for i in range(0, len(Obs_all["From Date"])):
    t1 = pd.Timestamp(Obs_all['From Date'].reset_index(drop = True)[i])
    t2 = pd.Timestamp(Obs_all['To Date'].reset_index(drop = True)[i])

    t3 = t1+(t2-t1)/2
    pd.to_datetime(t3, format='%Y-%m-%d %H:%M:%S')
    IST_time.append(t3)

(IST_time[i].to_pydatetime() for i in range(0,len(IST_time)))

Obs_all.insert(2,'IST',IST_time)
Obs_stn3 = Obs_all[(Obs_all['IST'] >= local_model_time[0]) & (Obs_all['IST'] <= local_model_time[-1])]
del Obs_all   

IST_time_sel = Obs_stn3["IST"].reset_index(drop = True)


Obs_PM2_5 = pd.DataFrame()
Obs_PM10 = pd.DataFrame()
Obs_NO = pd.DataFrame()
Obs_NO2 = pd.DataFrame()
Obs_NOx = pd.DataFrame()
Obs_SO2 = pd.DataFrame()
Obs_CO = pd.DataFrame()
Obs_O3 = pd.DataFrame()

Obs_PM2_5['Model UTC'], Obs_PM2_5['IST'] = model_time, IST_time_sel
Obs_PM10['Model UTC'], Obs_PM10['IST'] = model_time, IST_time_sel
Obs_NO['Model UTC'], Obs_NO['IST'] = model_time, IST_time_sel
Obs_NO2['Model UTC'], Obs_NO2['IST'] = model_time, IST_time_sel
Obs_NOx['Model UTC'], Obs_NOx['IST'] = model_time, IST_time_sel
Obs_SO2['Model UTC'], Obs_SO2['IST'] = model_time, IST_time_sel
Obs_CO['Model UTC'], Obs_CO['IST'] = model_time, IST_time_sel
Obs_O3['Model UTC'], Obs_O3['IST'] = model_time, IST_time_sel

Species_df = [Obs_PM2_5,Obs_PM10,Obs_NO,Obs_NO2,Obs_NOx,Obs_CO,Obs_O3,Obs_SO2]


for j in range(0,len(Species_df)):
    for i in range(0,len(Stations)):
        stationname = Stations[i] + xlsx_suffix
        interfile = pd.read_excel(stationname, header = 16, usecols="A:J",na_values=['None'])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("||-- Open excel file for",Species[j]+" -> Station"+" ",stationname+" --||")
        print("                         ~~~              ~~~~~~~~~~~~~")
        interfile["From Date"] = pd.to_datetime(interfile["From Date"],format='%d-%m-%Y %H:%M')
        interfile["To Date"] = pd.to_datetime(interfile["To Date"],format='%d-%m-%Y %H:%M')
        interfile.insert(2,'IST',IST_time)
        #Species_df[j]['Model UTC'] = model_time                            # defined previously
        #Species_df[j]['IST'] = IST_time_sel                                # defined previously
        interfile = interfile[(interfile['IST'] >= local_model_time[0]) & (interfile['IST'] <= local_model_time[-1])]
        Species_df[j] = pd.concat([Species_df[j], interfile[Species[j]].reset_index(drop=True)], axis = 1)
    Species_df[j]['mean'] = Species_df[j].mean(axis = 1, numeric_only = True)
    Species_df[j] = Species_df[j].drop([Species[j]], axis = 1)
    Species_df[j].columns = ['Model UTC', 'IST', Species[j]]

Obs_PM2_5,Obs_PM10,Obs_NO,Obs_NO2,Obs_NOx,Obs_CO,Obs_O3,Obs_SO2 = (Species_df[j] for j in range(0,len(Species_df)))


#%% Obs data preparation
# units note:-
''' CPCB data units:
PM2.5	ug/m3		
PM10	ug/m3	
NO	    ug/m3
NO2	    ug/m3	
NOx	    ppb                  ** ppb
SO2		ug/m3
CO	    mg/m3                ** mg
Ozone	ug/m3
Benzene	ug/m3
NH3     ug/m3
Temp    C
RH      %

PM2.5 (ug/m3)	PM10 (ug/m3)	NO (ug/m3)	NO2 (ug/m3)	NOx (ppb)	SO2 (ug/m3)	CO (mg/m3)	Ozone (ug/m3)	Temp (degree C)	RH (%)

Conversion between ppb and ug/m3 depends on temp and pressure 
Assuming 25 deg C and 1 atm, following conversion factors may be used
	1 ppb =
NO	    1.25      ug/m3
NO2	    1.88      ug/m3	
SO2		2.62      ug/m3
CO	    1.145     ug/m3
Ozone	1.96(2.0) ug/m3
Benzene	3.19      ug/m3
NH3     0.7       ug/m3
'''
Obs_NO["NO"] = Obs_NO["NO"]/1.25
Obs_NO2["NO2"] = Obs_NO2["NO2"]/1.88
Obs_CO["CO"] = (Obs_CO["CO"]*1000)/1.145
Obs_O3["O3"] = Obs_O3["O3"]/1.96
Obs_SO2["SO2"] = Obs_SO2["SO2"]/2.62
Species_df = [Obs_PM2_5,Obs_PM10,Obs_NO,Obs_NO2,Obs_NOx,Obs_CO,Obs_O3]#,Obs_SO2]

#%% masking
xlat = model_fl.XLAT
xlat = xlat.values
xlat = xlat[:,0]
xlong = model_fl.XLONG
xlong = xlong.values
xlong = xlong[0,:]

ind_shp = "C:\\Users\\abhinavs\\Documents\\CPCB_obs\\IND_adm/IND_adm1.shp"
shp = geopandas.read_file(ind_shp)                    # read as geopandas dataframe
delhi_shp = shp[shp['NAME_1'].str.contains("Delhi")]  # extract state of interest
# using module regionmask to return values inside shapefile and NANs outside
mask = regionmask.mask_geopandas(delhi_shp.geometry,xlong,xlat) 
# -->Warning<--
#values in mask get assigned a value equal to index of state require division
#mask.plot()                                          # for visualization
mask_value = (mask.values)/9

# making 4d mask for all dataset

dummy = model_fl["no"]

mask_4d = np.empty(dummy.shape)
for i in range(0,mask_4d.shape[0]):
    for j in range(0,mask_4d.shape[1]):
        mask_4d[i,j,:,:] = mask_value[:,:]

del dummy
#%% Model data processing
model_species = ['PM2_5_DRY', 'PM10', 'no', 'no2', 'co', 'o3']#, 'so2']

model_PM2_5 = np.empty(1)
model_PM10 = np.empty(1)
model_NO = np.empty(1)
model_NO2 = np.empty(1)
model_CO = np.empty(1)
model_O3 = np.empty(1)
#model_SO2 = np.empty(1)

model_array = [model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3]#, model_SO2]

for i in range(0,len(model_array)):
    model_array[i]=model_fl[model_species[i]]
    model_array[i]=model_array[i].values

model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3 = (model_array[i]*mask_4d for i in range(0,len(model_array)))

model_array = [model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3]#, model_SO2]
model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3 = (np.nanmean(model_array[i], axis = -1) for i in range(0,len(model_array)))

model_array = [model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3]#, model_SO2]
model_PM2_5, model_PM10, model_NO, model_NO2, model_CO, model_O3 = (np.nanmean(model_array[i], axis = -1) for i in range(0,len(model_array)))

model_PM2_5 = model_PM2_5[:,0]
model_PM10 = model_PM10[:,0]
model_NO = model_NO[:,0]*1000
model_NO2 = model_NO2[:,0]*1000
model_CO = model_CO[:,0]*1000
model_O3 = model_O3[:,0]*1000
model_NOx = model_NO + model_NO2

model_array = [model_PM2_5, model_PM10, model_NO, model_NO2, model_NOx, model_CO, model_O3]#, model_SO2]
Species = ["PM2.5","PM10","NO","NO2","NOx","CO","O3"]#,"SO2"]
#%% Plotting


date_format = mdates.DateFormatter('%H:%M \n%d-%m-%y\n')

fig = plt.figure(figsize=(20,14))
#fig.subplots_adjust(hspace = 0.25, wspace = 0.1)
fig.suptitle("\nModel output vs CPCB observation over Delhi", fontsize = 18)
a = 4
b = 2
for i in range(0, len(Species)):
    plt.subplot(a,b,i+1)
    plt.plot(local_model_time,model_array[i], label='Model')
    plt.plot(local_model_time,Species_df[i][Species[i]], label = 'Observation')
    plt.gca().xaxis.set_major_formatter(date_format)
    #plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    plt.title(Species[i])
    #plt.ylabel("Mixing ratio (PPBV)")
    plt.xlabel("Time")
    if (i == 0 or i == 1):
        print("Aerosol Found")
        plt.ylabel("Aerosol dry mass\n $\mu g/m^3$")
    else:
        plt.ylabel("Mixing ratio\n $PPBV$")
    plt.legend(loc = 'upper left')
fig.subplots_adjust(hspace = 0.70, wspace = 0.2)



mod_PBLH = model_fl.PBLH.sel()
mod_PBLH = mod_PBLH.values

plt.pcolormesh(mod_PBLH[10,:,:]), plt.colorbar()


mask_3d = np.empty(mod_PBLH.shape)

for i in range(0,mod_PBLH.shape[0]):
    mask_3d[i,:,:] = mask_value[:,:]


delhi_PBLH = mod_PBLH*mask_3d
#plt.pcolormesh(delhi_PBLH[10,:,:]), plt.colorbar()


delhi_PBLH = np.nanmean(delhi_PBLH, axis = -1)        # along longitude,
delhi_PBLH = np.nanmean(delhi_PBLH, axis = -1)        # along latitude, only time dim remain


plt.figure(figsize = (12,8), dpi =600)
plt.plot(local_model_time[0:72], delhi_PBLH[0:72])
plt.gca().xaxis.set_major_formatter(date_format)
plt.ylabel("PBL height \n$m$")
plt.xlabel("Time (IST)")
plt.title("PBL height variation over Delhi")

plt.figure(figsize = (12,8), dpi =600)
plt.plot(local_model_time[0:72], model_NOx[0:72])
plt.gca().xaxis.set_major_formatter(date_format)
plt.ylabel("NOx concentrationt \n$PPBV$")
plt.xlabel("Time IST")
plt.title("NOx concentration over Delhi (WRF output)")

plt.figure(figsize = (12,8), dpi =600)
plt.plot(local_model_time[0:72], Obs_NOx['NOx'][0:72])
plt.gca().xaxis.set_major_formatter(date_format)
plt.ylabel("NOx concentrationt \n$PPBV$")
plt.xlabel("Time IST")
plt.title("NOx concentration over Delhi (CPCB Observation)")

###############################################################################
#%% Checking for time zone of CBCP Obs

# # Conversion UTC to local time
# # From [https://thispointer.com/convert-utc-datetime-string-to-local-time-in-python/]
# from datetime import tzinfo
# from dateutil import tz
# import pytz

# dt_utc = (Model_run_time[i].replace(tzinfo = pytz.UTC) for i in range(0,len(Model_run_time)))

# emp = []
# local_time = np.empty(Model_run_time.shape)
# for i in range(0,len(Model_run_time)):
#     dt_utc = Model_run_time[i].replace(tzinfo = pytz.UTC)
#     local = tz.tzlocal()
#     dt_local = dt_utc.astimezone(local)
#     #local_time[i] = dt_local
#     emp.append(dt_local)


# test_df = Obs_stn3[['From Date','To Date','NOx']]
# test_df = test_df.reset_index(drop = True)
# test_df["IST"] = emp

# plt.figure(figsize = (16,6))
# plt.title("Original Time")
# plt.plot(test_df["To Date"][0:48],test_df["NOx"][0:48])
# #plt.gca().xaxis.set_major_formatter(date_format_days)
# plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
# plt.grid()

# plt.figure(figsize = (16,6))
# plt.title("IST Time")
# plt.plot(test_df["IST"][0:48],test_df["NOx"][0:48])
# #plt.gca().xaxis.set_major_formatter(date_format_days)
# plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
# plt.grid()

# +++++++++++++++++++++
# +  -->> NOTE <<--   +
# +                   +
# Key diurnal features match with original data time => CPCB obs data is in IST
# since peaks show at 8 AM and PM thus matching with local time not UTC



#%% Deterministic Statistical measures (NAN-friendly)
# Basic statistical measures

# O = Observation timeseries, M = Model o/p timeseries
'''
Measures to calculate:
    1. Correlation coeff (R)
    2. Normalized mean bias (NMB)
    3. Normalized mean error (NME)
    
pearsonR = cov(M,O)/[std(M)*std(O)]
NMB = [sigma(M-O)/sigma(O)] * 100%
NME = [sigma(abs(M-O))/sigma(O)] * 100%
acceptable if NME<35%
'''

def CorrelationCoef(O,M):
    '''
    Calculates Pearson Correlation Coefficient for two timeseries.
    Expects the timeseries to be 1-D arrays of equal length, best suited for \n ndarray or its derivatives (pd.Dataframe, xarray.values)

    Parameters
    ----------
    O : (N,) array_like
        Observation value timeseries.
    M : (N,) array_like
        Model output timeseries.

    Returns
    -------
    CorrCoef : float
               Pearson R - Correlation Coeficient 
    '''
    return (ma.corrcoef(ma.masked_invalid(M), ma.masked_invalid(O)))[1,0]


def Normalized_Mean_Bias(O,M):
    '''
    Calculates the Normalized Mean Bias of two timeseries.
    Expects the timeseries are 1-D arrays of equal length
    NMB = [sigma(M-O)/sigma(O)] * 100%
    Parameters
    ----------
    O : (N,) array_like
        Observation value array of 1 dimension
    M : (N,) array_like
        Model output timeseries of 1 dimension
        
    Returns
    -------
    NMB : float
          Normalized Mean Bias
    '''
    return (np.nansum(M-O)/np.nansum(O)) * 100

def Normalized_Mean_Error(O,M):
    '''
    Calculates the Normalized Mean Error of two timeseries.
    Expects the timeseries are 1-D arrays of equal length
    NME = [sigma(abs(M-O))/sigma(O)] * 100%
    Parameters
    ----------
    O : (N,) array_like
        Observation value array of 1 dimension
    M : (N,) array_like
        Model output timeseries of 1 dimension
        
    Returns
    -------
    NME : float
          Normalized Mean Error
    '''
    return (np.nansum(np.abs(M-O))/np.nansum(O)) * 100

def Rsquared(O,M):
    '''
    Calculates Coefficient of Determination or "R squared", written as R2 or r2.
    R2 values typically vary from 1 to 0, however, negative values are possible for poor model performance.
    If R2 = 0 => Model is able to predict the mean of Observed values
    If R2 < 0 => Model cannot even predict mean Obs, or the mean of the data provides a better fit to the outcomes than do the fitted function values.
    

    Parameters
    ----------
    O : (N,) array_like
        Observation value array of 1 dimension.
    M : (N,) array_like
        Model output timeseries of 1 dimension.

    Returns
    -------
    Rsquared : float
    Coefficient of Determination R-squared
    '''
    SSres = np.nansum((O-M)**2)
    SStot = np.nansum((O-np.nanmean(O))**2)
    return(1-(SSres/SStot))    
    
def Error(O,M):
    '''
    Calulates Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) of two 1-D arrays representing observations and predicted values.
    MAE = mean(abs(O-M))

    Parameters
    ----------
    O : (N,) array_like
        Observation value array of 1 dimension.
    M : (N,) array_like
        Model output timeseries of 1 dimension.

    Returns
    -------
    MAE : Mean Absolute Error
    MSE : Mean Squared Error
    RMSE : Root Mean Squared Error
    '''
    MAE = np.nanmean(np.abs(O-M))
    MSE = np.nanmean((O-M)**2)
    RMSE = MSE**0.5
    RMSPE = ((np.nanmean(((O-M)/np.nanmean(O))**2))**0.5)*100
    return MAE, MSE, RMSE, RMSPE


PearsonR = []
NMB = []
NME = []
R2 = []
MAE = []
MSE = []
RMSE = []
RMSPE = []

for i in range(0,len(Species)):
    pr = CorrelationCoef(Species_df[i][Species[i]], model_array[i])
    nmb = Normalized_Mean_Bias(Species_df[i][Species[i]], model_array[i])
    nme = Normalized_Mean_Error(Species_df[i][Species[i]], model_array[i])
    r2 = Rsquared(Species_df[i][Species[i]], model_array[i])
    mae, mse, rmse, rmspe = Error(Species_df[i][Species[i]], model_array[i])
    PearsonR.append(pr)
    NMB.append(nmb)
    NME.append(nme)
    R2.append(r2)
    MAE.append(mae)
    MSE.append(mse)
    RMSE.append(rmse)
    RMSPE.append(rmspe)

# # From [https://stats.stackexchange.com/questions/12900/when-is-r-squared-negative]

# ## making numpy array to xr.dataset or xr.datarray
# Obs_xr = xr.DataArray(
#     data = Obs_NO['NO'],
#     dims = ["time"],
#     coords=dict(
#         time=model_time,
#         reference_time = model_time[0],
#         ),
#     attrs = dict(
#         description = "NO mixing ratio from Observation",
#         units = "PPBV",
#         ),
#     )

# Mod_xr = xr.DataArray(
#     data = model_NO,
#     dims = ["time"],
#     coords=dict(
#         time=model_time,
#         reference_time = model_time[0],
#         ),
#     attrs = dict(
#         description = "NO mixing ratio from WRF-Chem output",
#         units = "PPBV",
#         ),
#     )

# # coefficient of determination = 1-(residual sum of squares)/(total sum of squares)
# COD = xs.r2(Obs_xr, Mod_xr, skipna = True)

#%% Writing text file

Metrices_txt = open(r"Deterministic_metrices_result.txt","w+")
Header = ["Species"+" "*15+"Pearson R"+" "*15+"Norm. Mean Bias"+" "*15+"Norm. Mean Error"+" "*15+"COD (R2)"+" "*15+"MAE"+" "*15+"MSE"+" "*15+"RMSE"+" "*15+"RMSPE (%)"]
Metrices_txt.writelines(Header)
Metrices_txt.writelines('\n'+'_'*200+'\n')

for i in range(0,len(Species)):    
    text = (Species[i]+' '*20+'{0:05.3f}'+' '*25+'{1:05.3f}'+' '*22+'{2:05.3f}'+' '*20+'{3:05.3f}'+' '*14+'{4:05.3f}'+' '*12+'{5:05.3f}'+' '*12+'{6:05.3f}'+' '*14+'{6:05.3f}\n').format(PearsonR[i],NMB[i],NME[i],R2[i],MAE[i],MSE[i],RMSE[i],RMSPE[i])
    Metrices_txt.write(text)

Metrices_txt.close()
