# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:50:15 2022

@author: Abhinav

@email: abhinavsharma@iirs.gov.in
        abhinaviirs@gmail.com
"""

import numpy as np
import geopandas
import regionmask

lon_mosaic = np.load('/home/abhinavj/FINN_2022/Emit_NParrays/lon_mosaic.npy')
lat_mosaic = np.load('/home/abhinavj/FINN_2022/Emit_NParrays/lat_mosaic.npy')
shp_fn = '/home/abhinavj/FINN_2022/Emit_NParrays/WRF_Domain/WRF_Domain.shp'
shp = geopandas.read_file(shp_fn)
mask = regionmask.mask_geopandas(shp.geometry,lon_mosaic,lat_mosaic) 
mask_value = mask.values
np.save("/home/abhinavj/FINN_2022/Emit_NParrays/WRF_Domain/WRF_Domain_mask.npy")


#%% Applying GDAL mask


def shapefilemask_fromGDAL(shapefilename, min_lat, min_lon, max_lat, max_lon ,nlons, nlats, maskvalue = 1):
    '''
    Makes a mask layer for all points within a shapefile.
    Requires extent of data to be masked, returns mask layer of same dimension with supplied mask value.
    
    Reference:
        https://gis.stackexchange.com/questions/16837/turning-shapefile-into-mask-and-calculating-mean
        
    Parameters
    ----------
    shapefilename : str
        Name of shapefile with full path as string.
    maskvalue : int, optional
        Value for mask. The default is 1.
    min_lat : float32
        Minimum latitude extent for data to be masked.
    min_lon : float32
        Minimum longitude extent for data to be masked.
    max_lat : float32
        Maximum latitude extent for data to be masked.
    max_lon : float32
        Maximum longitude extent for data to be masked.
    nlons : int
        no. of longitudes.
    nlats : TYPE
        no. of latitudes.

    Returns
    -------
    mask_arr : array-like
        A 2-D array of size ncols x nrows with mask values at shapefile points.

    '''
    
    from osgeo import gdal, ogr
    
    ncols, nrows = [nlons, nlats]
    xmin,ymin,xmax,ymax = [min_lon,min_lat,max_lon,max_lat]
    
    xres=(xmax-xmin)/float(ncols)
    yres=(ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)
    
    
    src_ds = ogr.Open(shapefilename)
    src_lyr = src_ds.GetLayer()
    
    
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)
    dst_ds.SetGeoTransform(geotransform)
    
    err = gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue])
    
    dst_ds.FlushCache()
    
    mask_arr=dst_ds.GetRasterBand(1).ReadAsArray()
    
    return mask_arr[::-1,:]
    