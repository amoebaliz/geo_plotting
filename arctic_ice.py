import netCDF4 as nc
import numpy as np

import matplotlib.pylab as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap, addcyclic
import datetime as dt

def polygon_patch(mapid,axs):
    mapid.drawcoastlines(linewidth=0)
    mapid.drawmapboundary(fill_color=[.9,.97,1])
    polys = []
    for polygon in mapid.landpolygons:
        polys.append(polygon.get_coords())

    lc = PolyCollection(polys, edgecolor='black',
         facecolor=(1,1,1), closed=False)
    axs.add_collection(lc)

def plot_polar_arctic(clon,clat,lon,lat,pclim,fclim,vardif):
    # NOTE: Not using U-point coordinates 
    # T-point method looks better w/ Basemap
    # continent placement... though maybe more accurate for
    # colormap handling (i.e., interprets coordinate info as 
    # the corners for the data)

    # FIGURE DETAILS
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
    ax = fig.add_subplot(111)

    # MAPPING SETUP
    m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l',ax=ax)
    #m = Basemap(projection='npstere',boundinglat=30,lon_0=0,ax=ax)
    # DATA PLOTTING
    X,Y = np.meshgrid(clon, clat) 
    P = m.pcolormesh(X,Y,vardif,cmap='bwr',vmin=-1, vmax=1,latlon = 'True')
    X,Y = np.meshgrid(lon, lat)
    m.contour(X,Y,pclim,[0],colors='b',linewidths=2,latlon='True')
    m.contour(X,Y,fclim,[0],colors='r',linewidths=2,latlon='True')
    polygon_patch(m,ax)

    m.drawparallels(np.arange(30.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,30.))
    m.colorbar(P)

lensdir = '/glade/p/cesmLE/CESM-CAM5-BGC-LE/'
models = ['ice','atm']
timdir = '/proc/tseries/monthly/'
vars = ['aice','ICEFRAC']

# FILE NAME STUFF
pfil_bas = '/b.e11.B20TRC5CNBDRD.f09_g16.'
ffil_bas = '/b.e11.BRCP85C5CNBDRD.f09_g16.'
fil_bas2 = ['.cice.h.aice_nh.','.cam.h0.ICEFRAC.']

mon_st = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

# YEARS IN CLIM
cyr = 25

for nv in [1]:
    # SET ARCTIC DOMAIN
    if nv == 0:
       a = 0
    else:
       a = 123

    print vars[nv] 
    # EACH MONTH
    for nm in range(12):
        print nm 
        # COLDEST, WARMEST
        # for nf in [12,24]:
        pclim = np.zeros((33,69,288)) 
        fclim = np.zeros((33,69,288))   
        clim_stor = np.zeros((33,69,288))    
        # AVERAGING OVER ALL LENS
        for nf in range(33): 
            if (nf == 0):
               pyr = 185001
            else: 
               pyr = 192001
            # HISTORICAL YEARS B20TRC5CNBDRD
            ncfil1 = lensdir + models[nv] + timdir + vars[nv] + pfil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(pyr) + '-' + str(200512)+ '.nc'
            # FUTURE YEARS BRCP85C5CNBDRD
            ncfil2 = lensdir + models[nv] + timdir + vars[nv] + ffil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(200601) + '-' + str(208012)+ '.nc' 
            ncfil3 = lensdir + models[nv] + timdir + vars[nv] + ffil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(208101) + '-' + str(210012)+ '.nc'

            #print ncfil1
            fid1 = nc.Dataset(ncfil1)
            fid2 = nc.Dataset(ncfil2)
            fid3 = nc.Dataset(ncfil3)

            time1 = fid1.variables['time'][:]
            time2 = fid2.variables['time'][:]
            time3 = fid3.variables['time'][:]
           
            nt1 = len(time1)
            nt2 = len(time2)
            nt3 = len(time3)
 
            if (nf ==0):
               if (nv == 0):
                  lat = fid1.variables['TLAT'][a:,:]
                  lon = fid1.variables['TLON'][a:,:]
                  clon = np.insert(lon,lon.shape[1],lon[:,0],axis=1)
                  clat = np.insert(lat,lat.shape[1],lat[:,0],axis=1)
               else: 
                  lat = fid1.variables['lat'][a:]
                  clat= fid1.variables['lat'][a-1:]
                  lon = fid1.variables['lon'][:] 
                  clon = np.insert(lon,lon.shape[0],lon[0],axis=0)

               # MAKE MAP CYCLIC - PCOLOR REMOVES LAST ROW/COL OF DATA
               # RESULTING IN A WHITE WEDGE AT THE LONG. EDGES
               # NOTE: not good if using contourf 
       

            # TIME BOUNDS 
            # past bounds for last 25 yrs 1920-2005
            A = np.arange(nt1-(cyr*12)+nm,nt1,12)
            # future bounds for last 5 yrs in 2006-2080
            B = np.arange(nt2-(5*12)+nm,nt2,12)
            # future bounds for all 20 yrs 2081-2100
            C = np.arange(nm,nt3,12)
 
            fclim[nf,:] = np.mean(np.concatenate((fid2.variables[vars[nv]][B,a:,:].squeeze(),\
                              fid3.variables[vars[nv]][C,a:,:].squeeze()),axis=0),axis=0)
 
            pclim[nf,:] =  np.mean(fid1.variables[vars[nv]][A,a:,:].squeeze(),axis=0)
            #SURFACE ONLY
            clim_stor[nf,:] = fclim[nf,:]-pclim[nf,:]

            #clim_stor[nf,:] = np.mean(np.concatenate((fid2.variables[vars[nv]][B,:,a:,:].squeeze(),\
            #                  fid3.variables[vars[nv]][C,:,a:,:].squeeze()),axis=0),axis=0) -      \
            #                  np.mean(fid1.variables[vars[nv]][A,:,a:,:].squeeze(),axis=0)
        tit_str = mon_st[nm] + '_avg_LENS_clim_dif_' + vars[nv]  
        plot_polar_arctic(clon,clat,lon,lat,np.mean(pclim,axis=0),np.mean(fclim,axis=0),np.mean(clim_stor,axis=0))
        #plt.show()
        plt.savefig(tit_str)
