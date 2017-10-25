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

def plot_polar_arctic(lon,lat,SST):
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

    # DATA PLOTTING
    P = m.pcolormesh(lon,lat,SST,cmap='bwr',vmin=-5, vmax=5,latlon=True)
    polygon_patch(m,ax)

    m.drawparallels(np.arange(30.,91.,30.))
    m.drawmeridians(np.arange(-180.,181.,30.))
    plt.colorbar(P)

lensdir = '/glade/p/cesmLE/CESM-CAM5-BGC-LE/'
models = ['ocn', 'ocn','ocn','ice','atm']
timdir = '/proc/tseries/monthly/'
vars = ['SST','TEMP','SALT','aice','ICEFRAC']

# FILE NAME STUFF
pfil_bas = '/b.e11.B20TRC5CNBDRD.f09_g16.'
ffil_bas = '/b.e11.BRCP85C5CNBDRD.f09_g16.'
fil_bas2 = ['.pop.h.SST.','.pop.h.TEMP.','.pop.h.SALT.','.cice.h.aice_sh.','.cam.h0.ICEFRAC.']

# ARCTIC DOMAIN
a = 230

# YEARS IN CLIM
cyr = 25

for nv in range(len(vars)):
 
    # EACH MONTH
    for nm in range(1):
        print nm 
        # COLDEST, WARMEST
        # for nf in [12,24]:
    
        # AVERAGING OVER ALL LENS
        for nf in range(2): 
            if (nf == 0):
               pyr = 185001
               fyr = 131 
            else: 
               pyr = 192001
               fyr = 60
            # HISTORICAL YEARS B20TRC5CNBDRD
            ncfil1 = lensdir + models[nv] + timdir + vars[nv] + pfil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(pyr) + '-' + str(200512)+ '.nc'
            # FUTURE YEARS BRCP85C5CNBDRD
            ncfil2 = lensdir + models[nv] + timdir + vars[nv] + ffil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(200601) + '-' + str(208012)+ '.nc' 
            ncfil3 = lensdir + models[nv] + timdir + vars[nv] + ffil_bas + \
                     str(nf+1).zfill(3) + fil_bas2[nv] + str(208101) + '-' + str(210012)+ '.nc'

            print ncfil1
            fid1 = nc.Dataset(ncfil1)
            fid2 = nc.Dataset(ncfil2)
            fid3 = nc.Dataset(ncfil3)

            time1 = fid1.variables['time'][:]
            time2 = fid2.variables['time'][:]
            time3 = fid3.variables['time'][:]
 
            if (nf ==0):
               lat = fid1.variables['TLAT'][a:,:]
               lon = fid1.variables['TLONG'][a:,:]

               # MAKE MAP CYCLIC - PCOLOR REMOVES LAST ROW/COL OF DATA
               # RESULTING IN A WHITE WEDGE AT THE LONG. EDGES
               # NOTE: not good if using contourf 
       
               clon = np.insert(lon,lon.shape[1],lon[:,0],axis=1)
               clat = np.insert(lat,lat.shape[1],lat[:,0],axis=1)

            # TIME BOUNDS 
            # future bounds for last 5 yrs
            A = np.arange((70*12)+ nm,(75*12),12)
            # future bounds for all 20 yrs
            B = np.arange(nm,20*12,12)
            # past bounds for last 25 yrs
            C = np.arange((fyr*12)+nm,(fyr+25)*12,12)
            print C
            print time1[:4], dt.datetime(1,1,1)+dt.timedelta(days=time1[0])
            print time1[-1], dt.datetime(1,1,1)+dt.timedelta(days=time1[-1])
            print dt.datetime(1,1,1)+dt.timedelta(days=time2[0])
            print dt.datetime(1,1,1)+dt.timedelta(days=time2[-1])
            print dt.datetime(1,1,1)+dt.timedelta(days=time3[0])
            print dt.datetime(1,1,1)+dt.timedelta(days=time3[-1])
            #for nt in range(len(C)):
            #    print dt.datetime(1,1,1)+dt.timedelta(days=time1[C[nt]])
            #print A
            #for nt in range(len(A)):
            #    print dt.datetime(1,1,1)+dt.timedelta(days=time2[A[nt]])
            #print B
            #for nt in range(len(B)):
            #    print dt.datetime(1,1,1)+dt.timedelta(days=time3[B[nt]])


            #clim_dif = np.mean(np.concatenate((fid2.variables[vars[nv]][A,:,a:,:].squeeze(),\
            #           fid3.variables[vars[nv]][B,:,a:,:].squeeze()),axis=0),axis=0) -      \
            #           np.mean(fid3.variables[vars[nv]][C,:,a:,:].squeeze(),axis=0)

        #plot_polar_arctic(lon,lat,clim_dif)

#plt.show()
