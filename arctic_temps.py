import netCDF4 as nc
import numpy as np

import matplotlib.pylab as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap, addcyclic


def polygon_patch(mapid,axs):
    mapid.drawcoastlines(linewidth=0)
    mapid.drawmapboundary(fill_color=[.9,.97,1])
    polys = []
    for polygon in mapid.landpolygons:
        polys.append(polygon.get_coords())

    lc = PolyCollection(polys, edgecolor='black',
         facecolor=(1,1,1), closed=False)
    axs.add_collection(lc)

# DATA FILE
fid = nc.Dataset('/glade/p/cesmLE/CESM-CAM5-BGC-LE/ocn/proc/tseries/monthly/SST/b.e11.BRCP85C5CNBDRD.f09_g16.001.pop.h.SST.208101-210012.nc')
SST = fid.variables['SST'][0,:].squeeze()

lat = fid.variables['TLAT'][:]
lon = fid.variables['TLONG'][:]

# NOTE: Not using U-point coordinates 
# T-point method looks better w/ Basemap
# continent placement... though maybe more accurate for
# colormap handling (i.e., interprets coordinate info as 
# the corners for the data)

# MAKE MAP CYCLIC - PCOLOR REMOVES LAST ROW/COL OF DATA
# RESULTING IN A WHITE WEDGE AT THE LONG. EDGES
# NOTE: not good if using contourf 

lon = np.insert(lon,lon.shape[1],lon[:,0],axis=1)
lat = np.insert(lat,lat.shape[1],lat[:,0],axis=1)

# FIGURE DETAILS
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111)

# MAPPING SETUP
m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l',ax=ax)

# DATA PLOTTING
P = m.pcolormesh(lon,lat,SST,latlon=True)
polygon_patch(m,ax)

m.drawparallels(np.arange(30.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,30.))

plt.show()
