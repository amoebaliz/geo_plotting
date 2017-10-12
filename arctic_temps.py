import netCDF4 as nc
import numpy as np

import matplotlib.pylab as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap


def polygon_patch(mapid,axs):
    mapid.drawcoastlines(linewidth=0)
    mapid.drawmapboundary(fill_color=[.9,.97,1])
    polys = []
    for polygon in mapid.landpolygons:
        polys.append(polygon.get_coords())

    lc = PolyCollection(polys, edgecolor='black',
         facecolor=(1,1,1), closed=False)
    axs.add_collection(lc)


# FIGURE DETAILS
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal')

# MAPPING COMMANDS
m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l',ax=ax)
# DATA PLOTTING
#P = m.pcolormesh(lon,lat,mask,vmin=.5,vmax=.75,edgecolors='face',cmap='Blues',zorder=map_order)
polygon_patch(m,ax)
#P.cmap.set_under('white')
#P.cmap.set_over([.9,.97,1])
print np.arange(30.,91.,30.)
m.drawparallels(np.arange(30.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,30.))
#m.drawmeridians([124,125], labels=[0,0,1,0], fmt='%d', fontsize=18)
#m.drawparallels([10,11], labels=[1,0,0,0], fmt='%d', fontsize=18)

plt.show()
