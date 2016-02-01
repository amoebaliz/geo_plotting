import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import shapefile as shp

# OBJECTIVE: Map of the coral triangle with maximum DHW during 1998
# IDEAS: 1) Alternate Calculation of DHW
#        2) Inset map of the Philippines
#        3) Convert time to dates for the 'WHEN' plot

def convert_time(time):
    # NOTE: TIME VARIABLES MUST BE IN DAYS SINCES (1900,1,1,0,0)
    ref = dt.datetime(1900,1,1,0,0)
    date_vals = np.zeros(len(time))
    for nt in range(len(time)):
        day_time = ref + dt.timedelta(days=np.float(time[nt]))
        date_vals[nt] = pltd.date2num(day_time)
    return date_vals

def polygon_patch(mapid,axs):
    mapid.drawcoastlines(linewidth=0)
    polys = []
    for polygon in mapid.landpolygons:
        polys.append(polygon.get_coords())

    lc = PolyCollection(polys, edgecolor='black',
         facecolor='0.8', closed=False)
    axs.add_collection(lc)

def plt_shp_coord(sf,m_fin):
    m0 = Basemap(projection = 'cea')
    x_0,y_0 = m0(0,0)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        x = np.array(x)-x_0
        y = np.array(y)+y_0
        lons,lats = m0(x,y,inverse=True)
        m_fin.plot(lons,lats,'.1',lw=6)
        m_fin.plot(lons,lats,'paleturquoise',lw=1.5)
##############################################

# FILE DETAILS
dhw_file = '/data/external/P1/Data/CORTAD/Version4/cortadv4_TSA_DHW_coral_1998-1999.nc'
fid = nc.Dataset(dhw_file)

cmap_file = '/home/frederic/python/cmap/dhw_noaa.cmap'

shp_file = '/t3/workdir/liz/external_data/CT_ATLAS/Coral_Triangle_Boundary/Coral_Triangle_Boundary_Line.shp'
sfid = shp.Reader(shp_file)

# EXTRACT VARIABLES
time = fid.variables['time'][:60]
plot_time = convert_time(time)
lats = fid.variables['lat'][:]
lons = fid.variables['lon'][:]

DHW = fid.variables['dhw'][:60,:,:]
DHW = np.ma.masked_where(DHW<0,DHW)
DHW_max = np.max(DHW,axis=0)
DHW_max_when = np.argmax(DHW,axis=0)

cmap = np.loadtxt(cmap_file)
cmap = cmap/256.
dhw_noaa = colors.ListedColormap(cmap)

# FIGURE DETAILS
fig = plt.figure(figsize=(15,9))
ax1 = fig.add_subplot(111)

cmin=0
cmax=16
# main map domain
lats1 = [-22.,25.]
lons1 = [90.,170.]
# inset map domain
lats2 = [12.4,15]
lons2 = [120.,122.5]

# PRIMARY CT MAP
m = Basemap(llcrnrlon=lons1[0],llcrnrlat=lats1[0],urcrnrlon=lons1[1],urcrnrlat=lats1[1],resolution='i')
m.pcolormesh(lons,lats,DHW_max,vmin=cmin,vmax=cmax,cmap=dhw_noaa,latlon=True)
polygon_patch(m,ax1)
plt_shp_coord(sfid,m)

m.drawmeridians([100,120,140,160], labels=[0,0,0,1], fmt='%d', fontsize=18)
m.drawparallels([-20,0,20], labels=[1,0,0,0], fmt='%d', fontsize=18)

cb_ticks = np.linspace(cmin,cmax,5)
clb = m.colorbar(ticks=cb_ticks)
for t in clb.ax.get_yticklabels():
    t.set_fontsize(18)

# INSET VIP MAP
axins = zoomed_inset_axes(ax1, 9, loc=1)

#axins.set_xlim(lons2[0], lons2[1])
#axins.set_ylim(lats2[0], lats2[1])

m2 = Basemap(llcrnrlon=lons2[0],llcrnrlat=lats2[0],urcrnrlon=lons2[1],urcrnrlat=lats2[1],resolution='f')
m2.pcolormesh(lons,lats,DHW_max,vmin=0,vmax=16,cmap=dhw_noaa,latlon=True)
polygon_patch(m2,axins)
m2.drawmapboundary(linewidth=3.0)

mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0",lw =3)

plt.figure()
plt.pcolor(DHW_max_when,vmin=20,vmax=40)
plt.colorbar()

plt.show()
