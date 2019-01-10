# this is a Python script that takes in binary output from TRACMASS,
# reads it, and plots an animation

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

def outline_mask(mapid,mask_img,val,x0,y0,x1,y1):
    mapimg = (mask_img == val)
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    l = []
    v = []
    # horizonal segments
    for p in zip(*hor_seg):
        v.append((plons[p[0]+1,p[1]],plats[p[0]+1,p[1]]))
        v.append((plons[p[0]+1,p[1]+1],plats[p[0]+1,p[1]+1]))

        l.append((np.nan,np.nan))
        v.append((np.nan,np.nan))
    #vertical segments
    for p in zip(*ver_seg):
        l.append((plons[p[0],p[1]+1],plats[p[0],p[1]+1]))
        l.append((plons[p[0]+1,p[1]+1],plats[p[0]+1,p[1]+1]))

        l.append((np.nan, np.nan))
        v.append((np.nan, np.nan))

    segments = np.array(l)
    vip_segments = np.array(v)
    print vip_segments.shape
    mapid.plot(segments[:,0], segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+2)
    mapid.plot(vip_segments[:,0], vip_segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+3)

# -------------------------------------------

grd_fil = '/Volumes/P1/ROMS-Inputs/MaPhil/Grid/MaPhil_grd_high_res_bathy_mixedJerlov.nc'
grd_fid = nc.Dataset(grd_fil)
mask_rho = grd_fid.variables['mask_rho'][:]
mask_psi = grd_fid.variables['mask_psi'][:]
rlats = grd_fid.variables['lat_rho'][:]
rlons = grd_fid.variables['lon_rho'][:]
plats = grd_fid.variables['lat_psi'][:]
plons = grd_fid.variables['lon_psi'][:]

### OFFSETS
joffset = 0
ioffset = 0

m_offset = 0.01
mask_val = 0
map_order = 30

#Set up figure 
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(0, mask_psi.shape[1]), ylim=(0, mask_psi.shape[0]))

m = Basemap(llcrnrlat=np.min(plats)-m_offset,urcrnrlat = np.max(plats)+m_offset,llcrnrlon=np.min(plons)-m_offset,urcrnrlon=np.max(plons)+m_offset, resolution='f', ax=ax)
P = m.pcolormesh(plons,plats,mask_rho[1:-1,1:-1],vmin=.5,vmax=.75,edgecolors='face',cmap='Blues',zorder=map_order)
P.cmap.set_under('white')
P.cmap.set_over([.9,.97,1])
#m.plot(rlons,rlats,'ko',markersize=.1,zorder=map_order+3)

# MAP DETAILING
outline_mask(m,mask_rho[1:-1,1:-1],mask_val,plons[0,0],plats[0,0],plons[-1,-1],plats[-1,-1])

#DOMAIN OUTLINE
for j in range(plats.shape[0]-1):
    m.plot((plons[j,0],plons[j+1,0]),(plats[j,0],plats[j+1,0]),linewidth=2,color='k',zorder=map_order+1)
    m.plot((plons[j,-1],plons[j+1,-1]),(plats[j,-1],plats[j+1,-1]),linewidth=2,color='k',zorder=map_order+1)
for ii in range(plats.shape[1]-1):
    m.plot((plons[0,ii],plons[0,ii+1]),(plats[0,ii],plats[0,ii+1]),linewidth=2,color='k',zorder=map_order+1)
    m.plot((plons[-1,ii],plons[-1,ii+1]),(plats[-1,ii],plats[-1,ii+1]),linewidth=2,color='k',zorder=map_order+1)

polygon_patch(m,ax)

m.drawmeridians([124,125], labels=[0,0,1,0], fmt='%d', fontsize=18)
m.drawparallels([10,11], labels=[1,0,0,0], fmt='%d', fontsize=18)

plt.savefig('Camotes_domain')
plt.show()
 
