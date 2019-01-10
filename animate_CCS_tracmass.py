# this is a Python script that takes in binary output from TRACMASS,
# reads it, and plots an animation

# NOTE: COMMENTED OUT CURSOR LINES IN BACKEND FILE: 
# /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/backends/backend_agg.py  
# OTHERWISE WON'T SAVE GIF ANIMATION

import pandas
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation

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
        v.append((plons[p[0]+1,p[1]],plat[p[0]+1,p[1]]))
        v.append((plons[p[0]+1,p[1]+1],plat[p[0]+1,p[1]+1]))

        l.append((np.nan,np.nan))
        v.append((np.nan,np.nan))
    #vertical segments
    for p in zip(*ver_seg):
        l.append((plons[p[0],p[1]+1],plat[p[0],p[1]+1]))
        l.append((plons[p[0]+1,p[1]+1],plat[p[0]+1,p[1]+1]))

        l.append((np.nan, np.nan))
        v.append((np.nan, np.nan))

    l_segments = np.array(l)
    v_segments = np.array(v)
    mapid.plot(l_segments[:,0], l_segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+2)
    mapid.plot(v_segments[:,0], v_segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+3)

def bilin_interp(x,y):
    lats = np.empty([len(np.array(y))])
    lons = np.empty([len(np.array(x))])
    for nt in range(len(x)):
        x1 = int(np.floor(x[nt])); x2 = int(np.ceil(x[nt]))
        y1 = int(np.floor(y[nt])); y2 = int(np.ceil(y[nt]))

        # VECTOR = 1x2
        xdifs = np.array([x2-x[nt], x[nt]-x1])
        # VECTOR = 2x1
        ydifs = np.array([[y2-y[nt]],\
                           [y[nt]-y1]]) 

        flats = np.array([[lat[y1,x1],lat[y2,x1]],\
                         [lat[y1,x2],lat[y2,x2]]])

        flons = np.array([[lon[y1,x1],lon[y2,x1]],\
                         [lon[y1,x2],lon[y2,x2]]])

        if ((x1 != x2) & (y1 != y2)):
           # run bilinear interp
           lats[nt] = np.dot(xdifs,np.dot(flats,ydifs)) 
           lons[nt] = np.dot(xdifs,np.dot(flons,ydifs))  

        elif ((x1 == x2) & (y1 != y2)):
             # INTERP BASED JUST ON y-values
             lats[nt] = np.dot(flats[0,:],ydifs) 
             lons[nt] = np.dot(flons[0,:],ydifs)

        elif ((x1 != x2) & (y1 == y2)):
             # INTEP BASED JUST ON x-values
             lats[nt] = np.dot(xdifs,flats[:,0])

        else:
             # FALLS EXACTLY ON RHO POINT; y1=y2 and x1=x2
             lats[nt] = lat[y1,x1]
             lons[nt] = lon[y1,x1]
     
    return lats,lons
     
nff = 1
#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#
outdatadir = '/Users/elizabethdrenkard/ANALYSES/CCS/tracmass_out/ccs/00040228-1200/'
filename = 'testCCS_run.bin'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
referencefile = str(outdatadir + filename)

#Load the file(s) 
# BELOW PULLED FROM PYTRAJ
runtraj = np.fromfile(open(referencefile), \
                      np.dtype([('ntrac','i4'), ('ints','f8'),('x','f4'), ('y','f4'),('z','f4')]))

# Use pandas to pull selected columns (.loc) and convert to numpy array (.values)
data = pandas.DataFrame(runtraj).loc[:,['ntrac','x','y']].values

#Determine number of steps and which particles they contain
data_dif = np.diff(data[:,0])
istep = (np.where(data_dif<1)[0])+1
istep = np.append([0],istep)
nstep = len(istep)

print nstep

# ROMS Grid information
grdfile = '/Users/elizabethdrenkard/ANALYSES/CCS/Inputs/Grid/CCS_grd_high_res_bathy_jerlov.nc' 
fid = nc.Dataset(grdfile)
mask_rho = fid.variables['mask_rho'][:]
rlat = fid.variables['lat_rho'][:]
rlon = fid.variables['lon_rho'][:]
plat = fid.variables['psi_rho'][:]
plon = fid.variables['psi_rho'][:]

### OFFSETS
joffset = 0
ioffset = 0

m_offset = 0.01
mask_val = 0
map_order = 30
#Set up figure and animation
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False) 

# WHOLE DOMAIN
m = Basemap(llcrnrlat=np.min(lat)-m_offset,urcrnrlat = np.max(lat)+m_offset,llcrnrlon=np.min(lon)-m_offset,urcrnrlon=np.max(lon)+m_offset, resolution='f', ax=ax)

P = m.pcolormesh(plon,plat,mask_rho[1:-1,1:-1], vmin=.5,vmax=.75,edgecolors='face',cmap='Blues',zorder=map_order)
P.cmap.set_under('white')
P.cmap.set_over([.9,.97,1])

# MAP DETAILING
outline_mask(m,mask_rho[1:-1,1:-1],mask_val,plon[0,0],plat[0,0],plon[-1,-1],plat[-1,-1])

#DOMAIN OUTLINE
for j in range(lat.shape[0]-2):
    m.plot((lon[j,0],lon[j+1,0]),(lat[j,0],lat[j+1,0]),linewidth=2,color='k',zorder=map_order+1)
    m.plot((lon[j,-1],lon[j+1,-1]),(lat[j,-1],lat[j+1,-1]),linewidth=2,color='k',zorder=map_order+1)
for ii in range(lat.shape[1]-2):
    m.plot((lon[0,ii],lon[0,ii+1]),(lat[0,ii],lat[0,ii+1]),linewidth=2,color='k',zorder=map_order+1)
    m.plot((lon[-1,ii],lon[-1,ii+1]),(lat[-1,ii],lat[-1,ii+1]),linewidth=2,color='k',zorder=map_order+1)

polygon_patch(m,ax)

m.drawmeridians([-116,-121], labels=[0,0,1,0], fmt='%d', fontsize=18,zorder=map_order+2)
m.drawparallels([30,35], labels=[1,0,0,0], fmt='%d', fontsize=18,zorder=map_order+2)

#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])

ax.set_xlim(-121.5-m_offset,-115.5+m_offset)
ax.set_ylim(30-.5,35+.5)
#ax.set_xlim(360,400)
#ax.set_ylim(80,200)

particles, = m.plot([], [], 'go', ms =2, mec = 'none',zorder=map_order+4) #mec='y',ms=4)
#particles, = ax.plot([], [], 'o', ms =4) #mec='y',ms=4)

def init():
#   initialize animation
    particles.set_data([], [])
    return particles,

def animate(i):
#   perform animation step
    print i
    row_start = istep[i]
    if i == nstep-1:
       row_end=None
    else:
       row_end = istep[i+1]

    if nff == 1:
       row_start = istep[i]
       if i == nstep-1:
          row_end=None
       else:
          row_end = istep[i+1]
    elif nff == -1: 
       row_start = istep[nstep-(i+1)]
       if i == 0:
          row_end=None
       else:
          row_end = istep[nstep-i]

 
    xvals = data[row_start:row_end,1]+ioffset
    yvals = data[row_start:row_end,2]+joffset
    #cvals = data2[row_start:row_end,0]
    #cvals = plt.cm.jet(norm(data2[row_start:row_end,0]))
    lat_vals, lon_vals = bilin_interp(xvals,yvals) 

    particles.set_data(lon_vals,lat_vals)
    #particles.set_markerfacecolor(cvals)

    return particles,

ani = animation.FuncAnimation(fig, animate, frames=(nstep), interval=200, blit=False, init_func=init)
#ani = animation.FuncAnimation(fig, animate, frames=3, interval=200, blit=True, init_func=init)
ani.save('CCS.gif',writer='imagemagick',fps=5)
#plt.show()
 
