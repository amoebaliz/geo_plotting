# this is a Python script that takes in binary output from TRACMASS,
# reads it, and plots an animation

# NOTE: COMMENTED OUT CURSOR LINES IN BACKEND FILE: 
# /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/backends/backend_agg.py  
# OTHERWISE WON'T SAVE GIF ANIMATION

import pytraj
import pyroms
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
        if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
           v.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
        else :
           l.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))

        if p[1] == mask_img.shape[1] - 1 :
           if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
               v.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
           else:
               l.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
        else :
           if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
              v.append((lon[p[0]+1,p[1]+1],lat[p[0]+1,p[1]+1]))
           else:
              l.append((lon[p[0]+1,p[1]+1],lat[p[0]+1,p[1]+1]))

        l.append((np.nan,np.nan))
        v.append((np.nan,np.nan))
    #vertical segments
    for p in zip(*ver_seg):
        if p[1] == mask_img.shape[1]-1:
           if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
              v.append((lon[p[0],p[1]],lat[p[0],p[1]]))
              v.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
           else:
              l.append((lon[p[0],p[1]],lat[p[0],p[1]]))
              l.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
        elif p[0] == mask_img.shape[0]-1:
             if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
              v.append((lon[p[0],p[1]],lat[p[0],p[1]]))
              v.append((lon[p[0]+1,p[1]],lat[p[0]+1,p[1]]))
             else:
              l.append((lon[p[0],p[1]],lat[p[0],p[1]]))
              l.append((lon[p[0],p[1]+1],lat[p[0],p[1]+1]))
        else:
           if (vip_eta[0] < p[0] < vip_eta[1] and vip_xi[0] < p[1] < vip_xi[1]):
              v.append((lon[p[0],p[1]+1],lat[p[0],p[1]+1]))
              v.append((lon[p[0]+1,p[1]+1],lat[p[0]+1,p[1]+1]))
           else:
              l.append((lon[p[0],p[1]+1],lat[p[0],p[1]+1]))
              l.append((lon[p[0]+1,p[1]+1],lat[p[0]+1,p[1]+1]))

        l.append((np.nan, np.nan))
        v.append((np.nan, np.nan))
    segments = np.array(l)
    vip_segments = np.array(v)
    mapid.plot(segments[:,0], segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+2)
    mapid.plot(vip_segments[:,0], vip_segments[:,1], latlon=True, color=(0,0,0), linewidth=.75,zorder=map_order+3)


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
     

#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#
outdatadir = '/Users/elizabethdrenkard/ANALYSES/CCS/tracmass_out/ccs/00050701-1200/'
filename = 'testCCS_run.bin'
#(CASENAME, PROJECTNAME) :: Initiates pytraj
tr = pytraj.Trm('ccs','ccs')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
referencefile = str(outdatadir + filename)

#Load the file(s) 
data1 = pandas.DataFrame(tr.readfile(referencefile))

#Adjust columns in the dataframes
data1 = data1.loc[:,['ntrac','x','y']]

#Change to numpy array
data2 = pandas.DataFrame.as_matrix(data1)

#Determine number of steps and which particles they contain
data_dif = np.diff(data2[:,0])
istep = (np.where(data_dif<1)[0])+1
istep = np.append([0],istep)
nstep = len(istep)

print nstep
print data2.shape

GRD = pyroms.grid.get_ROMS_grid('CCS')
mask = GRD.hgrid.mask_rho
lat = GRD.hgrid.lat_rho
lon = GRD.hgrid.lon_rho
### OFFSETS
joffset = 0
ioffset = 0

m_offset = 0.01
mask_val = 0
map_order = 30
vip_eta = [0,870]
vip_xi  = [0,376]
#Set up figure and animation
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False) #, xlim=(0, mask.shape[1]), ylim=(0, mask.shape[0]))

# SUBDOMAIN
#m = Basemap(llcrnrlat=30-.5,urcrnrlat = 35+.5,llcrnrlon=-121.5-m_offset,urcrnrlon=-115.5+m_offset, resolution='f', ax=ax)

# WHOLE DOMAIN
m = Basemap(llcrnrlat=np.min(lat)-m_offset,urcrnrlat = np.max(lat)+m_offset,llcrnrlon=np.min(lon)-m_offset,urcrnrlon=np.max(lon)+m_offset, resolution='f', ax=ax)

P = m.pcolormesh(lon,lat,mask,vmin=.5,vmax=.75,edgecolors='face',cmap='Blues',zorder=map_order)
P.cmap.set_under('white')
P.cmap.set_over([.9,.97,1])
#m.pcolor(lon,lat,mask,vmin = -.1, vmax =1.1, cmap='ocean',zorder=-3)

# MAP DETAILING
outline_mask(m,mask,mask_val,lon[0,0],lat[0,0],lon[-1,-1],lat[-1,-1])

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
#ax.set_xlim(-121.5-m_offset,-115.5+m_offset)
#ax.set_ylim(30-m_offset,35+m_offset)
#ax.set_xlim(360,400)
#ax.set_ylim(80,200)
print 'MEEP'
particles, = m.plot([], [], 'go', ms =1, mec = 'none',zorder=map_order+4) #mec='y',ms=4)
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
 
    xvals = data2[row_start:row_end,1]+ioffset
    yvals = data2[row_start:row_end,2]+joffset
    #cvals = data2[row_start:row_end,0]
    #cvals = plt.cm.jet(norm(data2[row_start:row_end,0]))
    lat_vals, lon_vals = bilin_interp(xvals,yvals) 

    particles.set_data(lon_vals,lat_vals)
    #particles.set_markerfacecolor(cvals)

    return particles,

ani = animation.FuncAnimation(fig, animate, frames=(nstep), interval=200, blit=True, init_func=init)
#ani = animation.FuncAnimation(fig, animate, frames=3, interval=200, blit=True, init_func=init)
ani.save('CCS.gif',writer='imagemagick',fps=5)
#plt.show()
 
