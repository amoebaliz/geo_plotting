# this is a Python script that takes in binary output from TRACMASS,

import pandas
import netCDF4 as nc
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
import matplotlib.dates as pltd
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation

########

#def convert_time(time):
#    ref = dt.datetime(1991,12,4,0,0)    
#    date_vals = np.zeros(len(time))
#    for nt in range(len(time)):
#        day_time = ref + dt.timedelta(seconds=np.float(time[nt]))
#        date_vals[nt] = pltd.date2num(day_time)
#    return date_vals

def bilin_interp(x,y):
    lats = np.empty([len(y)])
    lons = np.empty([len(x)])
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

def get_sst(i):
    ncfile = '/nbhome/Liz.Drenkard/tracmass_stuff/test_sst.nc'
    fid = nc.Dataset(ncfile)
    # NOTE: check grid geolat/lon_c vs geolat/lon.
    # 	    For pcolor,need one fewer T points than Corner points
    #       1) compare grid and sst xh and yh - confirm same slice
    #       2) plot geolat/lon_c vs geolat/lon to see positioning.
    # 
    #	    In this case, need two-fewer T points in latitude

    sst = fid.variables['tos'][i,1:-1,:].squeeze()
    return sst 

#def get_particles(i):
#    row_start = istep[i]
#    if i == nstep-1:
#       row_end=None
      
#    else:
#       row_end = istep[i+1]
#    xvals = data2[row_start:row_end,1]+ioffset
#    yvals = data2[row_start:row_end,2]+joffset
    #print xvals
    #lon_vals,lat_vals = convert_coord(yvals,xvals)
#    lon_vals = xvals
#    lat_vals = yvals
    #print lon_vals
    #lon_vals=xvals
    #lat_vals=yvals
#    print i
    #print len(xvals)
#    return lon_vals,lat_vals

# ANIMATION
#def updatefig(i):
#    global n,im1,im2,tx
    # REMOVE images after first step
#    if n > 0:
#       im1.remove()
#       im2.remove()
#    sst = get_sst(i)
#    lon_vals, lat_vals = get_particles(i)
#    im1   = m.pcolor(lons[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],\
#                     lats[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],\
#                      sst[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],vmin=vmin,vmax=vmax)
#    im2,  = m.plot(lon_vals,lat_vals,'yo',mec='y',ms=3,latlon=True)
    #tx_str = yr + '-' + mon + '-' + day
    #tx.set_text(tx_str)
    # ADD Colorbar on first iteration
#    if i == 0:
#       m.colorbar(im1)
#    n+=1

# NOTE: current stuff to use - trying to get SST animation to work before analyzing particles

#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#
#outdatadir ='/Volumes/Abalone/PIPA_trac/pipa/19931121-1200/'
#filename = 'test_pipa_run.bin'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#referencefile = str(outdatadir + filename)

#Load the file(s) 
#runtraj = np.fromfile(open(referencefile), \
#                      np.dtype([('ntrac','i4'), ('ints','f8'),('x','f4'), ('y','f4'),('z','f4')]))

# Use pandas to pull selected columns (.loc) and convert to numpy array (.values)
#data = pandas.DataFrame(runtraj).loc[:,['ntrac','x','y']].values

#Determine number of steps and which particles they contain
#data_dif = np.diff(data2[:,0])
#istep = (np.where(data_dif<1)[0])+1
#istep = np.append([0],istep)
#nstep = len(istep)

#print nstep

# GRID Information
grdfile = '/nbhome/Liz.Drenkard/tracmass_stuff/test_grd.nc'
fid = nc.Dataset(grdfile)

# Masking values: 1=ocean, 0=land
mask = np.squeeze(fid.variables['wet'][:])
lats = np.squeeze(fid.variables['geolat_c'][:])
lons = np.squeeze(fid.variables['geolon_c'][:])
lons[np.where(lons<0)] = 2*180 + lons[np.where(lons<0)]

# PLOTTING LIMITS
Lats = [-10,10]
Lons = [160, 220]

ilats = np.where((Lats[0]<lats[:,0]) & (lats[:,0]<Lats[1]))
ilons = np.where((Lons[0]<lons[0,:]) & (lons[0,:]<Lons[1]))
ilats = ilats[0]
ilons = ilons[0]

# SST COLOR LIMITS
vmin = 25
vmax = 30

#FIGURE and MAP SET UP
fig = plt.figure(figsize=(14,7))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False) 
plt.hold(True)

b = 0 	# buffer; degrees

m = Basemap(llcrnrlon=lons[ilats[0] ,ilons[0]] -b,\
            llcrnrlat=lats[ilats[0] ,ilons[0]] -b,\
            urcrnrlon=lons[ilats[-1],ilons[-1]]+b,\
            urcrnrlat=lats[ilats[-1],ilons[-1]]+b,resolution='c')

# MAP DETAILS
m.drawcoastlines()
m.fillcontinents()

parallels = [-10,-5,0,5,10]
meridians = [170,180,190,200,210,220]
m.drawparallels(parallels,labels=[True,False,False,False])
m.drawmeridians(meridians,labels=[False,False,False,True])

sstplt = m.pcolormesh(lons,lats,get_sst(0).squeeze())

#plt.show()
#particles, = m.plot([], [], 'go', ms =10, mec = 'none',zorder=map_order+4) 
#plt.title('SST (oC)')
#tx =plt.text(lons[ilats[0],ilons[0]]-b/2,lats[ilats[-1],ilons[-1]]+b/2,'', fontsize=14)


plt.hold(False)
def init():
    #initialize animation
    #particles.set_data([], [])
    sstplt.set_array([]) 
    #return particles, sstplt,
    return sstplt

def animate(i):
#   perform animation step
    #row_start = istep[i]
    #if i == nstep-1:
    #   row_end=None
    #else:
    #   row_end = istep[i+1]
 
    #xvals = data2[row_start:row_end,1]+ioffset
    #yvals = data2[row_start:row_end,2]+joffset
    #lat_vals, lon_vals = bilin_interp(xvals,yvals) 
    #print lat_vals
    #particles.set_data(lon_vals,lat_vals)
    #sstplt = m.pcolormesh(lons,lats,get_sst(i).squeeze(),shading='gouraud')
    sstplt.set_array(get_sst(i).flatten())
    #return particles,
    #return sstplt,

ani = animation.FuncAnimation(fig, animate,frames=10, blit=False, init_func=init)
#ani.save('GFDL_eq.gif', writer = 'imagemagick',fps=5)
plt.show()
