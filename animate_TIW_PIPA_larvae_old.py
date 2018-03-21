# this is a Python script that takes in binary output from TRACMASS,
#tr = pytraj.Trm('VIP','VIP')
# reads it, and plots an animation

import pytraj
import pandas
import pyroms
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

#def convert_coord(eta,xi):
#    lon_store = []
#    lat_store = []
#    n=0
#    for nt in range(len(eta)):
#        eta1 = int(np.floor(eta[nt]))
#        eta2 = int(np.ceil(eta[nt]))
#        xi1  = int(np.floor(xi[nt]))
#        xi2  = int(np.ceil(xi[nt]))

#        if eta1 < ilats[-1] and eta2 < ilats[-1] and \
#           eta1 > ilats[0]  and eta2 > ilats[0]  and \
#           xi1  < ilons[-1] and xi2  < ilons[-1] and \
#           xi1  > ilons[0]  and xi2  > ilons[0]  :
         
#           lat1 = lats[eta1,xi1]
#           lat2 = lats[eta2,xi2]
#           lon1 = lons[eta1,xi1]
#           lon2 = lons[eta2,xi2]
#           if lat1 == lat2:
#              lat_store.append(lat1)
#           else: 
#              lat_store.append((lat1*np.absolute(eta[nt]-eta2) + lat2*np.absolute(eta[nt]-eta1))/np.absolute(eta2-eta1))

#           if lon1 == lon2:
#              lon_store.append(lon1)
#           else:
#              lon_store.append((lon1*np.absolute(xi[nt]-xi2) + lon2*np.absolute(xi[nt]-xi1)) /np.absolute(xi2-xi1))
#           print 'LON STORE', lon_store.shape
#    return lon_store, lat_store

def get_sst(i):
    ref = dt.datetime(int(outdatadir[32:36]),int(outdatadir[36:38]),int(outdatadir[38:40]))
    filedate = ref + dt.timedelta(days = i)

    ncfile = '/Volumes/Abalone/GLORYS_CLIM/GLORYS_' + str(filedate.month).zfill(2) + '_' + str(filedate.day).zfill(2) + '_T.nc'
    print ncfile
    fid = nc.Dataset(ncfile)
    sst = np.squeeze(fid.variables['votemper'][:])
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

#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#
outdatadir ='/Volumes/Abalone/PIPA_trac/pipa/19931121-1200/'
filename = 'test_pipa_run.bin'
#(CASENAME, PROJECTNAME) :: Initiates pytraj
tr = pytraj.Trm('pipa','pipa')
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# FORTRAN TO PYTHON CONVERSION
ioffset = 0
joffset = 0
map_order = 2
# ACCESS GRID INFO
grid_file = '/Volumes/Abalone/GLORYS_CLIM/GL2V1_mesh_mask_new_sub.nc'
fid = nc.Dataset(grid_file)

mask = np.squeeze(fid.variables['tmask'][:])
lats = np.squeeze(fid.variables['nav_lat'][:])
lons = np.squeeze(fid.variables['nav_lon'][:])
lons[np.where(lons<0)] = 2*180 + lons[np.where(lons<0)]

# PLOTTING LIMITS
Lats = [-6,8]
Lons = [185, 235]

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

b = 1.5 	# buffer; degrees

m = Basemap(llcrnrlon=lons[ilats[0] ,ilons[0]] -b,\
            llcrnrlat=lats[ilats[0] ,ilons[0]] -b,\
            urcrnrlon=lons[ilats[-1],ilons[-1]]+b,\
            urcrnrlat=lats[ilats[-1],ilons[-1]]+b,resolution='f')

# MAP DETAILS
m.drawcoastlines()
m.fillcontinents()

parallels = [-5,0,5]
meridians = [190,200,210,220,230]
m.drawparallels(parallels,labels=[True,False,False,False])
m.drawmeridians(meridians,labels=[False,False,False,True])

particles, = m.plot([], [], 'go', ms =10, mec = 'none',zorder=map_order+4) 
#plt.title('SST (oC)')
#tx =plt.text(lons[ilats[0],ilons[0]]-b/2,lats[ilats[-1],ilons[-1]]+b/2,'', fontsize=14)

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
    lat_vals, lon_vals = bilin_interp(xvals,yvals) 
    print lat_vals
    particles.set_data(lon_vals,lat_vals)

    return particles,

ani = animation.FuncAnimation(fig, animate,frames=nstep, blit=True,init_func=init)
#ani = animation.FuncAnimation(fig, updatefig,frames=150, blit=False)
ani.save('PIPA_map.gif', writer = 'imagemagick',fps=5)
#plt.show()
