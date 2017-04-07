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

def convert_time(time):
    ref = dt.datetime(1991,12,4,0,0)    
    date_vals = np.zeros(len(time))
    for nt in range(len(time)):
        day_time = ref + dt.timedelta(seconds=np.float(time[nt]))
        date_vals[nt] = pltd.date2num(day_time)
    return date_vals

def convert_coord(eta,xi):
#    print eta[0:5]
#    print xi[0:5]
    lon_store = []
    lat_store = []
    n=0
    for nt in range(len(eta)):
        eta1 = int(np.floor(eta[nt]))
        eta2 = int(np.ceil(eta[nt]))
        xi1  = int(np.floor(xi[nt]))
        xi2  = int(np.ceil(xi[nt]))

        if eta1 < ilats[-1] and eta2 < ilats[-1] and \
           eta1 > ilats[0]  and eta2 > ilats[0]  and \
           xi1  < ilons[-1] and xi2  < ilons[-1] and \
           xi1  > ilons[0]  and xi2  > ilons[0]  :
         
           lat1 = lats[eta1,xi1]
           lat2 = lats[eta2,xi2]
           lon1 = lons[eta1,xi1]
           lon2 = lons[eta2,xi2]
           if lat1 == lat2:
              lat_store.append(lat1)
           else: 
              lat_store.append((lat1*np.absolute(eta[nt]-eta2) + lat2*np.absolute(eta[nt]-eta1))/np.absolute(eta2-eta1))

           if lon1 == lon2:
              lon_store.append(lon1)
           else:
              lon_store.append((lon1*np.absolute(xi[nt]-xi2) + lon2*np.absolute(xi[nt]-xi1)) /np.absolute(xi2-xi1))

    return lon_store, lat_store

def get_sst(i):
    #ref = dt.datetime(int(outdatadir[43:47]),int(outdatadir[47:49]),int(outdatadir[49:51]))
    #ref = dt.datetime(int(outdatadir[47:51]),int(outdatadir[51:53]),int(outdatadir[53:55]))
    #ref = dt.datetime(int(outdatadir[51:55]),int(outdatadir[55:57]),int(outdatadir[57:59]))
    ref = dt.datetime(int(outdatadir[51:55]),int(outdatadir[55:57]),int(outdatadir[57:59]))
    filedate = ref + dt.timedelta(days = i)
    #filedate = ref + dt.timedelta(days = nday+nff*i)

    ncfile = '/Volumes/P4/workdir/liz/external_data/GLORYS_files/GLORYS_' + str(filedate.year) + str(filedate.month).zfill(2) + str(filedate.day).zfill(2) + '_T.nc'
    fid = nc.Dataset(ncfile)
    sst = np.squeeze(fid.variables['votemper'][:,0,Ilats[0]:Ilats[1],Ilons[0]:Ilons[1]])

    return sst, str(filedate.year),str(filedate.month).zfill(2),str(filedate.day).zfill(2)

def get_particles(i):
    row_start = istep[i]
    if i == nstep-1:
       row_end=None
    else:
       row_end = istep[i+1]
    xvals = data2[row_start:row_end,1]+ioffset
    yvals = data2[row_start:row_end,2]+joffset
    lon_vals,lat_vals = convert_coord(yvals,xvals)
    print i
    #print len(xvals)
    return lon_vals,lat_vals

# ANIMATION
def updatefig(i):
    global n,im1,im2,tx
    # REMOVE images after first step
    if n > 0:
       im1.remove()
       im2.remove()
    sst,yr,mon,day = get_sst(i)
    lon_vals, lat_vals = get_particles(i)
    im1   = m.pcolor(lons[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],\
                     lats[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],\
                      sst[ilats[0]:ilats[-1]+1,ilons[0]:ilons[-1]+1],vmin=vmin,vmax=vmax)
    im2,  = m.plot(lon_vals,lat_vals,'yo',mec='y',ms=1)
    tx_str = yr + '-' + mon + '-' + day
    tx.set_text(tx_str)
    # ADD Colorbar on first iteration
    if i == 0:
       m.colorbar(im1)
    n+=1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#

#outdatadir = '/Volumes/P4/workdir/liz/pipa_tracmass/pipa/19920603-1200/'
#outdatadir = '/Volumes/P4/workdir/liz/PIPA_tracmass/forward/pipa/19930101-1200/'
outdatadir = '/Volumes/P4/workdir/liz/PIPA_tracmass/forward/pipa/19990501-0000/'
#outdatadir ='/Volumes/P4/workdir/liz/pipa_tracmass/pipa/20121201-0000/'
#outdatadir ='/Volumes/P4/workdir/liz/PIPA_tracmass/backward/pipa/19990301-1200/'
#outdatadir ='/Volumes/P4/workdir/liz/PIPA_tracmass/pipa/20090101-1200/'
filename = 'test_pipa_run.bin'

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
# DIRECTION OF TIMESTEP
nff = 1

# FORTRAN TO PYTHON CONVERSION
ioffset = 0
joffset = 0

# SUBSETTING GLOBAL GLORYS BASED ON TRACMASS GRID 
Ilons = [344-1,575]
Ilats = [450-1,560]

# ACCESS GRID INFO
grid_file = '/Volumes/P4/workdir/liz/external_data/GLORYS_files/GL2V1_mesh_mask_new.nc'  
fid = nc.Dataset(grid_file)

mask = np.squeeze(fid.variables['tmask'][0,Ilats[0]:Ilats[1],Ilons[0]:Ilons[1]])
lats = np.squeeze(fid.variables['nav_lat'][Ilats[0]:Ilats[1],Ilons[0]:Ilons[1]])
lons = np.squeeze(fid.variables['nav_lon'][Ilats[0]:Ilats[1],Ilons[0]:Ilons[1]])
lons[np.where(lons<0)] = 2*180 + lons[np.where(lons<0)]

# PLOTTING LIMITS
Lats = [-2,3]
Lons = [197, 202]

Lats = [-7,7]
Lons = [185, 205]

Lats = [-6,8]
Lons = [185, 235]

#Lats = [-9,9]
#Lons = [160, 215]

ilats = np.where((Lats[0]<lats[:,0]) & (lats[:,0]<Lats[1]))
ilons = np.where((Lons[0]<lons[0,:]) & (lons[0,:]<Lons[1]))
ilats = ilats[0]
ilons = ilons[0]

# SST COLOR LIMITS
vmin = 25
vmax = 30

#FIGURE and MAP SET UP
fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(111)
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

plt.title('SST (oC)')
tx =plt.text(lons[ilats[0],ilons[0]]-b/2,lats[ilats[-1],ilons[-1]]+b/2,'', fontsize=14)

#ani = animation.FuncAnimation(fig, updatefig,frames=10, blit=False)
n=0
ani = animation.FuncAnimation(fig, updatefig,frames=nstep, blit=False)

#ani = animation.FuncAnimation(fig, updatefig,frames=150, blit=False)
ani.save('PIPA_map.gif', writer = 'imagemagick',fps=5)
#plt.show()
