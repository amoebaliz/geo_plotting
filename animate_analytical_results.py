#this is a Python script that takes in binary output from TRACMASS,
# reads it, and plots an animation

import pytraj
import pyroms
import pandas
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation

def get_vel(i):
    # FILENAME
    ncfil = '/Users/elizabethdrenkard/external_data/analytical_tracmass/test_2010_01-' + \
    str(i+1).zfill(2) + '.nc' 
    
    # GET VELOCITY FIELDS
    fid = nc.Dataset(ncfil)
    u_vel = fid.variables['u'][:,-1,:].squeeze()
    v_vel = fid.variables['v'][:,-1,:].squeeze()

    # INTERPOLATE TO RHO POINTS
    #u_vel_2 = (u_vel[1:-1,:-1]+u_vel[1:-1,1:])/2
    #v_vel_2 = (v_vel[:-1,1:-1]+v_vel[1:,1:-1])/2

    # projection
    #u4 = u_vel_2*(np.cos(angs)) + v_vel_2*(np.cos(np.pi/2 + angs))
    #v4 = u_vel_2*(np.sin(angs)) + v_vel_2*(np.sin(np.pi/2 + angs))

    return u_vel, v_vel 




nff = 1
trmrn = 'analytical'
#(CASENAME, PROJECTNAME) :: Initiates pytraj
tr = pytraj.Trm(trmrn,trmrn)

#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#
grdfil     = '/Users/elizabethdrenkard/external_data/analytical_tracmass/test_grd.nc'
outdatadir = '/Users/elizabethdrenkard/external_data/analytical_tracmass/analytical/20100101-1200/'
filename   = 'test_analytical_t00040177_run.bin'
#filename   = 'test_analytical_t00040184_run.bin'
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

fid = nc.Dataset(grdfil)
mask = np.squeeze(fid.variables['mask_rho'][:])
#lons = np.squeeze(fid.variables['lon_rho'][:])
#lats = np.squeeze(fid.variables['lat_rho'][:])

#Set up figure and animation
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=.1, right=.9, bottom=.1, top=.9)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(0, mask.shape[1]-1), ylim=(0, mask.shape[0]-1))
ax.pcolor(mask,vmin = -.1, vmax =1.1, cmap='ocean')

#GRID
[ax.plot([x,x],[0,mask.shape[0]],color='lightsteelblue') for x in np.linspace(0,90,num=91)] 
[ax.plot([0,mask.shape[1]],[y,y],'lightsteelblue') for y in np.linspace(0,90,num=91)]

[ax.plot([x,x],[0,mask.shape[0]],'k') for x in np.linspace(0,90,num=10)] 
[ax.plot([0,mask.shape[1]],[y,y],'k') for y in np.linspace(0,90,num=10)]


#ax.xaxis.set_ticks([20,30,40,50])
#ax.yaxis.set_ticks([20,30,40,50])

ax.xaxis.set_ticks([0,10,20])
ax.yaxis.set_ticks([0,10,20])

ax.set_xlim(0,10)
ax.set_ylim(0,10)

# INITIALIZE FIGURE
im1, = ax.plot(0, 0, 'o', ms=6)
u,v = get_vel(0)
velx, vely = np.meshgrid(np.arange(.5,mask.shape[1]),np.arange(.5,mask.shape[0]))
im2 = ax.quiver(u,v,zorder=20)

def updatefig(i):
    global im1, im2  #,tx
    print i
    # REMOVE PREVIOUS FIGURE
    im1.remove() 
    im2.remove()
#   perform animation step
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
 
    xvals = data2[row_start:row_end,1]
    yvals = data2[row_start:row_end,2]
    print len(np.unique(xvals))
    #print np.unique(yvals)

    u,v= get_vel(i)
    im1,   = ax.plot(xvals, yvals, 'o', ms=6,mfc='orange',mec='k',zorder=11)
    im2   = ax.quiver(velx,vely,u,v,scale=.2,pivot='mid',color='darkseagreen',zorder=10)
    #tx_str = mon 
    #tx.set_text(tx_str)

ani = animation.FuncAnimation(fig, updatefig,frames=(nstep), blit=False)
#ani = animation.FuncAnimation(fig, animate, frames=(nstep), interval=200, blit=True, init_func=init)
#ani = animation.FuncAnimation(fig, animate, frames=3, interval=200, blit=True, init_func=init)
gif_name = 'analytical.gif'
ani.save(gif_name, writer = 'imagemagick',fps=5)
plt.show()
 
