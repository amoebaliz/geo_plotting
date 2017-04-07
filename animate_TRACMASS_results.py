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

nff = -1
ngr = 0
grds = ['VIP',   \
        'CORAL', \
        'MaPhil',\
        'ARCTIC' ] 

trmrn = ['vip', 'coral', 'maphil','arctic']
#(CASENAME, PROJECTNAME) :: Initiates pytraj
tr = pytraj.Trm(trmrn[ngr],trmrn[ngr])

#~~~~~~~~REPLACE FILENAME HERE~~~~~~~#

outdirs = ['/Volumes/P4/workdir/liz/VIP_tracmass/vip/19960706-0000/',\
           '/Volumes/P4/workdir/liz/CT_tracmass/coral/20030301-1200/',\
           '/Volumes/P4/workdir/liz/MAPHIL_tracmass/forward/maphil/20100520-0000/',\
           '/Volumes/P4/workdir/liz/arctic_tracmass/arctic/19960103-1200/']

outdatadir = outdirs[ngr]

flnms = ['testVIP_run.bin', \
         'CT-test_run.bin',\
         'test_maphil_run.bin',\
         'test_arctic_run.bin']

filename = flnms[ngr] 
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

if ngr < 3: 
   GRD = pyroms.grid.get_ROMS_grid(grds[ngr])
   mask = GRD.hgrid.mask_rho
else: # NOT A ROMS GRID
   grid_file = '/Volumes/P4/workdir/liz/external_data/ARCTIC_GLORYS_files/GL2V1_mesh_mask_new.nc'
   fid = nc.Dataset(grid_file)
#   mask = np.squeeze(fid.variables['tmask'][0,469:519,443:537])
#   lons = np.squeeze(fid.variables['nav_lon'][469:519,443:537])
#   lats = np.squeeze(fid.variables['nav_lat'][469:519,443:537])
   mask = np.squeeze(fid.variables['tmask'][:,0,:,:])
   lons = np.squeeze(fid.variables['nav_lon'][:])
   lats = np.squeeze(fid.variables['nav_lat'][:])

### OFFSETS
joffset = 0
ioffset = 0 

#Set up figure and animation
#m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180, lat_0=45,lon_0=-100,resolution='l')
fig = plt.figure(figsize=(14,7))
fig.subplots_adjust(left=.1, right=.9, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(0, mask.shape[1]-1), ylim=(0, mask.shape[0]-1))
ax.pcolor(mask,vmin = -.1, vmax =1.1, cmap='ocean')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

particles, = ax.plot([], [], 'o', ms=4) #mec='y',ms=4)
def init():
#   initialize animation
    particles.set_data([], [])
    return particles,

def animate(i):
#   perform animation step
    print i
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


 
    xvals = data2[row_start:row_end,1]+ioffset
    yvals = data2[row_start:row_end,2]+joffset
    #cvals = data2[row_start:row_end,0]
    #cvals = plt.cm.jet(norm(data2[row_start:row_end,0]))
    particles.set_data(xvals, yvals)
    #particles.set_markerfacecolor(cvals)

    return particles,

ani = animation.FuncAnimation(fig, animate, frames=(nstep), interval=200, blit=True, init_func=init)
#ani = animation.FuncAnimation(fig, animate, frames=3, interval=200, blit=True, init_func=init)
gif_name = grds[ngr] + '.gif'
ani.save(gif_name, writer = 'imagemagick',fps=5)
plt.show()
 
