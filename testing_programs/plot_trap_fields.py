import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import math

from spec_tools.load_default_field_profiles import(
    load_he6_trap,load_main_magnet,load_he6_coils)
from spec_tools.formatting.reformatter import config_tick_reformatter

#adjustable template to plot different properties of trap fields

plot_trap_coil_fields = True

main_field = 5
trap_strength = 1e-3
radius = 0
zpos = 0

trap_profile = load_he6_trap(main_field,trap_strength)


#field plot information

#x-axis values
xrange = [-4.5e-2,4.5e-2]
#xrange = [0,5.78e-3]

xpoints = 201
#xpoints = 50

#x-axis value formatter
xval_formatter = lambda x : x/1e-2


#y-axis value generator function

yval_generator = lambda x : trap_profile.field_strength(radius,x)
#yval_generator = lambda x : trap_profile.field_values((x,0,zpos))[2]
#yval_generator = lambda x : trap_profile.field_grad((radius,0,x),grad_coordinates="cylindrical")[2][0]/1e2

#axis labels

xaxis_label = r'$z$-position (cm)'
#xaxis_label = r'$r$-position (cm)'
#xaxis_label = r'$z$-position (cm)'

yaxis_label = r'Field Strength (T)'
#yaxis_label = r'Field Strength / Centimeter (T / cm)'

#plot legend format

#plot_legend = "radius = {} cm".format(radius/1e-2)
plot_legend = "$z$ = {} cm".format(zpos/1e-2)
show_legend = False

#plot title

plot_title = "$B_{z}$ Field of Trap Coil Geometry at" + " r = {} cm".format(radius/1e-2)
#plot_title = "$B_{z}$ Field of Trap Coil Geometry at" + " z = {} cm".format(zpos/1e-2)
#plot_title = r"Radial $dB_{z}/dr$ of Field of Trap Coil Geometry at" + " r = {} cm".format(radius/1e-2)


if plot_trap_coil_fields == True:

    fig, ax = plt.subplots(figsize=(12,6))
        
    tick_reformatter_x = config_tick_reformatter()
    tick_reformatter_y = config_tick_reformatter(5)
       
    ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_x))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_y))

    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    
    xvals = [i for i in np.linspace(xrange[0],xrange[1],xpoints)]
    yvals = []
    for k in range(len(xvals)):
        if (k+1) % 10 == 0:
            print(k+1,"/",len(xvals),"values left to calculate")
        yvals.append(yval_generator(xvals[k]))
       
    xvals = [xval_formatter(x) for x in xvals]
        
    ax.plot(xvals,yvals,label=plot_legend)
    
    ax.set_title(plot_title)
    if show_legend == True:
        ax.legend(loc='upper right')
    
    fig.subplots_adjust(left=0.16,bottom=0.12)
    fig.show()
