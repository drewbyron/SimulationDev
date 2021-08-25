import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import math

from spec_tools.coil_classes.coil_form import Coil_form
from spec_tools.coil_classes.field_profile import Field_profile
from spec_tools.load_field_profile import load_field_profile
from spec_tools.load_default_field_profiles import load_he6_coils
from spec_tools.load_trap_profile import load_trap_profile

from spec_tools.formatting.reformatter import config_tick_reformatter

#template for building field profiles from coils

plot_coil_fields = True
plot_individual_coils =  False

plot_field_lines = True
plot_coil_derivative = True
plot_coil_second_derivative = True

main_field = 0
radius = 0

center_coil = Coil_form(inner_radius=1e-2,
    outer_radius=2e-2,
    left_edge=-1e-2,
    right_edge=1e-2,
    z_center=0,
    num_windings=1000,
    current_per_wire=0.25,
    name="center coil")
    
#Helmholtz coil pair
left_coil = Coil_form(inner_radius=1e-2,
    outer_radius=1.1e-2,
    left_edge=-0.25e-2,
    right_edge=0.25e-2,
    z_center=-0.55960800611e-2,
    num_windings=1000,
    current_per_wire=0.25,
    name="left Helmholtz coil")

right_coil = Coil_form(inner_radius=1e-2,
    outer_radius=1.1e-2,
    left_edge=-0.25e-2,
    right_edge=0.25e-2,
    z_center=0.55960800611e-2,
    num_windings=1000,
    current_per_wire=0.25,
    name="right Helmholtz coil")
    
#quadrupole
left_quad_coil = Coil_form(inner_radius=1e-2,
    outer_radius=1.1e-2,
    left_edge=-0.25e-2,
    right_edge=0.25e-2,
    z_center=-0.55960800611e-2,
    num_windings=1000,
    current_per_wire=0.25,
    name="left quadrupole coil")

right_quad_coil = Coil_form(inner_radius=1e-2,
    outer_radius=1.1e-2,
    left_edge=-0.25e-2,
    right_edge=0.25e-2,
    z_center=0.55960800611e-2,
    num_windings=1000,
    current_per_wire=-0.25,
    name="right quadrupole coil")

    
#list_coils = [center_coil]
#list_coils = [left_coil,right_coil]
list_coils = [left_quad_coil,right_quad_coil]

rrange = [-5e-2,5e-2]
#zrange = [-2.5e-2,2.5e-2]
zrange = [-5e-2,5e-2]
zpoints = 200

field_line_points = 10

if plot_coil_fields == True:

    #field_profile = Field_profile(list_coils,main_field)
    #field_profile = load_field_profile("he6_trap_coils.json")
    #field_profile = load_trap_profile("he6_trap_2e-3.json")
    field_profile = load_he6_coils(1)
    
    fig, ax = plt.subplots(figsize=(12,6))
        
    tick_reformatter_x = config_tick_reformatter()
    tick_reformatter_y = config_tick_reformatter(2)
       
    ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_x))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_y))

    ax.set_xlabel(r'$z$-position (cm)')
    ax.set_ylabel(r'Field Strength (T)')
    
    xvals = [i for i in np.linspace(zrange[0],zrange[1],zpoints)]
    yvals = []
    for k in range(len(xvals)):
        if (k+1) % 10 == 0:
            print(k+1,"/",len(xvals),"values left to calculate")
        yvals.append(field_profile.field_values((radius,0,xvals[k]))[2])
       
    xvals = [x/1e-2 for x in xvals]
        
    ax.plot(xvals,yvals,label="total field of coils, radius = {}".format(radius))
    
    if plot_individual_coils == True:
        for coil in list_coils:
            yvals = []
            for k in range(len(xvals)):
                if (k+1) % 10 == 0:
                    print(k+1,"/",len(xvals),"values left to calculate")
                yvals.append(coil.field_values((radius,0,xvals[k]*1e-2))[2])
                
            ax.plot(xvals,yvals,label="Field of '{}', radius = {}".format(coil.name,radius))
        
    ax.set_title(r"Field of Current Coil Geometry")
    ax.legend(loc='upper right')
    
    fig.subplots_adjust(left=0.16,bottom=0.12)
    fig.show()
    
if plot_field_lines == True:
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    r_input = np.array([x for x in np.linspace(rrange[0],rrange[1],field_line_points)])
    z_input = np.array([x for x in np.linspace(zrange[0],zrange[1],field_line_points)])
    
    R,Z = np.meshgrid(r_input,z_input)
    
    r_shape = R.shape
    
    U = np.zeros(r_shape)
    V = np.zeros(r_shape)
    
    dr = lambda position : field_profile.field_values(position)[0]
    dz = lambda position : field_profile.field_values(position)[2]
    
    for i in range(r_shape[0]):
        for j in range(r_shape[1]):
        
            U[i,j] = dr((R[i,j],0,Z[i,j]))
            V[i,j] = dz((R[i,j],0,Z[i,j]))
            
            R[i,j] = R[i,j]/1e-2
            Z[i,j] = Z[i,j]/1e-2
    
    ax.streamplot(R,Z,U,V,density=3)
    ax.set_title("Current Coil Geometry Field Lines")
    
    ax.set_xlabel(r'$r$-position (cm)')
    ax.set_ylabel(r'$z$-position (cm)')
    
    ax.set_xlim(rrange[0]/1e-2,rrange[1]/1e-2)
    ax.set_ylim(zrange[0]/1e-2,zrange[1]/1e-2)
    fig.show()

    
if plot_coil_derivative == True:

    fig, ax = plt.subplots(figsize=(12,6))
     
    tick_reformatter_x = config_tick_reformatter()
    tick_reformatter_y = config_tick_reformatter(2)
    
    ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_x))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_y))
    
    ax.set_xlabel(r'$z$-position (cm)')
    ax.set_ylabel(r'Field Strength / Centimeter (T / cm)')
    
 
    xvals = [i for i in np.linspace(zrange[0],zrange[1],zpoints)]
    yvals = []
    for k in range(len(xvals)):
        if (k+1) % 10 == 0:
            print(k+1,"/",len(xvals),"values left to calculate")
        yvals.append(field_profile.field_derivative(radius,xvals[k])/1e2)
        
    xvals = [x/1e-2 for x in xvals]
      
    ax.plot(xvals,yvals,label="radius = {}".format(radius))
    ax.set_title(r"Axial $dB_{z}/dz$ of Field of Given Coil Geometry")
 
    fig.subplots_adjust(left=0.16,bottom=0.12)
    fig.show()
    
if plot_coil_second_derivative == True:

    fig, ax = plt.subplots(figsize=(12,6))
    
    tick_reformatter_x = config_tick_reformatter()
    tick_reformatter_y = config_tick_reformatter(2)
   
    ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_x))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_y))
   
    ax.set_xlabel(r'$z$-position (cm)')
    ax.set_ylabel(r'Field Strength / Centimeter Squared (T / $\mathrm{cm}^{2}$)')
   
    xvals = [i for i in np.linspace(zrange[0],zrange[1],zpoints)]
    yvals = []
   
    for k in range(len(xvals)):
        if (k+1) % 10 == 0:
            print(k+1,"/",len(xvals),"values left to calculate")
        yvals.append(field_profile.field_derivative(radius,xvals[k],deriv_order=2)/1e4)
   
    xvals = [x/1e-2 for x in xvals]
   
    #print("d^2Bz / dz^2 at z=0 cm:",field_profile.field_derivative(radius,0,deriv_order=2))
    ax.plot(xvals,yvals,label="radius = {}".format(radius))
    ax.set_title(r"Axial $d^{2}B_{z}/dz^{2}$ of Field of Given Coil Geometry")
   
    #ax.set_ylim(min(yvals),0)
   
    fig.subplots_adjust(left=0.16,bottom=0.12)
    fig.show()
    
