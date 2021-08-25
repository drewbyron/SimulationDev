import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import math

from matplotlib.patches import Rectangle

from spec_tools.coil_classes.coil_form import Coil_form
from spec_tools.coil_classes.field_profile import Field_profile
from spec_tools.load_default_field_profiles import (
    load_he6_trap,load_main_magnet)
from spec_tools.formatting.reformatter import config_tick_reformatter


main_field = 0.7
field_profile = load_he6_trap(main_field)

center_coil = Coil_form(inner_radius=1e-2,
    outer_radius=2e-2,
    left_edge=-1e-2,
    right_edge=1e-2,
    z_center=0,
    num_windings=1000,
    current_per_wire=0.25,
    name="center coil")

field_profile = Field_profile([center_coil])
field_profile = load_main_magnet(main_field)

plot_coil_geometry = True
show_legend = True

xaxis_label = r'$z$-position (cm)'
yaxis_label = r'$r$-position (cm)'

plot_title = "Main Magnet Model Coil Geometry"

xlim = 52.705
ylim = 20.5

#xlim = -6
#ylim = 2
colors = ["blue","blue","green","green","orange","orange","purple","purple","cyan","cyan"]
color_index = 0

labels = ["center coils","center coils", "inner coils","inner coils","mid coils","mid coils", "outer coils","outer coils", "trim coils","trim coils"]
label_index = 0

#field_profile = load_he6_trap(main_field)
#labels = ["center coils","center coils","outer coils","outer coils"]
#colors = ["blue","green","green"]


if plot_coil_geometry == True:

    fig, ax = plt.subplots(figsize=(12,6))
        
    tick_reformatter_x = config_tick_reformatter()
    tick_reformatter_y = config_tick_reformatter()
       
    ax.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_x))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter_y))

    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    
    for coil in field_profile.get_coil_list():
    
        current_color = colors[color_index]
        curr_label = labels[label_index]
    
        inner_radius = coil.inner_radius /1e-2
        outer_radius = coil.outer_radius /1e-2
        left_edge = coil.left_edge /1e-2
        right_edge = coil.right_edge /1e-2
        z_center = coil.z_center /1e-2
        num_windings = coil.num_windings
        name = coil.name
        
        
        coil_thickness = outer_radius - inner_radius
        coil_length = right_edge - left_edge
        
        windings = int(round(num_windings * coil_length*1e-2,0))
    
        upper_coils_edge = Rectangle((z_center + left_edge,inner_radius),
                        coil_length,
                        coil_thickness,
                        fc = "none",
                        ec = current_color)
                        
        lower_coils_edge = Rectangle((z_center + left_edge,-outer_radius),
                        coil_length,
                        coil_thickness,
                        fc = "none",
                        ec = current_color)
                        
        upper_coils_fill = Rectangle((z_center + left_edge,inner_radius),
                        coil_length,
                        coil_thickness,
                        fc = current_color,
                        ec = "none",
                        alpha = 0.4)
                        
        lower_coils_fill = Rectangle((z_center + left_edge,-outer_radius),
                        coil_length,
                        coil_thickness,
                        fc = current_color,
                        ec = "none",
                        alpha = 0.4)
        
        if label_index % 2 == 0:
            cross_section = Rectangle((z_center + left_edge,-inner_radius),
                                coil_length,
                                2 * inner_radius,
                                fc = "none",
                                ec = current_color,
                                lw = 1,
                                label=curr_label + ", {} windings".format(windings))
        else:
            cross_section = Rectangle((z_center + left_edge,-inner_radius),
                                coil_length,
                                2 * inner_radius,
                                fc = "none",
                                ec = current_color,
                                lw = 1)
                        
            
        ax.add_patch(upper_coils_edge)
        ax.add_patch(lower_coils_edge)
        ax.add_patch(upper_coils_fill)
        ax.add_patch(lower_coils_fill)
        ax.add_patch(cross_section)
        
        color_index = color_index + 1
        label_index = label_index + 1
    
    ax.set_title(plot_title)
    if show_legend == True:
        fig.legend(loc='upper right')
        
    ax.set_xlim(-xlim,xlim)
    ax.set_ylim(-ylim,ylim)
    
    fig.subplots_adjust(left=0.16,bottom=0.12)
    fig.show()
