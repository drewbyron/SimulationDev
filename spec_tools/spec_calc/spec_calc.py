import math
import numpy as np
import os
import json
from pathlib import Path
from scipy.integrate import romberg
import scipy.integrate as integrate
from scipy.optimize import fmin
from scipy.optimize import fminbound
from scipy.interpolate import interp1d
import scipy.special as ss
from numpy.random import uniform

from spec_tools.coil_classes.coil_form import Coil_form
from spec_tools.coil_classes.field_profile import Field_profile
from spec_tools.coil_classes.trap_profile import Trap_profile

#calculator functions
def freq_to_energy(frequency = 18e9, field = 1):
    """
    Calculates energy of beta particle in eV given cyclotron frequency in Hz, magnetic field in Tesla, and pitch angle at 90 degrees.
    """
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg.
    q = 1.602176634e-19 # Electron charge, in Coulombs
    
    gamma = q*field/(2*math.pi*frequency*m)
    if gamma < 1:
        gamma = 1
        max_freq = q*field/(2*math.pi*m)
        warning = "Warning: {:.3e} higher than maximum cyclotron frequency {:.3e}".format(frequency,max_freq)
        print(warning)
    return gamma * me - me
    
def energy_to_freq(energy = 0.283675e6, field = 1):
    """
    Calculates cyclotron frequency of beta particle in Hz given energy in eV, magnetic field in Tesla, and pitch angle at 90 degrees.
    """
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg.
    q = 1.602176634e-19 # Electron charge, in Coulombs

    gamma = (energy + me) / me
    cycl_freq = q * field / (2 * math.pi * gamma * m)
    return cycl_freq
    
def df_dt(energy,field,power,time=0):
    """
    Calculates cyclotron frequency rate of change of electron with
    kinetic energy in eV at field in T radiating energy at rate power in Watts.
    """

    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    q = 1.602176634e-19 # Electron charge, in Coulombs
    c = 299792458 #Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion

    energy_Joules = (energy + me) / JeV
    

    slope = (q * field * c**2)/(2 * math.pi) * (power)/(energy_Joules - power * time)**2
    
    return slope
    
    
def cyc_radius(energy, field, pitch_angle):
    """
    Calculates the instantaneous cyclotron radius of a beta electron given the energy in eV, magnetic field in T, and current pitch angle in degrees.
    """
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    q = 1.602176634e-19 # Electron charge, in Coulombs
    c = 299792458 #Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion
    
    gamma = energy / me + 1
    
    momentum = math.sqrt((pow((energy + me),2) - pow(me,2)) / pow(c,2)) / JeV
    velocity = momentum / (gamma * m)
    vel_perp = velocity * math.sin(pitch_angle * math.pi / 180)
    
    radius = (gamma * m * vel_perp) / (q * field)
    
    return radius
    
def velocity(energy, field, pitch_angle):
    """
    Calculates the transverse and axial velocities of a beta electron given the energy in eV, magnetic field in T, and current pitch angle in degrees.
    """
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    c = 299792458 #Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion
        
    gamma = energy / me + 1
    
    momentum = math.sqrt((pow((energy + me),2) - pow(me,2)) / pow(c,2)) / JeV
    velocity = momentum / (gamma * m)
    
    vel_perp = velocity * math.sin(pitch_angle * math.pi / 180)
    vel_parallel = velocity * math.cos(pitch_angle * math.pi /180)
    
    velocities = [vel_parallel,vel_perp]
    
    return velocities
    
def rad_power(energy,field,pitch_angle):
    """
    Calculates instantaneous radiation power of beta electron
    undergoing cyclotronic motion given initial energy in eV,
    magnetic field in T, and current pitch angle in degrees.
    Uses Lienard's generalization of Larmor's formula.
    """
    
    me = 5.10998950e5 # Electron rest mass, in eV
    c = 299792458 #Speed of light in vacuum, in m/s
    mu0 = 4 * math.pi * 1e-7
    q = 1.602176634e-19 # Electron charge, in Coulombs
       
    
    cyc_freq = energy_to_freq(energy, field)
    cyc_rad =  cyc_radius(energy, field, pitch_angle)
    v_perp = velocity(energy, field, pitch_angle)[1]
    
    gamma = energy / me + 1
    
    #print("cyc_freq:",cyc_freq/1e9)
    #print("cyc_rad:",cyc_rad/1e-3)
    
    coefficient_term = mu0/(6 * math.pi * c) * pow(q,2) * pow(gamma,6)
    acceleration = pow(2*math.pi*cyc_freq,2) * cyc_rad
    
    acceleration_term = pow(acceleration,2) - pow(abs(v_perp * acceleration)/c,2)
    
    #power = mu0/(6 * math.pi * c) * pow(q,2) * pow(acceleration,2)
    power = coefficient_term * acceleration_term
    
    return power
    
def generate_triple_coil_field(main_field=1):
    """
    Generates standard He6-CRES three-coil field profile with appropriate currents for given main_field.
    """

    #He6-CRES three-coil trap
    center_windings = 88 / (4.84e-3 * 2)
    edge_windings = 44 / (4.84e-3 * 2)
    current_per_wire = 0.3106 * main_field

    center_coil = Coil_form(1.15e-2,1.35e-2,-4.84e-3,4.84e-3,0,center_windings,-current_per_wire)

    left_edge_coil = Coil_form(1.15e-2,1.25e-2,-4.84e-3,4.84e-3,-4.5e-2,edge_windings,current_per_wire)
    right_edge_coil = Coil_form(1.15e-2,1.25e-2,-4.84e-3,4.84e-3,4.5e-2,edge_windings,current_per_wire)

    triple_field_profile = Field_profile([center_coil,
                                left_edge_coil,
                                right_edge_coil],
                                main_field)
                                
    return triple_field_profile
    
def generate_triple_coil_trap(main_field=1):
    """
    Generates standard He6-CRES three-coil trap profile with appropriate
    currents for given main_field.
    """

    #He6-CRES three-coil trap
    center_windings = 88 / (4.84e-3 * 2)
    edge_windings = 44 / (4.84e-3 * 2)
    current_per_wire = 0.3106 * main_field

    center_coil = Coil_form(1.15e-2,1.35e-2,-4.84e-3,4.84e-3,0,center_windings,-current_per_wire)

    left_edge_coil = Coil_form(1.15e-2,1.25e-2,-4.84e-3,4.84e-3,-4.5e-2,edge_windings,current_per_wire)
    right_edge_coil = Coil_form(1.15e-2,1.25e-2,-4.84e-3,4.84e-3,4.5e-2,edge_windings,current_per_wire)

    triple_coil_trap = Trap_profile([center_coil,
                            left_edge_coil,
                            right_edge_coil],
                            main_field)
                            
    return triple_coil_trap
    
def field_strength(radius,zpos,main_field=1):
    """
    Calculates the magnetic field strength in T at zpos on the z-axis
    in the decay cell given field of main magnet in T for the standard
    He6-CRES three-coil trap.
    """
    
    triple_field_profile = generate_triple_coil_field(main_field)
                                
    return triple_field_profile.field_strength(radius,zpos)
    
def theta_center(zpos,pitch_angle,trap_profile=0):
    """
    Calculates the pitch angle an electron with current z-coordinate zpos
    in m and current pitch angle pitch_angle in degrees takes at the
    center of given trap.
    """
    
    if trap_profile == 0:
        field_func = field_strength
    else:
        field_func = trap_profile.field_strength
    
    min_field = field_func(0,0) #field's lowest point in center of trap
    curr_field = field_func(0,zpos)
    
    theta_center_calc = math.asin(math.sqrt(min_field / curr_field) * math.sin(pitch_angle * math.pi / 180)) * 180 / math.pi
    
    return theta_center_calc
  
def trap_width_calc(trap_profile=0):
    """
    Calculates the trap width given Field_profile object trap_profile.
    """
    
    field_func = 0
    if trap_profile == 0:
        field_func = field_strength
    else:
        field_func = trap_profile.field_strength
    
    def func(z):
        return -1 * field_func(0,z)
        
    maximum = fmin(func,0,xtol=1e-12)[0]
    print("Trap width: ({},{})".format(-maximum,maximum))
    print("Maximum Field: {}".format(-1 * func(maximum)))
    
    trap_width = (-maximum,maximum)
    return trap_width
    
def min_theta(zpos,trap_profile):
    """
    Calculates the minimum pitch angle theta at which an electron at zpos is trapped given trap_profile
    """
    
    if trap_profile.is_trap == True:
    
        Bmax = trap_profile.field_strength(0,trap_profile.trap_width[1])
        Bz = trap_profile.field_strength(0,zpos)
   
        theta = math.asin(math.sqrt(Bz / Bmax)) * 180 / math.pi
        return theta
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
def curr_pitch_angle(zpos,center_pitch_angle,trap_profile):
    """
    Calculates the current pitch angle of an electron at zpos given center pitch angle and main field strength.
    """
    
    if trap_profile.is_trap == True:
    
        min_field = trap_profile.field_strength(0,0) #field's lowest point in center of trap
        max_z = max_zpos(center_pitch_angle,trap_profile)
        max_reached_field = trap_profile.field_strength(0,max_z)
    
        #print("max z:",max_z)
    
        if (abs(zpos) > max_z):
            print("Electron does not reach given zpos")
            curr_pitch = "FAIL"
        else:
            curr_field = trap_profile.field_strength(0,zpos)
            curr_pitch = math.asin(math.sqrt(curr_field / max_reached_field))*180/math.pi
    
        return curr_pitch
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
def max_radius(energy,center_pitch_angle,trap_profile):
    """
    Calculates the maximum cyclotron radius of a beta electron given the energy in eV, main magnetic field in T, and center pitch angle (pitch angle at center of trap) in degrees.
    """
    
    if trap_profile.is_trap == True:
       
        min_field = trap_profile.field_strength(0,0) #field's lowest point in center of trap
        
        center_radius = cyc_radius(energy,min_field,center_pitch_angle)
        max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
        end_radius = cyc_radius(energy,max_reached_field,pitch_angle = 90)
    
        if (center_radius >= end_radius):
            return center_radius
        else:
            return end_radius
            
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
        
def max_zpos(center_pitch_angle,trap_profile,debug=False):
    """
    Calculates the maximum axial length main field strength in T and center pitch angle (angle at center of trap) in degrees given trap_profile.
    """
    
    if trap_profile.is_trap == True:
    
        if center_pitch_angle < min_theta(0,trap_profile):
            print("WARNING: Electron not trapped")
            return math.inf
            
        else:
    
            min_field = trap_profile.field_strength(0,0) #field's lowest point in center of trap
            max_field = trap_profile.field_strength(0,trap_profile.trap_width[1]) #field's highest point at edge of trap
    
            max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
    
            #initial guess based on treating magnetic well as v-shaped
            slope = (max_field - min_field) / (trap_profile.trap_width[1])
            initial_z = (max_reached_field - min_field) / slope
    
            if initial_z == trap_profile.trap_width[1]:
                return initial_z
    
    
            def func(z):
                curr_field = trap_profile.field_strength(0,z)
                return abs(curr_field - max_reached_field)
            
            max_z = fminbound(func,0,trap_profile.trap_width[1],xtol=1e-15)
            curr_field = trap_profile.field_strength(0,max_z)
            
            if (curr_field > max_reached_field) and debug == True:
                print("Final field greater than max allowed field by: ",curr_field-max_reached_field)
                print("Bmax reached: ", curr_field)
        
            if debug == True:
                print("zlength: ", max_z)
        
            if (max_z > trap_profile.trap_width[1]):
                print("Error Rate: ",max_z - trap_profile.trap_width[1])
            
            return max_z
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
    
def zmax_calc(zmax, trap_profile):
    """
    Calculates the pitch angle that gives zmax as the maximum axial amplitude.
    """
    if trap_profile.is_trap == True:
        def func(center_pitch_angle):
            return abs(zmax - max_zpos(center_pitch_angle, trap_profile))
        
        min_pitch_angle = fminbound(func,min_theta(0,trap_profile),90,xtol=1e-12)
        return min_pitch_angle
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
def avg_cyc_freq(energy,center_pitch_angle,trap_profile):
    """
    Calculates the average cyclotron frquency of an electron given energy in eV, main field strength in T, and center pitch angle (angle at center of trap) in degrees. Returns 0 if electron is not trapped.
    """
    
    if trap_profile.is_trap == True:
        
        min_field = trap_profile.field_strength(0,0) #field's lowest point in center of trap
        min_trapped_angle = min_theta(0,trap_profile)
        max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
       
        if (center_pitch_angle < min_trapped_angle):
            print("Warning: electron not trapped")
            return 0
   
        if (center_pitch_angle == 90):
            avg_frequency = energy_to_freq(energy,min_field)
        else:
            cyc_freqs = []
            
            max_z = max_zpos(center_pitch_angle,trap_profile)
            curr_z = 0
            curr_pitch = center_pitch_angle
            time = 0
            # Kris had set to 1e-12
            dtime = 1e-12
            end = False
            cyc_freqs.append(energy_to_freq(energy,min_field))
        
        
            #plot z-trajectory
            while end != True:
                vel_parallel, vel_perp = velocity(energy,trap_profile.field_strength(0,curr_z), curr_pitch)
                curr_z = curr_z + vel_parallel * dtime
            
                if (curr_z > max_z) or (curr_pitch == 90):
                    end = True
                    if curr_pitch == 90:
                        print("ends with pitch angle 90:", center_pitch_angle, " zpos: ", curr_z)
                else:
                    time = time + dtime
                    curr_field = trap_profile.field_strength(0,curr_z)
                    curr_pitch = curr_pitch_angle(curr_z,center_pitch_angle,trap_profile)
                    cyc_freqs.append(energy_to_freq(energy,curr_field))
                
            avg_frequency = sum(cyc_freqs)/len(cyc_freqs)
    
        return avg_frequency
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
            
def axial_freq(energy, center_pitch_angle,trap_profile):
    """
    Caculates the axial frequency of a trapped electron in Hz. NOT accurate if center_pitch_angle close to minimum trapping angle.
    """
    
    if trap_profile.is_trap == True:
    
        min_field = trap_profile.field_strength(0,0) #field's lowest point in center of trap
        min_trapped_angle = min_theta(0,trap_profile)
        max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
        
        # This is actually not right. Should think about how to change this. 
        if abs(center_pitch_angle) == 90:
            return 0 
    
        if (center_pitch_angle < min_trapped_angle):
            print("Warning: electron not trapped")
            return 0
        
        max_z = max_zpos(center_pitch_angle,trap_profile)
        curr_z = 0
        curr_pitch = center_pitch_angle
        time = 0
        # Kris had dtime @ 1e-12
        dtime = 1e-12
        end = False
    
        print("Calculating axial frequency...")
        #plot z-trajectory
        while end != True:
            vel_parallel, vel_perp = velocity(energy,trap_profile.field_strength(0,curr_z), curr_pitch)
            curr_z = curr_z + vel_parallel * dtime
        
            if vel_parallel < 10000:
                dtime = 1e-14
            elif vel_parallel < 1000:
                dtime = 1e-15
            elif vel_parallel < 100:
                dtime = 1e-18
            elif vel_parallel < 10:
                dtime = 1e-20
    
            if (curr_z > max_z) or (curr_pitch == 90):
                if curr_pitch == 90:
                    print("ends with pitch angle 90:", center_pitch_angle, " zpos: ", curr_z)
                axial_frequency =  1/ ( 4 * time)
                end = True
            else:
                time = time + dtime
                curr_field = trap_profile.field_strength(0,curr_z)
                curr_pitch = curr_pitch_angle(curr_z,center_pitch_angle,trap_profile)
            
        return axial_frequency
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def axial_freq_P_over_E(energy, center_pitch_angle,trap_profile, trap_strength):
    """
    Caculates the axial frequency of a trapped electron in Hz. Here we use the fact that the axial frequency scales as p/E to make this a faster
    function. Need to be sure axial_freq (on which this is based) is accurate.
    """
    if trap_profile.is_trap == True:
        
        # energy to use for lookup_table
        energy_for_lookup_table = 30e3

        min_field = trap_profile.field_strength(0,0) # field's lowest point in center of trap
        min_trapped_angle = min_theta(0,trap_profile)
    
        if (center_pitch_angle < min_trapped_angle):
            print("Warning: electron not trapped")
            return False
        if (center_pitch_angle == 90):
            print("90 degree pitch angle. ")
            return 0
    
        print("Calculating axial frequency using P_over_E lookup_table...")

        # construct lookup_table path
        filename = "P_over_E_lookup_table.json"
        lookup_table_dir = os.getcwd() + "/spec_tools/spec_calc/axial_freq_P_over_E/" 
        lookup_table_path = lookup_table_dir + filename

        # check to see if lookup_table_dir already exists
        if Path(lookup_table_dir).is_dir(): 
            print("The lookup_table directory exists.")

        else: 
            # create lookup_table directory and file if they don't already exist
            Path(lookup_table_dir).mkdir(parents=True, exist_ok=True)
            print("The lookup_table directory doesn't exist. Made the lookup_table directory.")


        # check to see if the file exists, if not make it
        if os.path.exists(lookup_table_path):

            print("File exists.")
            # file_index = file_index+1
            # simulation_results_file = "{}_{}.json".format(simulation_results_file_prefix,file_index)
            # file_path = os.path.join(os.getcwd(),simulation_results_dir,simulation_results_file)

        else:
            print("File doesn't exist. Creating json. Writing theta_array to json.")

            
            # create theta_array to write to json
            # doesn't matter if it's not evenly spaced.   
            center_theta_array = np.arange(min_trapped_angle + .1,89.9,.1)
            # center_theta_array = np.append(center_theta_array, 90)
            

            # array2 = np.arange(1,500,1.675)
            dict_basic = {"energy_for_lookup_table": energy_for_lookup_table, "center_theta_array":center_theta_array.tolist()}
            with open(lookup_table_path,"w") as write_file:
                json.dump(dict_basic,write_file)

        with open(lookup_table_path,"r") as read_file:
                lookup_table_dict = json.load(read_file)

        # sanity checks (take out later)
        if ("center_theta_array" not in lookup_table_dict):
            print ("Error: theta_array not in json file.")

        # convert theta_array list to np array
        center_theta_array = np.array(lookup_table_dict['center_theta_array'])
        # print("center_theta_array: ", center_theta_array)


        if ("{}".format(trap_strength) not in lookup_table_dict):
            print ("Current trap_strength ({}) not found in P_over_E_lookup_table.".format(trap_strength))
            print("Filling P_over_E_lookup_table for current trap strength.")

            curr_trap_strength_axial_freq = np.empty_like(center_theta_array)

            for index, center_theta in enumerate(center_theta_array): 
                curr_trap_strength_axial_freq[index] = axial_freq(energy_for_lookup_table, center_theta,trap_profile)
            print(curr_trap_strength_axial_freq)

            # Append the new dict to the list and overwrite whole file
            with open(lookup_table_path, mode='w') as write_file:
                lookup_table_dict.update({'{}'.format(trap_strength):curr_trap_strength_axial_freq.tolist()})
                json.dump(lookup_table_dict, write_file)

        # now reopen, since the trap_strenght axial_freq array is either already present or was created
        with open(lookup_table_path,"r") as read_file:
                lookup_table_dict = json.load(read_file)


        x = np.array(lookup_table_dict["center_theta_array"])
        y = np.array(lookup_table_dict["{}".format(trap_strength)])
        # print(x,y)
        
        func_interp = interp1d(x, y, kind='cubic', fill_value = 'extrapolate')
        # axial freq at energy_for_lookup_table
        axial_freq_at_fixed_energy = func_interp(center_pitch_angle)

        ## now scale axial_freq_at_fixed_energy according to p/E:  

        me = 5.10998950e5 # Electron rest mass, in eV
        m = 9.1093837015e-31 # Electron rest mass, in kg
        c = 299792458 # Speed of light in vacuum, in m/s
        JeV = 6.241509074e18 # Joule-ev conversion

        p = lambda E : np.sqrt(E**2 - (me)**2)
        


        E_fixed = energy_for_lookup_table + me
        E_input = energy + me

        p_over_E_ratio = (p(E_input)*E_fixed)/(p(E_fixed)*E_input)

        
        # p_over_E_input = p(E_input)/E_input

        axial_freq_P_over_E = axial_freq_at_fixed_energy*p_over_E_ratio
        # print(p_over_E_1/p_over_E_2)
        # print(energy_1_axial_freq/energy_2_axial_freq)

        # print(energy_1_axial_freq*p_over_E_2/p_over_E_1)
        # print(energy_2_axial_freq)

        return axial_freq_P_over_E
        # return axial_frequency
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False


def axial_freq_Ali(energy, center_pitch_angle,trap_profile):
    """
    Caculates the axial frequency of a trapped electron in Hz.
    Uses Eq. 3.23 from Ali Esfahani's Thesis. However, two relativistic adjustments were made: 
    1. The integrand v_z^-1 was changed to be the relativistic velocity. 
    2. mu (Eq. 3.10) was changed to be KE*sin^2(theta)/B so that the mu*B term has a maximal value of KE. 
    """

    # fixed parameters: 
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    c = 299792458 # Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 # Joule-ev conversion

    zmax = max_zpos(center_pitch_angle,trap_profile,debug=True)

    Bmax = trap_profile.field_strength(0,zmax)
    Bmin = trap_profile.field_strength(0,0)

    mu = energy/JeV*(1/Bmax)

    # v_z derived from E_tot = KE + mc^2 = gamma* mc^2
    v_z = lambda z: c*(np.sqrt(1-((energy-mu*trap_profile.field_strength(0,z)*JeV)/me+1)**(-2))) # energy in eV
    integrand = lambda z: (v_z(z))**-1 # KE in eV

    axial_freq_ali = (2/math.pi*integrate.quad(integrand, 0, zmax)[0])**-1

    return axial_freq_ali

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
def axial_freq_new(energy, center_pitch_angle,trap_profile):
    """
    Caculates the axial frequency of a trapped electron in Hz.
    NOT accurate if center_pitch_angle close to minimum trapping angle.
    """
    
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    c = 299792458 #Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion
          
    gamma = energy / me + 1
      
    momentum = math.sqrt((pow((energy + me),2) - pow(me,2)) / pow(c,2)) / JeV
    velocity = momentum / (gamma * m)
    
    if trap_profile.is_trap == True:
    
        if center_pitch_angle == 90:
            return 0
    
        zmax = max_zpos(center_pitch_angle,trap_profile) - 1e-7
        func = lambda z : 1 / math.cos(curr_pitch_angle(z,center_pitch_angle,trap_profile)*math.pi/180)
        axial_frequency = 1/((4/velocity) * romberg(func,0,zmax,divmax=10))
    
        return axial_frequency 
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
def cyl_prob(rmin = 0, rmax = 0.578e-2, zmin = -5e-2, zmax = 5e-2, trap_radius = 0.578e-2, trap_limits = (-5e-2,8.65e-2)):
    """
    Calculates the probability that a created beta in the He6 decay cell will have a radius between rmin and rmax and a z-coordinate between zmin and zmax.
    """
    trap_length = trap_limits[1]-trap_limits[0]
    
    #normalization parameters for PDF functions (radius_PDF = a*r, z_PDF = a)
    rad_norm = 2 / trap_radius**2
    cyl_norm = 1 / trap_length
    
    #restricting input values to trap dimensions
    if (rmin < 0 or rmin >= trap_radius):
        raise ValueError('rmin must be greater than or equal to 0')
    if (rmin >= trap_radius):
        raise ValueError('rmin must be less than {}'.format(trap_radius))
    if (rmax < rmin):
        print(rmax)
        raise ValueError('rmax must be greater than or equal to rmin')
    if (rmax > trap_radius):
        raise ValueError('rmax must be less than or equal to {}'.format(trap_radius))
    
    if (zmin < trap_limits[0]):
        raise ValueError('zmin must be greater than or equal to {}'.format(trap_limits[0]))
    if (zmin > trap_limits[1]):
        raise ValueError('zmin must be less than {}'.format(trap_limits[1]))
    if (zmax < zmin):
        raise ValueError('zmax must be less than or equal to zmin')
    if (zmax > trap_limits[1]):
        raise ValueError('zmax must be less than or equal to {}'.format(trap_limits[1]))
    
    #calculating probabilities
    rad_prob = (rad_norm / 2) * (rmax**2 - rmin**2)
    cyl_prob = (cyl_norm) * (zmax - zmin)
    
    tot_prob = rad_prob * cyl_prob
    
    return tot_prob
    
def sph_prob(theta_min, theta_max, phi_min = 0, phi_max = 360):
    """
    Calculates the probability that a created beta will have an initial momentum direction between theta_min, theta_max and phi_min, phi_max.
    """
    
    phi_norm = 1 / (2 * math.pi)
    theta_norm = 1 / 2
    
    phi_prob = phi_norm * (phi_max  - phi_min) / 180 * math.pi
    theta_prob = theta_norm * ((1 - math.cos(theta_max / 180 * math.pi)) - (1 - math.cos(theta_min  / 180 * math.pi)))
    
    tot_prob = phi_prob * theta_prob
    
    return tot_prob
    
def collision_rate(energy,pressure=0,temperature=0):
    """
    Calculates the number of collisions per second between He6 beta electrons and stray H2 gas as a function of energy in eV
    """
    a0 = 5.29177210903e-11 #Bohr radius From NIST website
    me = 5.10998950e5 #electron rest energy from NIST website
    density = 3.86e16 #Density of H2 gas calculated from Project 8 data; could be different in He6 experiment
    c = 299792458 #Speed of light in vacuum, in m/s
    
    k = 1.380649e-23 #Boltzmann constant from NIST website
    
    if not temperature == 0:
        density =  pressure / (k * temperature)
    print(density)
    gamma = (energy / me) + 1
    
    #parameters for total inelastic and elastic scattering cross-sections
    
    ToverR = (me / (2*13.6)) * (pow(gamma,2) - 1) / pow(gamma,2)
    k0square = (me / 13.6) * (gamma - 1)
    
    el_cross = (math.pi * pow(a0,2)) / k0square * (4.2106 - (2 / k0square))
    inel_cross = (4 * math.pi * pow(a0,2)) / ToverR * (1.5487 * math.log(pow(gamma,2)-1)+17.4615)
    
    rate = density * (el_cross + inel_cross) * math.sqrt((pow(gamma,2)-1)/pow(gamma,2)) * c
    
    return rate
    
def trapping_prob(energy,trap_radius = 0.578e-2,trap_profile=0):
    """
    Calculates the trapping probability of He6 beta electrons given energy in eV, trap radius in meters, and a valid trap_profile.
    """

    if trap_profile == 0:
        trap_profile = generate_triple_coil_trap(main_field=1)

    if trap_profile.is_trap == True:

        #Experiment parameters determined by simulation,fabrication, or experiment
        min_field = trap_profile.field_strength(0,0)#field's lowest point in center of trap
        max_field = trap_profile.field_strength(0,trap_profile.trap_width[1]) #field's highest point at edge of trap
    
        max_z = trap_profile.trap_width[1]
        cyl_edges = [z for z in np.linspace(trap_profile.trap_width[1],0,10000)]
    
        trap_prob = 0
        if energy > 0:
            for k in range(0,len(cyl_edges)-1):
    
                min_trapped_angle = min_theta(cyl_edges[k+1],trap_profile)
                min_center_pitch = theta_center(cyl_edges[k+1],90,trap_profile)
            
                #calculate cylinder sliver probability
                max_trapping_radius = trap_radius - max_radius(energy,min_center_pitch,trap_profile)
            
                if max_trapping_radius < 0:
                    max_trapping_radius = 0
    
                cyl_probability = cyl_prob(rmin=0,rmax=max_trapping_radius,zmin = cyl_edges[k+1],zmax = cyl_edges[k],trap_radius=trap_radius)
            
                #calculate spherical probability
                sph_probability = 2 * sph_prob(min_trapped_angle,90,0,360)
    
                #Total sliver probability
                trap_prob = trap_prob + 2 * cyl_probability * sph_probability
        else:
            trap_prob = 0
    
        return trap_prob
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False

def trapping_pitch(energy,center_pitch_angles = [0,180], trap_radius = 0.578e-2,trap_profile=0):
    """
    Calculates the trapping probability of He6 beta electrons between center pitch angles (angle at center of trap) given main field in Tesla and energy in eV.
    """
    
    if trap_profile == 0:
        trap_profile = generate_triple_coil_trap(main_field=1)

    if trap_profile.is_trap == True:
    
        #Experiment parameters determined by simulation,fabrication, or experiment
        min_field = trap_profile.field_strength(0,0)#field's lowest point in center of trap
        max_field = trap_profile.field_strength(0,trap_profile.trap_width[1]) #field's highest point at edge of trap

        cyl_edges = [z for z in np.linspace(trap_profile.trap_width[1],0,10000)]
    
        trap_prob = 0
    
        if center_pitch_angles[1] >= 90:
            center_pitch_angles[1] = 90
    
        for k in range(0,len(cyl_edges)-1):
    
            min_trapped_angle = min_theta(cyl_edges[k+1],trap_profile)
            curr_field = trap_profile.field_strength(0,cyl_edges[k+1])
        
            min_center_angle = theta_center(cyl_edges[k+1],min_trapped_angle,trap_profile)
            max_center_angle = theta_center(cyl_edges[k+1],90,trap_profile)
        
            #print("zpos: ",cyl_edges[k+1] ," min: ",min_center_angle," max: ",max_center_angle)
        
            if (center_pitch_angles[0] < max_center_angle):
        
                if (center_pitch_angles[0] <= min_center_angle):
                    pitch_low = min_trapped_angle
                else:
                    pitch_low = math.asin(math.sqrt(curr_field/min_field) * math.sin(center_pitch_angles[0] * math.pi / 180)) * 180 / math.pi
                if (center_pitch_angles[1] >= max_center_angle):
                    pitch_high = 90
                else:
                    pitch_high = math.asin(math.sqrt(curr_field/min_field) * math.sin(center_pitch_angles[1] * math.pi / 180)) * 180 / math.pi

                #print("zpos: ",cyl_edges[k+1]," pitch low: ",pitch_low, " pitch high: ",pitch_high)

                if (pitch_high >= min_trapped_angle):
            
                    if (pitch_low < min_trapped_angle):
                        pitch_low = min_trapped_angle
                
                    #calculate cylinder sliver probability
                    max_trapping_radius = trap_radius - max_radius(energy,center_pitch_angles[1],trap_profile)
    
                    cyl_probability = cyl_prob(rmin=0,rmax=max_trapping_radius,zmin = cyl_edges[k+1],zmax = cyl_edges[k],trap_radius=trap_radius)
                
                    #calculate spherical probability
                    sph_probability = 2 * sph_prob(pitch_low,pitch_high,0,360)
    
                    #Total silver probability
                    trap_prob = trap_prob + 2 * cyl_probability * sph_probability
            #elif (center_pitch_angles[0] >= max_center_angle):
                #print("zpos: ",cyl_edges[k+1] ," min: ",min_center_angle," max: ",max_center_angle)
    
        return trap_prob
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
def sideband_calc(avg_cycl_freq, axial_freq, zmax, num_sidebands = 7):
    """
    Calculates relative magnitudes of num_sidebands sidebands from average cyclotron frequency avg_cycl_freq, axial frequency axial_freq, and maximum axial amplitude zmax.
    """
    
    #Physical and mathematical constants
    m = 9.1093837015e-31 # Electron rest mass, in kg.
    c = 299792458 #Speed of light in vacuum, in m/s
    p11prime = 1.84118 #first zero of J1 prime
    
    #fixed experiment parameters
    waveguide_radius = 0.578e-2
    kc = p11prime / waveguide_radius
    
    #calculated parameters
    omega = 2 * math.pi * avg_cycl_freq
    k_wave = omega / c
    beta = math.sqrt(k_wave**2 - kc**2)
    
    phase_vel = omega / beta
    
    mod_index = omega * zmax / phase_vel
    
    #Calculate K factor
    K = 2 * math.pi * avg_cycl_freq * zmax / phase_vel
    
    #Calculate list of (frequency, amplitude) of sidebands
    sidebands= []
    
    for k in range(-num_sidebands,num_sidebands + 1):
    
        if axial_freq == "Indexed":
            freq = k
        else:
            freq = avg_cycl_freq + k * axial_freq
            
        magnitude = abs(ss.jv(k,K))
    
        pair = (freq,magnitude)
        sidebands.append(pair)

    return sidebands, mod_index
    
def mod_index_calc(avg_cycl_freq, zmax):
    """
    Calculates modulation index from average cyclotron frequency avg_cycl_freq and maximum axial amplitude zmax.
    """

    #Physical and mathematical constants
    m = 9.1093837015e-31 # Electron rest mass, in kg.
    c = 299792458 #Speed of light in vacuum, in m/s
    p11prime = 1.84118 #first zero of J1 prime

    #fixed experiment parameters
    waveguide_radius = 0.578e-2
    kc = p11prime / waveguide_radius

    #calculated parameters
    omega = 2 * math.pi * avg_cycl_freq
    k_wave = omega / c
    beta = math.sqrt(k_wave**2 - kc**2)
    
    phase_vel = omega / beta

    mod_index = omega * zmax / phase_vel

    return mod_index
    
def mod_index_finder(mod_index,frequency,trap_profile=0,tolerance=1e-7):
    """
    Calculates minimum pitch angle to obtain given modulation index. Tends to be very slow.
    """
    
    if trap_profile == 0:
        trap_profile = generate_triple_coil_trap(main_field=1)

    if trap_profile.is_trap == True:
    
        main_field = trap_profile.main_field
        energy = freq_to_energy(frequency,main_field)
        min_allowed_theta = min_theta(0,trap_profile)
        
        def func(center_pitch_angle):
    
            maxz = max_zpos(center_pitch_angle, trap_profile)
            avg_freq = avg_cyc_freq(energy,center_pitch_angle,trap_profile)
        
            #print(avg_freq,maxz)
            #print(abs(mod_index - mod_index_calc(avg_freq,maxz)))
            #input("PAUSE")
        
            return abs(mod_index - mod_index_calc(avg_freq,maxz))
        
        min_pitch_angle = fminbound(func,min_allowed_theta,90,xtol=tolerance)
        print("Absolute error in calculated versus desired modulation   index:",func(min_pitch_angle))
        return min_pitch_angle
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
    
def beta_calc(avg_cycl_freq):
    """
    Calculates beta propagation constant in TE11 mode in He6_CRES cylindrical waveguide.
    """

    #Physical and mathematical constants
    m = 9.1093837015e-31 # Electron rest mass, in kg.
    c = 299792458 #Speed of light in vacuum, in m/s
    p11prime = 1.84118 #first zero of J1 prime
    
    #fixed experiment parameters
    waveguide_radius = 0.578e-2
    kc = p11prime / waveguide_radius
    
    #calculated parameters
    omega = 2 * math.pi * avg_cycl_freq
    k_wave = omega / c
    beta = math.sqrt(k_wave**2 - kc**2)
    
    return beta
    
def random_beta_generator(parameter_dict):
    """
    Generate a random beta in the trap with pitch angle between min_theta and max_theta , and initial position (rho,0,z) between min_rho and max_rho and min_z and max_z.
    """
    
    min_rho = parameter_dict["min_rho"]
    max_rho = parameter_dict["max_rho"]
    
    min_z = parameter_dict["min_z"]
    max_z = parameter_dict["max_z"]
    
    min_theta = parameter_dict["min_theta"] * math.pi / 180
    max_theta = parameter_dict["max_theta"] * math.pi / 180
    
    rho_initial = math.sqrt(uniform(0,1) * (max_rho**2 - min_rho**2))
    phi_initial = 2*math.pi * uniform(0,1) * 180/math.pi
    # phi_initial = 0
    z_initial = uniform(min_z,max_z)
    
    u_min = (1-math.cos(min_theta))/2
    u_max = (1-math.cos(max_theta))/2
    
    sphere_theta_initial = math.acos(1 - 2 * (uniform(u_min,u_max))) * 180 / math.pi
    sphere_phi_initial = 2 * math.pi * uniform(0,1)* 180 / math.pi
    
    position = [rho_initial,phi_initial,z_initial]
    direction = [sphere_theta_initial,sphere_phi_initial]
    
    return position, direction
