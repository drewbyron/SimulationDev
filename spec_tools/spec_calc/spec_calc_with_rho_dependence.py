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

def theta_center(zpos, rho, pitch_angle, trap_profile):
    """
    Calculates the pitch angle an electron with current z-coordinate zpos
    (m), rho (m) and current pitch angle pitch_angle in degrees takes at the
    center of given trap.
    """
    if trap_profile.is_trap == True:

        Bmin = trap_profile.field_strength_interp(rho,0.0)
        Bcurr = trap_profile.field_strength_interp(rho,zpos)
        
        theta_center_calc = math.asin(math.sqrt(Bmin / Bcurr) * math.sin(pitch_angle * math.pi / 180)) * 180 / math.pi
        theta_center_calc = np.arcsin(math.sqrt(Bmin / Bcurr) * np.sin(pitch_angle * math.pi / 180)) * 180 / math.pi
        return theta_center_calc
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
    
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

def max_radius(energy,center_pitch_angle,rho,trap_profile):
    """
    Calculates the maximum cyclotron radius of a beta electron given the energy in eV, trap_profile, and center pitch angle (pitch angle at center of trap) in degrees.
    """
    
    if trap_profile.is_trap == True:
       
        min_field = trap_profile.field_strength_interp(rho,0) #field's lowest point in center of trap
        
        center_radius = cyc_radius(energy,min_field,center_pitch_angle)
        max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
        end_radius = cyc_radius(energy,max_reached_field,pitch_angle = 90)
    
        if (center_radius >= end_radius):
            return center_radius
        else:
            print("Warning: max_radius is occuring at end of trap (theta=90). Something odd may be going on.")
            return end_radius
            
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False

def min_theta(rho,zpos,trap_profile):
    """
    Calculates the minimum pitch angle theta at which an electron at zpos is trapped given trap_profile
    """
    
    if trap_profile.is_trap == True:
    
        Bmax = trap_profile.field_strength_interp(rho,trap_profile.trap_width[1])
        Bz = trap_profile.field_strength_interp(rho,zpos)
   
        theta = math.asin(math.sqrt(Bz / Bmax)) * 180 / math.pi
        return theta
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False

def max_zpos(center_pitch_angle,rho, trap_profile,debug=False):
    """
    Calculates the maximum axial length from center of trap as a function of center_pitch_angle and rho. 
    """
    
    if trap_profile.is_trap == True:
    
        if center_pitch_angle < min_theta(rho,0,trap_profile):
            print("WARNING: Electron not trapped")
            return math.inf
            
        else:
    
            min_field = trap_profile.field_strength_interp(rho,0) #field's lowest point in center of trap
            max_field = trap_profile.field_strength_interp(rho,trap_profile.trap_width[1]) #field's highest point at edge of trap
    
            max_reached_field = min_field / pow(math.sin(center_pitch_angle * math.pi / 180),2)
    
            # initial guess based on treating magnetic well as v-shaped
            slope = (max_field - min_field) / (trap_profile.trap_width[1])
            initial_z = (max_reached_field - min_field) / slope
    
            if initial_z == trap_profile.trap_width[1]:
                return initial_z
    
    
            def func(z):
                curr_field = trap_profile.field_strength_interp(rho,z)
                return abs(curr_field - max_reached_field)
            
            max_z = fminbound(func,0,trap_profile.trap_width[1],xtol=1e-14)
            curr_field = trap_profile.field_strength_interp(rho,max_z)
            
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

def curr_pitch_angle(rho, zpos,center_pitch_angle,trap_profile):
    """
    Calculates the current pitch angle of an electron at zpos given center pitch angle and main field strength.
    """
    
    if trap_profile.is_trap == True:
    
        min_field = trap_profile.field_strength_interp(rho,0) #field's lowest point in center of trap
        max_z = max_zpos(center_pitch_angle,rho,trap_profile)
        max_reached_field = trap_profile.field_strength_interp(rho,max_z)
    
        #print("max z:",max_z)
    
        if (abs(zpos) > max_z):
            print("Electron does not reach given zpos")
            curr_pitch = "FAIL"
        else:
            curr_field = trap_profile.field_strength_interp(rho,zpos)
            curr_pitch = math.asin(math.sqrt(curr_field / max_reached_field))*180/math.pi
    
        return curr_pitch
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False

def velocity(energy, pitch_angle):
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


def axial_freq(energy, center_pitch_angle,rho, trap_profile):
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
            center_pitch_angle = 89.9999
    
        zmax = max_zpos(center_pitch_angle, rho, trap_profile)
        B = lambda z : trap_profile.field_strength_interp(rho, z) 
        Bmax = trap_profile.field_strength_interp(rho,zmax)

        sec_theta = lambda z : (1-B(z)/Bmax)**(-.5)  # secant of theta as function of z. Use conserved mu to derive. 
        T_a = 4/velocity*integrate.quad(sec_theta,0,zmax, epsrel = 10**-2)[0]
        axial_frequency = 1/T_a
    
        return axial_frequency 
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False


def avg_cyc_freq(energy, axial_freq, center_pitch_angle, rho, trap_profile):
    """
    Calculates the average cyclotron frquency of an electron given energy in eV, main field strength in T, and center pitch angle (angle at center of trap) in degrees. Returns 0 if electron is not trapped.
    """
    
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    c = 299792458 #Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion
    q = 1.602176634e-19 # Electron charge, in Coulombs

    gamma = energy / me + 1
    momentum = math.sqrt((pow((energy + me),2) - pow(me,2)) / pow(c,2)) / JeV
    velocity = momentum / (gamma * m)

    if trap_profile.is_trap == True:
        
        Bmin = trap_profile.field_strength_interp(rho,0) #field's lowest point in center of trap
        min_trapped_angle = min_theta(rho, 0, trap_profile)
        

        if (center_pitch_angle < min_trapped_angle):
            print("Warning: electron not trapped")
            return 0
   
        if (center_pitch_angle == 90):
            avg_frequency = energy_to_freq(energy,Bmin)
        else:
            zmax = max_zpos(center_pitch_angle, rho, trap_profile)
            B = lambda z : trap_profile.field_strength_interp(rho, z) 
            Bmax = trap_profile.field_strength_interp(rho,zmax)
            integrand = lambda z : B(z)*((1-B(z)/Bmax)**(-.5))

            # use the time average of the magnetic field to find the avg cycl freq
            avg_cyc_freq = 4*q*axial_freq/(2*math.pi*momentum)*integrate.quad(integrand,0,zmax, epsrel = 10**-2)[0]

        return avg_cyc_freq
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False

def t(energy, zpos, center_pitch_angle, rho, trap_profile):
    """
    Caculates the time for electron to travel from z = 0 to z.
    NOT accurate if center_pitch_angle close to minimum trapping angle.
    """
    
    me = 5.10998950e5 # Electron rest mass, in eV
    m = 9.1093837015e-31 # Electron rest mass, in kg
    c = 299792458 # Speed of light in vacuum, in m/s
    JeV = 6.241509074e18 #Joule-ev conversion
          
    gamma = energy / me + 1
      
    momentum = math.sqrt((pow((energy + me),2) - pow(me,2)) / pow(c,2)) / JeV
    velocity = momentum / (gamma * m)
    
    if trap_profile.is_trap == True:
    
        if center_pitch_angle == 90:
            center_pitch_angle = 89.9999
    
        zmax = max_zpos(center_pitch_angle, rho, trap_profile)
        B = lambda z : trap_profile.field_strength_interp(rho, z) 
        Bmax = trap_profile.field_strength_interp(rho,zmax)

        sec_theta = lambda z : (1-B(z)/Bmax)**(-.5)  # secant of theta as function of z. Use conserved mu to derive. 
        t = 1/velocity*integrate.quad(sec_theta,0,zpos, epsrel = 10**-2)[0]
        
        if zpos > zmax: 
            print("ERROR: zpos equal to or larger than zmax.")

        return t 
        
    else:
        print("ERROR: Given trap profile is not a valid trap")
        return False
