#!/usr/bin/env python
# coding: utf-8

# # README
# 
# Ingredients:
# - model : Field Epileptor Model 
# - connectivity : weights and lengths as computed above
# - local connectivity : using gdists as above (missing area-scaling?) + Heaviside coupling
# - integrator : euler deterministic
# - monitors : Raw or temporal average
# 
# 
# Downscaling to be further checked. 
# 
# Possible issues found:
# - length matrix with non zero values on the diagonal (e.g. ca1-ca1 length higher than ca1-ca2 length)
# - gdist shorter than length (one is across the surface, one along the white fibers tracts respectively)
# - weights non-matching with the circle shown above (e.g. ca1 more connected to parahippocampus than enthorinal cortex)

# # Load libraries and data

# ## Load libraries

#---------------------
# Importing libraries
#---------------------

import numpy as np                             # np
from pathlib import Path                       # Path
from scipy.sparse import load_npz, diags       # For Loading npz

# TVB stuff
from tvb.simulator.lab import integrators, monitors, simulator, \
                            connectivity, \
                            surfaces, cortex, \
                            region_mapping, local_connectivity,\
                            equations, patterns

# Freesurfer, to read the parcellation files
from nibabel.freesurfer.io import read_geometry, read_morph_data#, read_annot

# Import utils library
import os
import sys
utils_dir = os.path.abspath("../utils_py")
sys.path.append(utils_dir)
from utils import Heaviside, Heaviside_Sparse, SpatEpi, \
                  delete_vertices_from_triangular_mesh, CorticalSurface_new, \
                  Connectivity_Sparse, LocalConnectivity_new, \
                  LaplaceKernel


# ## Select Working directories

#-----------
# Data dirs
#-----------
# Low res
lowres_dir = Path("/Data/Low_resolution/")
if Path.exists(lowres_dir):
    print("Found lowres directory")
else:
    print("Lowres dir not found")

# High res
highres_dir = Path("/Data/high_density_brain/")
if Path.exists(highres_dir):
    print("Found highres directory")
else:
    print("Highres dir not found")

# Results
results_dir = Path("/Results/TVB/Lowres_Epileptor/")

if Path.exists(results_dir):
    print("Found results directory")
else:
    print("results dir not found")


# # Set some global parameters

# Time parameters
dt = 0.1             # Time step
maxdt = 0.01         # Max time step, for variable time-step simulations (eg: Runge-Kutta)

# For stochastic simulations
navg = 0             # Noise average
nsigma = 0.035*2     # Noise amplitude 
nsig = np.array([    # noise amplitude array
    nsigma,   # x1
    0.0,      # y1
    0.0,      # z
    nsigma,   # x2
    0.0,      # y2
    0.0       # g
])

#-------------------
# Simulation length
#-------------------
simulation_length = 1e3*60*5     # Simulation length in ms

#--------------
# Excitability
#--------------
# healthy_excitability = -2.3    # healthy excitability to initialize the model
healthy_excitability = float(sys.argv[1])
print(f'value for healthy_excitability is {healthy_excitability}')


# # Set the brain's volume and surface

# ## Load regions and nodes, and select ROIs

#------------------------
# Load regions and nodes
#------------------------
# List of regions (an ordered list of names, eg: "ctx-lh-bankssts"
region_names = list(np.genfromtxt(f"{utils_dir}/fs_default_HC.txt", usecols=1, dtype=str))
n_regions = len(region_names)

# Mapping of nodes to regions
# # Eg, if region 9 is 'ctx-lh-lateraloccipital' and node 0 corresponds to region 9, 
# # then node 0 corresponds to 'ctx-lh-lateraloccipital')
# ### Highres
# hr_region_map = np.genfromtxt(highres_dir/"T1w/Diffusion/vert2vert_region_mapping_resliced.txt") # For highres, the mapping is on a file
# n_hr_nodes=len(hr_region_map)   # Total number of available nodes for high-resolution
### Lowres
lr_region_map = np.arange(n_regions) # For lowres, the mapping isthe identity (vertex i corresponds to region i)
n_lr_nodes = len(lr_region_map) # Total number of available nodes for low-resolution

#----------------
# Select my ROIs
#----------------
selected_regions = region_names
# selected_regions = ['ctx-rh-fusiform','ctx-rh-parahippocampal','ctx-rh-entorhinal','ctx-rh-Subiculum', 
#         'ctx-rh-CA1', 'ctx-rh-CA2', 'ctx-rh-CA3', 'ctx-rh-CA4', 'ctx-rh-Dentate-gyrus']

# Create a mask for selected regions
# highres_mask = np.zeros(n_hr_nodes, dtype=bool)  # initialize a mask for highres
lowres_mask = np.zeros(n_lr_nodes, dtype=bool)   # initialize a mask for lowres
# For each selected region, select the nodes that correspond to those regions
for region in selected_regions:
    region_number = region_names.index(region)              # What number corresponds to this region?
    # highres_mask[hr_region_map == region_number] = True     # add the nodes maping to that number (highres)
    lowres_mask[region_number] = True                       # add the nodes maping to that number (lowres)

# A masked mapping of nodes to regions
# highres_region_map = np.array(hr_region_map[highres_mask],dtype='i') #list(np.argwhere(highres_mask==True))
# n_highres_nodes = len(highres_region_map)
lowres_region_map  = np.array(lr_region_map[lowres_mask],dtype='i')  #list(np.argwhere(lowres_mask==True)) 
n_lowres_nodes = len(lowres_region_map)

# Output some info on my selection:
# print(f"Selected {len(selected_regions)} regions; which correspond to \
# {n_highres_nodes} nodes in highres and {n_lowres_nodes} nodes in lowres")


# ## Global Connectivity

# #### Weights, lengths and centers

#---------
# Weights 
#---------
# Highres
# hr_weights = load_npz(highres_dir/"T1w/Diffusion/vert2vert_weights_15M.npz")
# Lowres
lr_weights  = np.load(lowres_dir/"weight_lr.npy")

#---------
# Lengths
#---------
# Highres
# hr_lengths = load_npz(highres_dir/"T1w/Diffusion/vert2vert_lengths_15M.npz")
# Lowres
lr_lengths  = np.load(lowres_dir/"length_lr.npy")

#---------
# Centers
#---------
# For highres, define al centers to (0, 0, 0)
# hr_centers = np.zeros((n_hr_nodes,3))               # Why not set it to the vertices' positions though?
# Lowres (read from dataset) 
lr_centers = np.load(lowres_dir/"centers_lr.npy")  # I guess gdist will be infered from this(?)

#------
# Mask
#------
# Highres
# highres_lengths = hr_lengths[highres_mask,:][:,highres_mask]      # Lengths
# highres_weights = hr_weights[highres_mask,:][:,highres_mask]      # Weights
# Lowres
lowres_lengths = lr_lengths[lowres_mask,:][:,lowres_mask]      # Lengths
lowres_weights = lr_weights[lowres_mask,:][:,lowres_mask]      # Weights
lowres_centers = lr_centers[lowres_mask]                       # Centers

#-----------------------
# Normalize weights (?)
#-----------------------
# highres_weights/=np.max(highres_weights)  # Normalize (or don't)
lowres_weights/=np.max(lowres_weights)*100


# #### Propagation speed and configure connectivity objects

#-------------------
# Propagation speed
#-------------------
# Highres
# highres_speed = 3.6
lowres_speed  = 3.6

#--------------
# Connectivity
#--------------
### Highres
# Connectivity_Sparse is derived from connectivity.Connectivity, adapted to use npz data
# highres_connectivity = Connectivity_Sparse(weights=highres_weights,   # weights 
#                     tract_lengths=highres_lengths,                    # lengths
#                     speed=np.array([highres_speed]),                  # Propagation speed (in m/s?)
#                     #region_labels=np.array([str(i) for i in range(n_regions)]),  # labels for regions
#                     centres = np.zeros((n_highres_nodes,3)),          # the center of each region is (0,0,0)
#                     cortical=np.ones((n_highres_nodes),dtype=bool))   # consider all as cortical (?)
# highres_connectivity.configure()
# highres_connectivity.set_idelays(dt)   # Convert time delays (from ms?) to time step units. 

### Lowres
lowres_connectivity=connectivity.Connectivity(weights=lowres_weights, # weights              
                               tract_lengths=lowres_lengths,          # lengths
                               speed=np.array([lowres_speed]),        # Propagation speed (in m/s)
                               region_labels=np.array([str(i) for i in range(n_lowres_nodes)]), # labels for regions
                               centres=lowres_centers                 # centers
                                )
lowres_connectivity.configure()


# ## Coupling

#----------
# Coupling
#----------
# Coulpling: A function rescaling the connectivity according to pre- and/or post-synaptic processing, 
# # and/or a rescaling of the connectivity.
# # It is evaluated after passing throught the long-range connectivity and before the local dynamics. 

# Heaviside_Sparse is defined in utils. It's derived from SparseCoupling and implements a rescaled 
# # heaviside function.
# # "a" rescales postsynaptically and theta filters through a heaviside function 
# # (0 if x<theta, 0.5 if x=theta, 1 if x>theta). 
# coupl = Heaviside_Sparse(a=np.array([0.01]), b=np.array([0]))
### Highres
# highres_coupling = Heaviside_Sparse(a=np.array([0.01]), theta=np.array([-1])) 
### Lowres
lowres_coupling = Heaviside(a=np.array([1]), theta=np.array([-1]))


# # Cortex (for the highres data)

# ## Cortical surface

# ## Region mapping

# ## Local connectivity

# # Define the model's parameters and initial conditions

# ## Model parameters

#-----------------
# Model of choice 
#-----------------
epileptors=SpatEpi()
epileptors.variables_of_interest = ['u1', 'u2', 's', 'q1', 'q2', 'g', 'q1 - u1'] 
# aka                                x1,   y1,   z,   x2,   y2,   g,    x2-x1
# epileptors.variables_of_interest=['u1','q1 - u1','s']

#-------------------------
# Parameters of the model
#-------------------------
# # I'm using the parameters from the reference paper (Proix, Jirsa, et al (2018). Predicting the spatiotemporal diversity of seizure....
# Global parameters of the model
epileptors.tt = np.array([0.17])              # "Time scaling of the whole system"

# Parameters for subsystem 1 (u1, fast activity):
# # For u1
epileptors.Iext = np.array([3.1])             # I_{ext1}
epileptors.gamma11 = np.array([1.0])          # gamma_{11}
epileptors.theta11 = np.array([-1.0])         # theta_{11}
# gamma_glob somehow defines the heterogeneous (aka global) connectivity; 
# # that is, \sum_j \gamma_{het,j} w_{het} S(u_{1,j} \theta_{het})
epileptors.gamma_glob=np.array([0.3])          # gamma_{het, j} (same for all j
# Not really sure how \gamma_{het,j}; w_{het}; theta_{het}, 
# # and even S(u_i, \theta_{i,j}) are defined. Very confusing stuff

# Parameters for subsystem 2 (q1 and q2):
# # for q1
epileptors.Iext2 = np.array([0.45])           # I_{ext2}
epileptors.gamma22 = np.array([1.0])          # gamma_{22}
epileptors.theta22 = np.array([-0.5])         # theta_{22}
# # For q2
epileptors.tau2 = np.array([10.0])            # tau_2

# For the permitivity variable (s)
epileptors.x0 = np.ones((n_lowres_nodes)) * healthy_excitability # Excitability  
epileptors.tau0 = np.array([2857.0])          # tau_0

# Parameters for g:
epileptors.gamma12 = np.array([0.1])          # gamma_{12}
epileptors.theta12 = np.array([-1.0])         # theta_{12}
# tau_[12} is fixed, but not really sure to which value, check equations for ydot[5]; 
# # is y[5] really g?
# same for a_{12}, I think it's fixed to 0.3 (hence the 0.003 = a_{12}/tau_{12}) in 0.003, but it should be fixed to 3

# ## Epileptogenic, healthy and propagation zones

# ----------------------------------------------------
# Define epileptogenic, healthy and propagation zones 
# ----------------------------------------------------
# IMPORTANT! TO DO!
# # These values need to be further analysed. 
# # From the paper, epileptic zone has x0 >-2.91 (or 2.1? paper is confusing), 
# # propagation zone has -2.91>x>(missing value in the paper?), 
# # while non-affected nodes for lower values 
# # paper: https://www.nature.com/articles/s41467-018-02973-y

# In the "classic" epileptor model: : 
# an epileptogenic zone (EZ) has high excitability x0>-1.6, 
# a healthy zone (HZ) has low excitability x_0<-2.06 and 
# a propagation zone (PZ) has excitability between these two values

# Initialize all regions to a healthy value (-2.1)
#zones_x0 = {region:-2.3 for region in selected_regions}
# I need a list of size n_highres_nodes (n_lowres_nodes), initilized to a healthy value 
# # It's already there in epilleptors.x0
# # then create a mask of the same size, with False on all nodes, except the ones 
# # corresponding to the indices linked to a list to epilleptogenic zones

# A dict of epilleptogenic regions with a different value each
epilleptogenic_regions = {
    "ctx-rh-Subiculum": -1.9,          
    "ctx-rh-CA1" : -1.6,
    'ctx-rh-parahippocampal': -1.9,
    'ctx-rh-CA2': -1.9,
    'ctx-rh-CA3': -1.9,
    'ctx-rh-CA4': -1.9,
    'ctx-rh-entorhinal': -1.9 }

# For each epilleptogenic region, change the excitability
for region, excitability in epilleptogenic_regions.items():
    # List of indices that correspond to this region
    region_number = region_names.index(region)                 #What's the region number
    epileptors.x0[lowres_region_map==region_number] = excitability  #Write there


# ## Initial conditions

#--------------------
# Initial conditions
#--------------------
# IMPORTANT! Initial Conditions

ic =[-1.5, -11,   3,  -0.9,  0.3, -0.1] # stable point, interictal state
#   ['u1', 'u2', 's', 'q1', 'q2', 'g',  'q1-u1'] 
#ic.append(ic[3]-ic[0])                # q1-u1

n_variables = len(ic)                               # 6; q1-u1 should be automatic
n_modes = 1                                         # one mode (for now xP)
max_delay = lowres_connectivity.delays.max() + 1    # 
# shape: (n_var, n_regions, n_mode)                 #
ic_full = np.array(ic).reshape(n_variables, 1, 1)   #
# Initial conditions for all variables on all regions and modes
ic_full = np.tile(ic_full, (1, n_regions, n_modes)) 

# few nodes starting in an ictal state --- 
# # can be avoided and wait for a spontaneous seizure onset
onset_regions=['ctx-rh-entorhinal','ctx-rh-Dentate-gyrus', 'ctx-rh-CA1']
onset_mask = np.zeros((n_regions),dtype=bool)
for reg in onset_regions:
    onset_mask[region_names.index(reg)]=1
ic_full[0,onset_mask,:] = 0  #set u1 value to 0
ic_full[1,onset_mask,:] = -9 #set u2 value to -5
ic_full[2,onset_mask,:]= 2  #set s  value to 2
ic_full[3,onset_mask,:] = 0. #set q1 value to 0
ic_full[4,onset_mask,:] = 0. #set 12 value to 0
ic_full[5,onset_mask,:] = 0  #set g  value to 0
#ic_full[5,:][onset_mask] = 0  #set q1-u1 

# ic_full = np.repeat(ic, n_vert).reshape(( len(ic), n_vert, 1))
# Repeat initial conditions on all time history
ic_full = np.repeat(ic_full[np.newaxis, ...], max_delay, axis=0)


ic_full[:,      1,       79,    0   ]
#     delay, variable, region, mode


# # Configure the simulator

# ## Select integrator and monitors

#------------
# Integrator
#------------
# Euler deterministic should be enough), but the paper I'm checking on the epilleptor uses Euler–Maruyama
euler_integrator = integrators.EulerDeterministic(dt=dt)

#----------
# Monitors 
#----------
# Let's go with Raw
Raw_monitor = monitors.Raw()
my_monitors = [Raw_monitor]

# ## Setup a stimulus
stimulated_regions = ['ctx-lh-caudalanteriorcingulate'] 
stimulated_nodes = [region_names.index(region) for region in stimulated_regions]

stim_weights = np.zeros((lowres_connectivity.number_of_regions, 1))
stim_weights[stimulated_nodes] = np.array([1.0])[:, np.newaxis]
eqn_t = equations.PulseTrain()
eqn_t.parameters["onset"] = 400.0 # ms
eqn_t.parameters["tau"]   = 10.0  # ms
eqn_t.parameters["T"]     = 1.0e7 # 0.002kHz repetition frequency
eqn_t.parameters["amp"]   = 1.    # Pulse amplitude

pulse_stimulus = patterns.StimuliRegion(temporal = eqn_t,
                                  connectivity = lowres_connectivity, 
                                  weight = stim_weights)


# ## Setup the simulator

sim = simulator.Simulator(
    model=epileptors,                   # model
    connectivity=lowres_connectivity,   # connectivity
    coupling=lowres_coupling,           # coupling
    integrator=euler_integrator,        # integrator
    monitors=my_monitors,                  # monitors
    initial_conditions=ic_full.copy(),  # initial conditions
    stimulus = pulse_stimulus,          # stimulus
    simulation_length=simulation_length # <-- specify length here (in ms)
)

# Configure the simulator
sim.configure()

# Convert simulation length (in ms) to time steps
n_simulation_steps = int(simulation_length / euler_integrator.dt)

print("Running simulation for", simulation_length, "ms -> steps:", n_simulation_steps)


# # Run the simulation

# Run (restituisce una lista di arrays, un elemento per monitor)
raw_data = sim.run()
# raw_data = sim.run(sim_length)


# ## Export results

# raw data is a list of length number of monitors (1 if only raw monitors)
# raw_data[mon_id] is a tuple of length 2 (0 for timestamps, 1 for states)
# # the timestamp is in ms and starts at max_delay
# raw_data[mon_id][1] is an array of shape (tsteps,n_var_of_interest,n_regions,1)

timestamps = raw_data[0][0]
np.save(results_dir/f"time_stamps_{healthy_excitability}", timestamps)

raw_monitor_results = raw_data[0][1]
np.save(results_dir/f"raw_monitor_{healthy_excitability}",raw_monitor_results)

