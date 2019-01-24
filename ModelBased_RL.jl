## Model Based RL

# Initializations of N(s,a,s'), rho(s,a) and Q(s,a)
using Distributions;

include("FP_Setup.jl")
# FP_Setup.jl includes the following:
    #include("Exploration_epsgreedy.jl")
    #include("Round.jl")
    #include("Pad.jl")

#N_sasp = zeros(num_states, num_actions)
N_sasp = zeros(num_states, num_actions, num_states)

# Uniform Initialization
N_sasp_init_count = 100

LI_Nsasp_upperbound = ones(1,MaxRate_LIBattery)*N_sasp_init_count
Hydro_Nsasp_upperbound = ones(1,MaxRate_Hydro)*N_sasp_init_count



rho_sa = zeros(num_states, num_actions)

Q_sa = zeros(num_states, num_actions)

## For each state, add Nsasp_upperbound on top and below. in the sp dimension of N_sasp tensor
for H=0:Hydro_step:CAP_Hydro
    for B=0:LIBattery_step:CAP_LIBattery
        for T=0:1:23
            T = PadT(T)
            B = PadBattery(B,CAP_LIBattery)
            H = PadHydro(H,CAP_Hydro)
#Fetch state number
            states_dict[string(H,B,T)] = state_num
            N_sasp[]



## Algorithm

T = 0
state_num = 1 # initial state T,B,H are zero
s0 = state_num # initial state

#H=0:Hydro_step:CAP_Hydr
#B=0:LIBattery_step:CAP_LIBattery
#T=0:1:23

#T = PadT(T)
#B = PadBattery(B,CAP_LIBattery)
#H = PadHydro(H,CAP_Hydro)
#states_dict[string(H,B,T)] = state_num
