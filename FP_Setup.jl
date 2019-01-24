using DataFrames;
using CSV;
using DelimitedFiles;
using Plots;
using Random;
using Distributions;

include("Exploration_epsgreedy.jl")
include("Round.jl")
include("Pad.jl")

#path = "C:\\Users\\Ben\\Documents\\smartgridrl\\FINLAND_ENERGY.csv"
ext = pwd()
path = "$ext/FINLAND_ENERGY.csv"
energy_Data = readdlm(path,',')[2:end,:]
num_rows, num_cols = size(energy_Data)


function storage_depreciation(storage_level, self_discharge_day)
    return storage_level * (1 - self_discharge_day)^(1/24)
end

#HYDRO - STATE (STORAGE 1 FOR OUR MODEL)
Hydro_step = 1
CAP_Hydro = 50
storage_Hydro = transpose(collect(0:Hydro_step:CAP_Hydro)) #capacity: i.e. what it can hold - Y
#HYDRO - ACTION
MaxRate_Hydro = 5
out_Hydro = transpose(collect(0:Hydro_step:MaxRate_Hydro)) #power rating: what it can take up/supply - My
in_Hydro = transpose(collect(0:Hydro_step:MaxRate_Hydro)) #what it can pull/take in - Ny
#HYDRO - PHYSICAL PROPERTIES
discharge_Hydro_day = 0.01/100
#Assuming storing has same efficiency as regenerating
η_pull_Hydro = sqrt(80/100)
η_send_Hydro = sqrt(80/100)

#LIB - STATE (STORAGE 2 FOR OUR MODEL)
LIBattery_step = 1
CAP_LIBattery = 30
storage_LIBattery = transpose(collect(0:LIBattery_step:CAP_LIBattery)) #capacity: i.e. what it can hold - X
#LIB - ACTION
MaxRate_LIBattery = 10
out_LIBattery = transpose(collect(0:LIBattery_step:MaxRate_LIBattery)) #power rating: what it can take up/supply - Mx
in_LIBattery = transpose(collect(0:LIBattery_step:MaxRate_LIBattery)) #what it can pull/take in - Nx
#LIB - PHYSICAL PROPERTIES
discharge_LIBattery_day = 0.3/100
#Assuming storing has same efficiency as regenerating
η_pull_LIBattery = sqrt(65/100)
η_send_LIBattery = sqrt(65/100)




#Encode Actions always out with out/in with in. Never mix eg out with in
actions_dict = Dict()
#Action in pairs -> send in to storage
for H_act_in = 0:Hydro_step:MaxRate_Hydro
    if H_act_in == 0
        global act = 1
    end
    for LIB_act_in = 0:LIBattery_step:MaxRate_LIBattery
        #Pad each action out to length of max sized action
        H_act_in = Pad_H_acts(H_act_in,MaxRate_Hydro)
        LIB_act_in = Pad_LIB_acts(LIB_act_in,MaxRate_LIBattery)
        actions_dict[string("SEND",H_act_in,LIB_act_in)] = act
        global act = act + 1
    end
end
#Action out pairs -> pull out of storage
for H_act_out=0:Hydro_step:MaxRate_Hydro
    for LIB_act_out = 0:LIBattery_step:MaxRate_LIBattery
        H_act_out = Pad_H_acts(H_act_out,MaxRate_Hydro)
        LIB_act_out = Pad_LIB_acts(LIB_act_out,MaxRate_LIBattery)
        actions_dict[string("PULL",H_act_out,LIB_act_out)] = act
        global act = act + 1
    end
end

#Num states = all possible timesteps {0,1,2...,23} x all possible storage amounts
num_states = (length(storage_Hydro))*(length(storage_LIBattery))*(24) #time range length = 24
num_actions = length(actions_dict)
Q = zeros(num_states, num_actions)

#Encode States in dict
states_dict = Dict()
for H=0:Hydro_step:CAP_Hydro
    for B=0:LIBattery_step:CAP_LIBattery
        for T=0:1:23
            if H == 0
                global state_num = 1
            end
            #Pad out state string
            T = PadT(T)
            B = PadBattery(B,CAP_LIBattery)
            H = PadHydro(H,CAP_Hydro)
            states_dict[string(H,B,T)] = state_num
            state_num = state_num + 1
        end
    end
end

function get_action(solar_p,wind_p,demand_t,Hydro_amount, Hydro_step, MaxRate_Hydro, LIBattery_amount, LIBattery_step, MaxRate_LIBattery,
                    CAP_Hydro, CAP_LIBattery, actions_dict, η_pull_Hydro, η_pull_LIBattery, η_send_Hydro, η_send_LIBattery)
    delta = solar_p + wind_p - demand_t
    if delta < 0 #energy deficit = renew.s not meeting demand -> pull = pull out of storage/grid
        pull_Hydro, pull_LIBattery, pull_grid, x = action_pull(abs(delta), Hydro_amount, Hydro_step, MaxRate_Hydro, LIBattery_amount, LIBattery_step, MaxRate_LIBattery, η_pull_Hydro, η_pull_LIBattery)
        #print("PULL",'\t',pull_Hydro,'\t', pull_LIBattery,'\t', pull_grid,'\n')

        #Update storage levels based on physical behaviour of the storages
        #Added in physical behaviour
#        Hydro_amount = Hydro_amount - pull_Hydro/η_pull_Hydro
#        LIBattery_amount = LIBattery_amount - pull_LIBattery/η_pull_LIBattery
        #Get action from dict
        out_H = Pad_H_acts(round2step(pull_Hydro, Hydro_step),MaxRate_Hydro)
        out_LIB = Pad_LIB_acts(round2step(pull_LIBattery, Hydro_step),MaxRate_LIBattery)
        action_t = actions_dict[string("PULL",out_H,out_LIB)]
    elseif delta > 0 #energy surplus = renew.s greater than demand -> send = send to storage/waste
        send_Hydro, send_LIBattery, waste,x = action_send(delta, Hydro_amount, Hydro_step, CAP_Hydro, MaxRate_Hydro, LIBattery_amount, LIBattery_step, CAP_LIBattery, MaxRate_LIBattery, η_send_Hydro, η_send_LIBattery)
        #print("SEND",'\t',send_Hydro,'\t', send_LIBattery,'\t', waste,'\n')
        #Update storage levels

        #Get action from dict
        in_H = Pad_H_acts(round2step(send_Hydro, Hydro_step),MaxRate_Hydro)
        in_LIB = Pad_LIB_acts(round2step(send_LIBattery, LIBattery_step),MaxRate_LIBattery)
        action_t = actions_dict[string("SEND",in_H,in_LIB)]
    end
    return action_t
end
