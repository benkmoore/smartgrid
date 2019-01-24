using DataFrames;
using CSV;
using DelimitedFiles;
using Plots;
using Random;
using Distributions;

include("Exploration_epsgreedy.jl")
include("Round.jl")
include("Pad.jl")
include("FP_Setup.jl")

path = "C:\\Users\\Ben\\Documents\\smartgridrl\\FINLAND_ENERGY.csv"
energy_Data = readdlm(path,',')[2:end,:]
num_rows, num_cols = size(energy_Data)

for row = 1 : num_rows
    #Parse time step as Int from time string
    energy_Data[row,1] = parse(Int, energy_Data[row,1][12:13])
end

Q = zeros(num_states, num_actions)
N = zeros(num_states, num_actions)

for t = 1 : num_rows
    if t == 1
        #Storage Initialise = 0
        global Hydro_amount = 0
        global LIBattery_amount = 0
    end
    time_day = energy_Data[t,1]
    cost_GP = energy_Data[t,5]
    demand_t = energy_Data[t,6]
    solar_p = energy_Data[t,7]
    wind_p = energy_Data[t,8]

    #1) read in encoded state based on storage_Hydro, storage_LIBattery & time day => S
    time_day = PadT(time_day)
    #To be done: state_t = state_prime (and delete following three lines)
    Hydro = PadHydro(round2step(Hydro_amount, Hydro_step),CAP_Hydro)
    LIBattery = PadBattery(round2step(LIBattery_amount, LIBattery_step),CAP_LIBattery)
    state_t = states_dict[string(Hydro,LIBattery,time_day)]

    #2) choose Action based on exploration strategy
    #Re Initialise waste and pull_grid each iteration
    pull_grid = 0
    waste = 0
    delta = solar_p + wind_p - demand_t
    if delta < 0 #energy deficit = renew.s not meeting demand -> pull = pull out of storage/grid
        pull_Hydro, pull_LIBattery, pull_grid, x = action_pull(abs(delta), Hydro_amount,
        Hydro_step, MaxRate_Hydro, LIBattery_amount, LIBattery_step, MaxRate_LIBattery,
        η_pull_Hydro, η_pull_LIBattery)
        #print("PULL",'\t',pull_Hydro,'\t', pull_LIBattery,'\t', pull_grid,'\n')
        #Update storage levels
        Hydro_amount = Hydro_amount - pull_Hydro/η_pull_Hydro
        LIBattery_amount = LIBattery_amount - pull_LIBattery/η_pull_LIBattery
        #Get action from dict
        out_H = Pad_H_acts(round2step(pull_Hydro, Hydro_step), MaxRate_Hydro)
        out_LIB = Pad_LIB_acts(round2step(pull_LIBattery, LIBattery_step), MaxRate_LIBattery)
        action_t = actions_dict[string("PULL",out_H,out_LIB)]
    elseif delta > 0 #energy surplus = renew.s greater than demand -> send = send to storage/waste
        send_Hydro, send_LIBattery, waste,x = action_send(delta, Hydro_amount, Hydro_step, CAP_Hydro,
        MaxRate_Hydro, LIBattery_amount, LIBattery_step, CAP_LIBattery, MaxRate_LIBattery,
        η_send_Hydro, η_send_LIBattery)
        #print("SEND",'\t',send_Hydro,'\t', send_LIBattery,'\t', waste,'\n')
        #Update storage levels
        Hydro_amount = Hydro_amount + send_Hydro*η_send_Hydro
        LIBattery_amount = LIBattery_amount + send_LIBattery*η_send_LIBattery
        #Get action from dict
        in_H = Pad_H_acts(round2step(send_Hydro, Hydro_step),MaxRate_Hydro)
        in_LIB = Pad_LIB_acts(round2step(send_LIBattery, LIBattery_step),MaxRate_LIBattery)
        action_t = actions_dict[string("SEND",in_H,in_LIB)]
    end
    #print(Hydro_amount,'\t',LIBattery_amount,'\n')
    Hydro_amount = storage_depreciation(Hydro_amount, discharge_Hydro_day)
    LIBattery_amount = storage_depreciation(LIBattery_amount, discharge_LIBattery_day)

    #3) calculate reward
    r = -(pull_grid*cost_GP) - (waste*cost_GP)

    #4) update Q via Bellman
    #Find next state and observations at next state
    if t != num_rows
        time_day_prime = energy_Data[t+1,1]
        time_day_prime = PadT(time_day_prime)
        Hydro_prime = PadHydro(round2step(Hydro_amount, Hydro_step),CAP_Hydro)
        LIBattery_prime = PadBattery(round2step(LIBattery_amount, LIBattery_step),CAP_LIBattery)
        state_prime = states_dict[string(Hydro_prime,LIBattery_prime,time_day_prime)]

        N[state_t,action_t] = N[state_t,action_t] + 1
        Q_star = maximum(Q[state_prime,:])
        trace_delta = r + Q_star - Q[state_t,action_t]
        alpha = 0.5; lambda = 0.8
        for s = 1 : 1 : num_states
            Q[s,:] = Q[s,:] .+ (alpha*trace_delta*N[s,:])
            N[s,:] = lambda*N[s,:]
        end
    end

end

using JLD;
save("C:\\Users\\Ben\\Documents\\smartgridrl\\Q_QL_traces.jld", "data", Q)
