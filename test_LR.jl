#TEST SCRIPT
using DataFrames;
using CSV;
using DelimitedFiles;
using Plots;
using Random;
using Distributions;
using JLD;

include("storage.jl")
include("FP_Setup.jl")
action_array = convert(Array{Any} ,zeros(num_actions,1))
for key in keys(actions_dict)
        action_array[actions_dict[key],1] = key
end

#Read in data to test on
path = "C:\\Users\\Ben\\Documents\\smartgridrl\\test_data2018.csv"
test_Data = readdlm(path,',')[2:end,:]
num_rows, num_cols = size(test_Data)

for row = 1 : num_rows
    #Parse time step as Int from time string
    test_Data[row,1] = parse(Int, test_Data[row,1][12:13])
end

#Read in Q matrix populated with s,a values - Q matrix of alg. u want to test
Q = load("C:\\Users\\Ben\\Documents\\smartgridrl\\globalApprox_Q.jld")["data"]

#Find optimal strategy, Initialise 1D vector
optimal_a = convert(Array{Int64} ,zeros(num_states,1))
for state = 1 : num_states
    #Create num_states x 1 dim vector
    maxQ, optimal_a[state,1] = findmax(Q[state,:])
end

for t = 1 : num_rows
    if t == 1
        #Initialise = 0
        global Hydro_amount = 0
        global LIBattery_amount = 0
        global r_total = 0
        global length_H = length(string(MaxRate_Hydro))
        global length_LIB = length(string(MaxRate_LIBattery))
    end
    time_day = test_Data[t,1]
    cost_GP = test_Data[t,5]
    demand_t = test_Data[t,6]
    solar_p = test_Data[t,7]
    wind_p = test_Data[t,8]

    #Find the state at timestep t
    time_day = PadT(time_day)
    Hydro = PadHydro(round2step(Hydro_amount, Hydro_step),CAP_Hydro)
    LIBattery = PadBattery(round2step(LIBattery_amount, LIBattery_step),CAP_LIBattery)
    state_t = states_dict[string(Hydro,LIBattery,time_day)]

    #Find action from max_a Q
    optimal_action = optimal_a[state_t,1]
    #Re Initialise waste and pull_grid each iteration
    pull_grid = 0
    waste = 0
    delta = solar_p + wind_p - demand_t
    #Get H and LIB in/out amounts via dicts from optimal action
    action_string = action_array[optimal_action]

    #Update storage amounts
    if (action_string[1:4] == "SEND") && (delta > 0)
        send_Hydro = parse(Int64,action_string[(5):(4+length_H)])
        send_LIBattery = parse(Int64,action_string[(5+length_H):end])
        #print(action_string,'\t',send_Hydro,'\t',send_LIBattery,'\n')
        #Enforce Phyiscal storage behaviour
        send_Hydro, send_LIBattery, waste = Send_behaviour(delta, send_Hydro,
        Hydro_amount, CAP_Hydro, MaxRate_Hydro, send_LIBattery, LIBattery_amount,
        CAP_LIBattery, MaxRate_LIBattery, η_send_Hydro, η_send_LIBattery)

        Hydro_amount = Hydro_amount + send_Hydro*η_send_Hydro
        LIBattery_amount = LIBattery_amount + send_LIBattery*η_send_LIBattery

    elseif (action_string[1:4] == "PULL") && (delta < 0)
        pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
        pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
        #print(action_string,'\t',pull_Hydro,'\t',pull_LIBattery,'\n')
        #Enforce Phyiscal storage behaviour
        pull_Hydro, pull_LIBattery, pull_grid = Pull_behaviour(abs(delta),
        pull_Hydro, Hydro_amount, MaxRate_Hydro, pull_LIBattery, LIBattery_amount,
        MaxRate_LIBattery, η_pull_Hydro, η_pull_LIBattery)

        Hydro_amount = Hydro_amount - pull_Hydro/η_pull_Hydro
        LIBattery_amount = LIBattery_amount - pull_LIBattery/η_pull_LIBattery
    end
    #print(Hydro_amount,'\t',LIBattery_amount,'\n')
    Hydro_amount = storage_depreciation(Hydro_amount, discharge_Hydro_day)
    LIBattery_amount = storage_depreciation(LIBattery_amount, discharge_LIBattery_day)

    #Calculate total reward/cost
    r_total = r_total - (pull_grid*cost_GP) - (waste*cost_GP)
end
print(r_total)
