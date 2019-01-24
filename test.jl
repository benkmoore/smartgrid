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
path = "C:\\Users\\bruis\\Google Drive\\Stanford\\Stanford\\Courses\\AA228\\Final Project\\Clone3\\test_data2018.csv"
test_Data = readdlm(path,',')[2:end,:]
num_rows, num_cols = size(test_Data)

for row = 1 : num_rows
    #Parse time step as Int from time string
    test_Data[row,1] = parse(Int, test_Data[row,1][12:13])
end

#Read in Q matrix populated with s,a values
#Q = load("C:\\Users\\Ben\\Documents\\smartgridrl\\globalApprox_Q.jld")["data"]

# Construct Q matrix using 1d q vector and feature matrix:

q_1d_path = "C:\\Users\\bruis\\Google Drive\\Stanford\\Stanford\\Courses\\AA228\\Final Project\\Clone3\\test_q_predictions.csv"
q_1d = readdlm(q_1d_path,',')
test_all_features_path = "C:\\Users\\bruis\\Google Drive\\Stanford\\Stanford\\Courses\\AA228\\Final Project\\Clone3\\test_all_features.csv"
test_all_features = readdlm(test_all_features_path,',')

q= q_1d
X = test_all_features


### Using algorithm from globalApprox lines 72 -104:

#Initialize Q with average q value or 0
factor_init = 0
#factor_init = sum(q)/length(q)
Q = ones(num_states,num_actions)*factor_init

global theta = inv(transpose(X)*X)*transpose(X)*q
print(theta,'\n')

for i = 1:length(q_1d)
    T = trunc(Int,test_all_features[i,2])
    H = trunc(Int,test_all_features[i,3])
    B = trunc(Int,test_all_features[i,4])
    Tstr = PadT(T)
    Bstr = PadBattery(B,CAP_LIBattery)
    Hstr = PadHydro(H,CAP_Hydro)
    send_H = trunc(Int,test_all_features[i,5])
    send_B = trunc(Int,test_all_features[i,6])
    pull_H = trunc(Int,test_all_features[i,7])
    pull_B = trunc(Int,test_all_features[i,8])


    if (send_H + send_B) == 0
        out_H = Pad_H_acts(pull_H,MaxRate_Hydro)
        out_LIB = Pad_LIB_acts(pull_B,MaxRate_LIBattery)
        action_t = actions_dict[string("PULL",out_H, out_LIB)]
    elseif pull_H + pull_B == 0
        in_H = Pad_H_acts(send_H,MaxRate_Hydro)
        in_LIB = Pad_LIB_acts(send_B,MaxRate_LIBattery)
        action_t = actions_dict[string("SEND",in_H, in_LIB)]
    end

    a = action_t
    s = states_dict[string(Hstr,Bstr,Tstr)]
    t = T
    LIB = B
    features = convert(Array{Float64}, [1,t,H,LIB,send_H, send_B,pull_H, pull_B])
    Q[s,a] = sum(theta.*features)
end



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
    #delta = solar_p + wind_p - demand_t
    #Get H and LIB in/out amounts via dicts from optimal action
    action_string = action_array[optimal_action]

    #Update storage amounts
    if (action_string[1:4] == "SEND")
        send_Hydro = parse(Int64,action_string[5])
        send_LIBattery = parse(Int64,action_string[(4+length_H):end])
        #print(action_string,'\t',send_Hydro,'\t',send_LIBattery,'\n')
        #Enforce Phyiscal storage behaviour
        send_Hydro, send_LIBattery = Send_behaviour(send_Hydro,
        Hydro_amount, CAP_Hydro, MaxRate_Hydro, send_LIBattery, LIBattery_amount,
        CAP_LIBattery, MaxRate_LIBattery, η_send_Hydro, η_send_LIBattery)

        delta = solar_p + wind_p - demand_t - send_Hydro - send_LIBattery
        if delta > 0
            waste = delta
        else
            pull_grid=abs(delta)
        end

        Hydro_amount = Hydro_amount + send_Hydro*η_send_Hydro
        LIBattery_amount = LIBattery_amount + send_LIBattery*η_send_LIBattery

    elseif (action_string[1:4] == "PULL")
        pull_Hydro = parse(Int64,action_string[5])
        pull_LIBattery = parse(Int64,action_string[(4+length_H):end])
        #print(action_string,'\t',pull_Hydro,'\t',pull_LIBattery,'\n')
        #Enforce Phyiscal storage behaviour
        pull_Hydro, pull_LIBattery = Pull_behaviour(pull_Hydro, Hydro_amount,
        MaxRate_Hydro, pull_LIBattery, LIBattery_amount,
        MaxRate_LIBattery, η_pull_Hydro, η_pull_LIBattery)

        delta = solar_p + wind_p - demand_t + pull_Hydro + pull_LIBattery
        if delta > 0
            waste = delta
        else
            pull_grid = abs(delta)
        end

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
