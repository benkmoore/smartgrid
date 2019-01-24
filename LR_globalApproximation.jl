using LinearAlgebra
include("FP_Setup.jl")
#Read in Q matrix populated with s,a values
#Q = load("C:\\Users\\Ben\\Documents\\smartgridrl\\Q_matrix_QLearning.jld")["data"]
Q = load("C:\\Users\\Ben\\Documents\\smartgridrl\\Q_SL.jld")["data"]
num_states, num_actions = size(Q)

#Change state & action dicts to array
states_array = convert(Array{Any} ,zeros(num_states,1))
for key in keys(states_dict)
        states_array[states_dict[key],1] = key
end
action_array = convert(Array{Any} ,zeros(num_actions,1))
for key in keys(actions_dict)
        action_array[actions_dict[key],1] = key
end

#Get weights for theta vector
num_data_pts = count(i -> i!=0,Q)
q = zeros(num_data_pts,1)
X = zeros(num_data_pts,16)
X[:,1] = 1 .+ X[:,1]
H_str_length = length(string(CAP_Hydro))
LIB_str_length = length(string(CAP_LIBattery))
for s = 1 : 1 : num_states
    if s == 1
        global i = 1
    end
    for a = 1 : 1 : num_actions
        if Q[s,a] != 0
            #Populate q matrix
            q[i,1] = Q[s,a]
            #Get t, H, L from str(H,B,T)
            state_str = states_array[s,1]
            H = parse(Int64,state_str[1:(H_str_length)])
            LIB = parse(Int64,state_str[(1+H_str_length):(H_str_length+LIB_str_length)])
            t = parse(Int64,state_str[(1+H_str_length+LIB_str_length):end])
            #Get action: in s_or_p: pull = - and send = +
            action_string = action_array[a,1]
            if (action_string[1:4] == "SEND")
                send_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                send_LIBattery = parse(Int64,action_string[(5+length_H):end])
                send = send_Hydro + send_LIBattery
                pull = 0
            elseif (action_string[1:4] == "PULL")
                pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
                pull = pull_Hydro + pull_LIBattery
                send = 0
            end
            X[i,2] = t
            X[i,3] = H
            X[i,4] = LIB
            X[i,5] = send
            X[i,6] = pull
            X[i,7] = t^2
            X[i,8] = H^2
            X[i,9] = LIB^2
            X[i,10] = send^2
            X[i,11] = pull^2
            X[i,12] = t^3
            X[i,13] = H^3
            X[i,14] = LIB^3
            X[i,15] = send^3
            X[i,16] = pull^3
            i = i + 1
        end
    end
end

global theta = inv(transpose(X)*X)*transpose(X)*q
print(theta,'\n')

for s = 1 : 1 : num_states
    if s == 1
        global i = 1
    end
    for a = 1 : 1 : num_actions
        if Q[s,a] == 0
            #Get t, H, L from str(H,B,T)
            state_str = states_array[s,1]
            H = parse(Int64,state_str[1:(H_str_length)])
            LIB = parse(Int64,state_str[(1+H_str_length):(H_str_length+LIB_str_length)])
            t = parse(Int64,state_str[(1+H_str_length+LIB_str_length):end])
            #Get action
            action_string = action_array[a,1]
            if (action_string[1:4] == "SEND")
                send_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                send_LIBattery = parse(Int64,action_string[(5+length_H):end])
                send = send_Hydro + send_LIBattery
                pull = 0
            elseif (action_string[1:4] == "PULL")
                pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
                pull = pull_Hydro + pull_LIBattery
                send = 0
            end
            features = convert(Array{Float64}, [1,t,H,LIB,send,pull,t^2,H^2,LIB^2,send^2,pull^2,t^3,H^3,LIB^3,send^3,pull^3])
            Q[s,a] = sum(theta.*features)
            i = i + 1
        end
    end
end

#Save updated Q
using JLD;
save("C:\\Users\\Ben\\Documents\\smartgridrl\\globalApprox_Q.jld", "data", Q)
