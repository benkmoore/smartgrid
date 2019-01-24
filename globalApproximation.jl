using JLD
using DataFrames
include("FP_Setup.jl")

include("QLearning.jl")

Q_orig = Q


#X = readdlm(path,',')[2:end,:]
energy_Data = readdlm(path,',')[2:end,:]

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
X = zeros(num_data_pts,8) #or 16
X[:,1] = 1 .+ X[:,1]
H_str_length = length(string(CAP_Hydro))
length_H = H_str_length
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
                pull_Hydro= 0
                pull_LIBattery = 0
            elseif (action_string[1:4] == "PULL")
                pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
                pull = pull_Hydro + pull_LIBattery
                send = 0
                send_Hydro = 0
                send_LIBattery = 0

            end
            X[i,2] = t
            X[i,3] = H
            X[i,4] = LIB
            X[i,5] = send_Hydro
            X[i,6] = send_LIBattery
            X[i,7] = pull_Hydro
            X[i,8] = pull_LIBattery
#=
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
            =#

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
                pull_Hydro = 0
                pull_LIBattery = 0
            elseif (action_string[1:4] == "PULL")
                pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
                pull = pull_Hydro + pull_LIBattery
                send = 0
                send_Hydro = 0
                send_LIBattery = 0
            end
            features = convert(Array{Float64}, [1,t,H,LIB,send,pull,t^2,H^2,LIB^2,send^2,pull^2,t^3,H^3,LIB^3,send^3,pull^3])
            features_NN = convert(Array{Float64}, [1,t,H,LIB,send_Hydro,send_LIBattery, pull_Hydro,pull_LIBattery])
            Q[s,a] = sum(theta.*features)
            i = i + 1
        end
    end
end

#Save updated Q
using JLD;
save("C:\\Users\\bruis\\Google Drive\\Stanford\\Stanford\\Courses\\AA228\\Final Project\\Clone3\\globalApprox_Q.jld", "data", Q)
CSV.write("featureX.csv",  DataFrame(X), writeheader=false)
CSV.write("global_q.csv",  DataFrame(q), writeheader=false)

####
#Create q_nv and X_nv, complements to the previous q and X
####

####

# Reset Q to non-apprroximated Q
Q = Q_orig

num_data_pts = count(i -> i==0,Q)
q_nv = zeros(num_data_pts,1)
X_nv = zeros(num_data_pts,8)
X_nv[:,1] = 1 .+ X_nv[:,1]

for s = 1 : 1 : num_states
    if s == 1
        global i = 1
    end
    for a = 1 : 1 : num_actions
        if Q[s,a] != 0
            #Populate q matrix
            q_nv[i,1] = Q[s,a]
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
                X_nv[i,5] = send_Hydro
                X_nv[i,6] = send_LIBattery
                X_nv[i,7] = 0
                X_nv[i,8] = 0
            elseif (action_string[1:4] == "PULL")
                pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
                pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
                pull = pull_Hydro + pull_LIBattery
                send = 0
                X_nv[i,5] = 0
                X_nv[i,6] = 0
                X_nv[i,7] = pull_Hydro
                X_nv[i,8] = pull_LIBattery
            end
            X_nv[i,2] = t
            X_nv[i,3] = H
            X_nv[i,4] = LIB
            #X_nv[i,5] = send_Hydro
            #X_nv[i,6] = send_send_LIBattery
            #X_nv[i,7] = pull_Hydro
            #X_nv[i,8] = pull_LIBattery
            i = i + 1
        end
    end
end

CSV.write("featureX_nv.csv",  DataFrame(X_nv), writeheader=false)
CSV.write("global_q_nv.csv",  DataFrame(q_nv), writeheader=false)




####

####
#Create the complete q and X
####

# Reset Q to non-apprroximated Q
Q = Q_orig

num_data_pts = num_states*num_actions
q_all = zeros(num_data_pts,1)
X_all = zeros(num_data_pts,8)
X_all[:,1] = 1 .+ X_all[:,1]

for s = 1 : 1 : num_states
    if s == 1
        global i = 1
    end
    for a = 1 : 1 : num_actions
        #Populate q matrix
        q_all[i,1] = Q[s,a]
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
            X_all[i,5] = send_Hydro
            X_all[i,6] = send_LIBattery
            X_all[i,7] = 0
            X_all[i,8] = 0
        elseif (action_string[1:4] == "PULL")
            pull_Hydro = parse(Int64,action_string[(5):(4+length_H)])
            pull_LIBattery = parse(Int64,action_string[(5+length_H):end])
            pull = pull_Hydro + pull_LIBattery
            send = 0
            X_all[i,5] = 0
            X_all[i,6] = 0
            X_all[i,7] = pull_Hydro
            X_all[i,8] = pull_LIBattery
        end
        X_all[i,2] = t
        X_all[i,3] = H
        X_all[i,4] = LIB
        #X_all[i,5] = send_Hydro
        #X_all[i,6] = send_LIBattery
        #X_all[i,7] = pull_Hydro
        #X_all[i,8] = pull_LIBattery
        i = i + 1

    end
end

CSV.write("featureX_all.csv",  DataFrame(X_all), writeheader=false)
CSV.write("global_q_all.csv",  DataFrame(q_all), writeheader=false)

CSV.write("Q_orig.csv",  DataFrame(Q_orig), writeheader=false)
