using DataFrames;
using CSV;
using DelimitedFiles;
using Plots;

include("storage.jl")
include("FP_Setup.jl")
include("Exploration_epsgreedy.jl")

ext = pwd()
path = "$ext/test_data2018.csv"
energy_Data = readdlm(path,',')[2:end,:]
num_rows, num_cols = size(energy_Data)

for row = 1 : num_rows
    #Parse time step as Int from time string
    energy_Data[row,1] = parse(Int, energy_Data[row,1][12:13])
end

for t = 1 : num_rows
    if t == 1
        global r = zeros(num_rows,1)
        global GP = zeros(num_rows,1)
        global W = zeros(num_rows,1)
        global H = zeros(num_rows,1)
        global L = zeros(num_rows,1)
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
    Hydro = PadHydro(round2step(Hydro_amount, Hydro_step),CAP_Hydro)
    LIBattery = PadBattery(round2step(LIBattery_amount, LIBattery_step),CAP_LIBattery)
    state_t = states_dict[string(Hydro,LIBattery,time_day)]

    #2) choose Random action
    pull_grid = 0
    waste = 0
    #Binary random determines whether to pull or send
    decision = rand(0:1)
    x,y = rand(2)
    if decision==1
        #sending
        send_Hydro = x*MaxRate_Hydro
        send_LIBattery = y*MaxRate_LIBattery
        send_Hydro, send_LIBattery = Send_behaviour( send_Hydro,
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

    else #decision ==0
        pull_Hydro = x*MaxRate_Hydro
        pull_LIBattery = y*MaxRate_LIBattery

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
    delta = solar_p + wind_p - demand_t

    #3) calculate reward
    Hydro_amount = storage_depreciation(Hydro_amount, discharge_Hydro_day)
    LIBattery_amount = storage_depreciation(LIBattery_amount, discharge_LIBattery_day)
    r[t,1] = -(pull_grid*cost_GP) - (waste*cost_GP)

    #Collect all data for standard output
    GP[t,1] = pull_grid
    W[t,1] = waste
    H[t,1] = Hydro_amount
    L[t,1] = LIBattery_amount

end
print(sum(r))
#=
t = collect(1:num_rows)

NET2 = W .- GP .- H_delta .- L_delta

g2 = plot(t,NET2, label="net")
g2 =plot!(t,W, label="waste")
g2 = plot!(t,GP, label="pull grid")
g2 = plot!(t,H_delta, label = "pull hydrogen")
g2 =plot!(t,L_delta,title="Waste & Pull Energy",label="pull LIB")
savefig("$ext/HR_results/WasteAndPull.png")

Solar = energy_Data[:,7]
Wind = energy_Data[:,8]
Demand = energy_Data[:,6]
NET1 = Solar .+ Wind .- Demand

g1 =plot(t,NET1,label="net")
g1 =plot!(t,Wind,label="wind")
g1 =plot!(t,Solar,label="solar")
g1 =plot!(t,-Demand,label="demand",title="Energy Demand & Production")
savefig("$ext/HR_results/EnergyDAndP.png")

Hydrogen_accumulated = zeros(2000,1)
plot(t[1:2000], L[1:2000],title="LIB",label="Amount in LIB")
plot!(t[1:2000,1],L_delta[1:2000,1],label="LIB delta")
savefig("$ext/HR_results/LIB_zoom.png")
plot(t[1:2000], H[1:2000],title="Hydrogen",label="Amount in Hydrogen",)
plot!(t[1:2000,1],H_delta[1:2000,1],label="Hydrogen delta")
savefig("$ext/HR_results/Hydro_zoom.png")
