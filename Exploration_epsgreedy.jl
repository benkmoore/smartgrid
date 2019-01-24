#-> pull = pull out of storage
function action_pull(total_pull, storage_1, step_store1,
    max_rate1, storage_2, step_store2, max_rate2, η_pull_s1, η_pull_s2)
    x = rand(Uniform(0,1))
    #If storage is empty
    if  (storage_1 == 0 && storage_2 == 0)
        pull_s1 = 0
        pull_s2 = 0
        pull_grid = total_pull
    # Accounting for the physical behaviour of sending/pulling energy (not η=100%)
    else
        #Prevent energy stored going negative
        if (x*total_pull) > storage_1 * η_pull_s1
            pull_s1 = storage_1 * η_pull_s1
        else
            pull_s1 = x*total_pull
        end
        if ((1-x)*total_pull) > storage_2 * η_pull_s2
            pull_s2 = storage_2 * η_pull_s2
        else
            pull_s2 = (1-x)*total_pull
        end
        #Enforce max rates
        if pull_s2 > max_rate2
            pull_s2 = max_rate2
        end
        if pull_s1 > max_rate1
            pull_s1 = max_rate1
        end
        #pull_s1 and pull_s2 already account for physical behaviour of device
        pull_grid = total_pull - pull_s1 - pull_s2
        #Account for rounding error in julia
        if pull_grid < 0
            pull_grid = 0
        end
    end
    return pull_s1, pull_s2, pull_grid, x
end

#-> send = send to storage/waste
function action_send(total_send, storage_1, step_store1, CAP1,
    max_rate1, storage_2, step_store2, CAP2, max_rate2, η_send_s1, η_send_s2)
    x = rand(Uniform(0,1))
    send_s1 = x*total_send
    send_s2 = (1-x)*total_send
    send_s1_real = η_send_s1 * send_s1
    send_s2_real = η_send_s2 * send_s2
    #Enforce storage capacity limits
    if (send_s1_real + storage_1) > CAP1
        send_s1_real = CAP1 - storage_1
    end
    if (send_s2_real + storage_2) > CAP2
        send_s2_real = CAP2 - storage_2
    end
    send_s1 = send_s1_real / η_send_s1
    send_s2 = send_s2_real / η_send_s2
    #Enforce max rates
    if send_s2 > max_rate2
        send_s2 = max_rate2
    end
    if send_s1 > max_rate1
        send_s1 = max_rate1
    end
    #Update send_s1 and send_s2 based on changed real physical values

    send_waste = total_send - send_s2 - send_s1
    #Account for rounding error in julia
    if send_waste < 0
        send_waste = 0
    end
    return send_s1, send_s2, send_waste, x
end
