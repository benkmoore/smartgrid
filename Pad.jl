function PadHydro(H,max_Hydro)
    if length(string(H)) < length(string(max_Hydro))
        while length(string(H)) < length(string(max_Hydro))
            H = string(0,H)
        end
    end
    return H
end

function PadBattery(B,max_LIBattery)
    if length(string(B)) < length(string(max_LIBattery))
        while length(string(B)) < length(string(max_LIBattery))
            B = string(0,B)
        end
    end
    return B
end

function PadT(T)
    if length(string(T)) != 2
        T = string(0,T)
    end
    return T
end

function Pad_H_acts(H_act,max_rate_H)
    if length(string(H_act)) < length(string(max_rate_H))
        while length(string(H_act)) < length(string(max_rate_H))
            H_act = string(0,H_act)
        end
    end
    return H_act
end

function Pad_LIB_acts(LIB_act,max_rate_LIB)
    if length(string(LIB_act)) < length(string(max_rate_LIB))
        while length(string(LIB_act)) < length(string(max_rate_LIB))
            LIB_act = string(0,LIB_act)
        end
    end
    return LIB_act
end
