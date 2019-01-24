##round to nearest step for given storage
#function round2step(number,step)
#    #floors number of times step divides into number
#    x,rem = divrem(number,step)
#    return (Int(x*step))
#end

function round2step(x, base)
    return Int64(base * round(Float64(x)/base))
end
