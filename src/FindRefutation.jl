using PositivstellensatzRefutations

max_half_degree = parse(Int, ARGS[1])
num_variables = parse(Int, ARGS[2])
cert = positivstellensatz_refutation(max_half_degree, num_variables, ARGS[3:end]...)
if !isnothing(cert)
    println("Refutation certificate found.")
else
    println("No refutation certificate found.")
end