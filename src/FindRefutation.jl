using PositivstellensatzRefutations

cert = positivstellensatz_refutation(ARGS...)
if !isnothing(cert)
    println("Refutation certificate found.")
else
    println("No refutation certificate found.")
end