module PositivstellensatzRefutations
using PrecompileTools: @setup_workload, @compile_workload
using DynamicPolynomials
include("interfacle.jl")

export positivstellensatz_refutation, pos_refutation

@setup_workload begin
    @polyvar x[1:3] monomial_order = Graded{Reverse{LexOrder}}
    g1 = 1 - x[3]^2 - (x[1] + 4)^2 - (x[2])^2 
    example_h = [(x[2]^2) + (0),]
    g2 = 1 - x[3]^2 - (x[1] - 4)^2 - (x[2])^2
    g3 = (1+x[1]^2+x[2]^2+x[3]^2)
    example_gs = [g1,g2,g3 ]
    example_gs = [polynomial(g, BigInt) for g in example_gs]
    example_h = [polynomial(h, BigInt) for h in example_h]
    @compile_workload begin
        certificate = pos_refutation(8, vec(x), example_gs, example_h)
        certificate = pos_refutation(8, vec(x), example_gs)
    end
end

end