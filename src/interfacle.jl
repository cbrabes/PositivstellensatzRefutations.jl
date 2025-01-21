using DynamicPolynomials
using MultivariatePolynomials
using DelimitedFiles
include("positivstellensatz.jl")
function build_polynomial(coeffs::AbstractVector, basis::MonomialVector)
    return  polynomial(coeffs, basis[1:length(coeffs)])
end

function read_coeff_file(source_file)
    C = readdlm(source_file, BigInt)
end

function write_refutation_certificate(output_path::String, cert::pos_certificate)
    for (index, u) in enumerate(cert.SOS_U_factors)
        path = joinpath(output_path, string("SOS_U_factor_",index-1, ".txt"))
        open(path, "w") do io
            writedlm(io, u)
        end
    end
    path = joinpath(output_path, "D_factor.txt")
    open(path, "w") do io
        writedlm(io, cert.diagonally_dominant_gram)
    end

    if !isnothing(cert.equality_coefficients)
        path = joinpath(output_path, "equality_coefficients.txt")
        open(path, "w") do io
            writedlm(io, zip(cert.equality_coefficients...))
        end
    end
    path = joinpath(output_path, "constant_term.txt")
    open(path, "w") do io
        writedlm(io, [cert.constant_term,])
    end
end 


function positivstellensatz_refutation(max_half_degree::Integer, num_variables::Integer, inequalities_file::String, equalities_file::String, output_path::String)
    @polyvar x[1:num_variables] monomial_order = Graded{Reverse{LexOrder}}
    if !isfile(inequalities_file)
        error("Inequalities coefficient file does not exist.")
    end
    if !isfile(equalities_file)
        error("Equalities coefficient file does not exist.")
    end
    output_dir_path = joinpath(output_path, "refutation_output")
    rm(output_dir_path, force=true, recursive=true)
    mkpath(output_dir_path)

    ineq_coeffs = read_coeff_file(inequalities_file)
    eq_coeffs = read_coeff_file(equalities_file)

    mono_basis = monomials(x, 0:2*max_half_degree)
    inequalities = [build_polynomial(row, mono_basis) for row in eachrow(ineq_coeffs)]
    equalities = [build_polynomial(row, mono_basis) for row in eachrow(eq_coeffs)]

    certificate =pos_refutation(max_half_degree,vec(x),inequalities, equalities)
    if isnothing(certificate)
        rm(output_dir_path, force=true, recursive=true)
        return nothing
    end
    write_refutation_certificate(output_dir_path, certificate)
    return certificate
end

function positivstellensatz_refutation(max_half_degree::Integer, num_variables::Integer, inequalities_file::String, output_path::String)
    @polyvar x[1:num_variables] monomial_order = Graded{Reverse{LexOrder}}
    if !isfile(inequalities_file)
        error("Inequalities coefficient file does not exist.")
    end
    output_dir_path = joinpath(output_path, "refutation_output")
    rm(output_dir_path, force=true, recursive=true)
    mkpath(output_dir_path)

    ineq_coeffs = read_coeff_file(inequalities_file)

    mono_basis = monomials(x, 0:2*max_half_degree)
    inequalities = [build_polynomial(row, mono_basis) for row in eachrow(ineq_coeffs)]

    certificate =pos_refutation(max_half_degree,vec(x),inequalities)
    if isnothing(certificate)
        rm(output_dir_path, force=true, recursive=true)
        return nothing
    end
    write_refutation_certificate(output_dir_path, certificate)
    return certificate
end
