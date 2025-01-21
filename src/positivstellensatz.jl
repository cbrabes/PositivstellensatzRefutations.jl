using Hypatia
import DynamicPolynomials: @polyvar, Variable
using StaticPolynomials
import MultivariatePolynomials: Polynomial
using MultivariatePolynomials
using MultivariateBases
using LinearAlgebra
using JuMP

include("gram_matrix_extraction.jl")
include("ChebVandermond.jl")
struct pos_certificate
    max_half_degree::Integer
    SOS_U_factors::Vector{UpperTriangular{BigInt, Adjoint{BigInt, Matrix{BigInt}}}}
    diagonally_dominant_gram::Symmetric{Rational{BigInt}, Matrix{Rational{BigInt}}}
    constant_term::BigInt
    equality_coefficients::Union{Vector{Vector{BigInt}}, Nothing}
    x::Vector{<:Variable}
    ineq_polynomials::Vector{<:AbstractPolynomial}
    eq_polynomials::Union{Vector{<:AbstractPolynomial}, Nothing}
end

struct NoCertificateFound <: Exception end


function create_pos_sos_dual_problem(max_degree::Int, poly_vars::Vector{<:Variable}, G::Vector{<:AbstractPolynomial}, H::Vector{<:AbstractPolynomial})
    n_vars = length(poly_vars)
    box = Hypatia.PolyUtils.FreeDomain{Float64}(n_vars)
    #(U, pts, P0, V, w) = Hypatia.PolyUtils.interp_box(box, n_vars, cld(max_degree,2), true, false)
    (U, pts, P0, V) = fast_SOS_data(Float64, n_vars, cld(max_degree,2))
    model = JuMP.Model(() -> Hypatia.Optimizer{Float64}(verbose = true, tol_slow = 1e-2, tol_rel_opt = 1e-13))
    @variable(model, y[1:U])
    c0 = @constraint(model, y in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, P0, true))
    @objective(model, Min, -sum(y))
    static_pts = [vec(r) for r in eachrow(pts)]
    GSystem = PolynomialSystem(G, variables = poly_vars)
    HSystem = PolynomialSystem(H, variables = poly_vars)

    eval_ineqs(x) = evaluate(GSystem, x)
    eval_eqs(x) = evaluate(HSystem, x)
    ineq_vals = stack(eval_ineqs.(static_pts))
    eq_vals = stack(eval_eqs.(static_pts))
    
    Ps = [P0[1][:,1:Hypatia.PolyUtils.get_L(n_vars, cld((max_degree-maxdegree(g)), 2))] for g in G]
    c = @constraint(model, [i = 1:length(G)], y.*ineq_vals[i, :] in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, [Ps[i],], true))

    Vs = [V[:,1:Hypatia.PolyUtils.get_L(n_vars, max_degree-maxdegree(h))] for h in H]
    lc = @constraint(model, [i = 1:length(H)], Vs[i]'*(eq_vals[i,:].*y)==0)
    return (model, P0[1], Ps, Vs, ineq_vals, eq_vals, c0, c, lc)
end

function create_pos_sos_dual_problem_ineq_only(max_degree::Int, poly_vars::Vector{<:Variable}, G::Vector{<:AbstractPolynomial})
    n_vars = length(poly_vars)
    box = Hypatia.PolyUtils.FreeDomain{Float64}(n_vars)
    (U, pts, P0, V) = fast_SOS_data(Float64, n_vars, cld(max_degree,2))
    model = JuMP.Model(() -> Hypatia.Optimizer{Float64}(verbose = false, tol_slow = 1e-2, tol_rel_opt = 1e-13))
    @variable(model, y[1:U])
    c0 = @constraint(model, y in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, P0, true))
    @objective(model, Min, -sum(y))
    static_pts = [vec(r) for r in eachrow(pts)]
    GSystem = PolynomialSystem(G, variables = poly_vars)

    eval_ineqs(x) = evaluate(GSystem, x)
    ineq_vals = stack(eval_ineqs.(static_pts))
    
    Ps = [P0[1][:,1:Hypatia.PolyUtils.get_L(n_vars, cld((max_degree-maxdegree(g)), 2))] for g in G]
    c = @constraint(model, [i = 1:length(G)], y.*ineq_vals[i, :] in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, [Ps[i],], true))


    return (model, P0[1], Ps, ineq_vals, c0, c)
end

function create_pos_sos_dual_problem_eq_only(max_degree::Int, poly_vars::Vector{<:Variable}, H::Vector{<:AbstractPolynomial})
    n_vars = length(poly_vars)
    box = Hypatia.PolyUtils.FreeDomain{Float64}(n_vars)
    (U, pts, P0, V, w) = Hypatia.PolyUtils.interp_box(box, n_vars, cld(max_degree,2), true, false)
    model = JuMP.Model(() -> Hypatia.Optimizer{Float64}(verbose = false, tol_slow = 1e-2, tol_rel_opt = 1e-13))
    @variable(model, y[1:U])
    c0 = @constraint(model, y in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, P0, true))
    @objective(model, Min, -sum(y))
    static_pts = [vec(r) for r in eachrow(pts)]
    HSystem = PolynomialSystem(H, variables = poly_vars)

    eval_eqs(x) = evaluate(HSystem, x)
    eq_vals = stack(eval_eqs.(static_pts))
    

    Vs = [V[:,1:Hypatia.PolyUtils.get_L(n_vars, max_degree-maxdegree(h))] for h in H]
    lc = @constraint(model, [i = 1:length(H)], Vs[i]'*(eq_vals[i,:].*y)==0)
    return (model, P0[1], eq_vals, c0, lc)
end

function calc_SOS_Matrix(P::Matrix{Float64},primal_point::Vector{Float64}, dual_point::Vector{Float64})
    U = length(primal_point)
    F = cholesky(Hermitian(P'*diagm(primal_point)*P, :L))
    Hess = factorize(Hermitian(((P/F)*P').^2, :L))
    w = Hess\dual_point 
    lambda_i_p = cholesky(Hermitian(P'*diagm(primal_point)*P,:L))
    lambda_w = P'*diagm(w)*P
    return Hermitian((lambda_i_p\lambda_w)/lambda_i_p,:L)
end



function monomial_to_cheb_vandermonde(x::Vector{<:Variable}, max_degree)
    monomial_basis = [p for p in maxdegree_basis(MonomialBasis, x, max_degree)]
    chebyshev_basis = [p for p in maxdegree_basis(ChebyshevBasis, x, max_degree)]
    C = zeros(Float64, length(monomial_basis), length(monomial_basis))

    for (ind, poly) in enumerate(chebyshev_basis)
        C[ind, 1:ind] .= MultivariatePolynomials.coefficients(poly, monomial_basis[1:ind])
    end
    return C
end


function extract_integer_matrix_reps(c0, c, P0, Ps, eq_coeffs, max_degree, x::Vector{<:Variable})
    C = monomial_to_cheb_vandermonde(x, max_degree)
    S0 = calc_SOS_Matrix(P0, value(c0), dual(c0))
    S0 = Hermitian(C[1:size(S0,1), 1:size(S0,1)]'*S0*C[1:size(S0,1), 1:size(S0,1)],:L)
    eig0 = eigmin(S0)
    S_mats =calc_SOS_Matrix.(Ps, value.(c), dual.(c))
    S_mats = [Hermitian(C[1:size(s,1), 1:size(s,1)]'*s*C[1:size(s,1), 1:size(s,1)],:L) for s in S_mats]
    F0 = cholesky(S0 - eig0/2*I)
    Fs = cholesky.(S_mats)
    Ls = [F0.L, [f.L for f in Fs]...]
    eq_coeffs = [C[1:length(ec), 1:length(ec)]'*ec for ec in eq_coeffs]
    max_val = maximum(norm.([Ls...,eq_coeffs...,eig0], Inf))
    scaling_factor = 1e15/max_val
    intLs = [round.(BigInt,scaling_factor*l) for l in Ls]
    int_eq_coeffs = [round.(BigInt,scaling_factor^2*ec) for ec in eq_coeffs]
    D0 = round.(BigInt,eig0/2*scaling_factor^2)
    return (intLs, (D0*I)(size(Ls[1],1)), int_eq_coeffs)
end
function extract_integer_matrix_reps_ineq(c0, c, P0, Ps, max_degree, x::Vector{<:Variable})
    C = monomial_to_cheb_vandermonde(x, max_degree)
    S0 = calc_SOS_Matrix(P0, value(c0), dual(c0))
    S0 = Hermitian(C[1:size(S0,1), 1:size(S0,1)]'*S0*C[1:size(S0,1), 1:size(S0,1)],:L)
    eig0 = eigmin(S0)
    S_mats =calc_SOS_Matrix.(Ps, value.(c), dual.(c))
    S_mats = [Hermitian(C[1:size(s,1), 1:size(s,1)]'*s*C[1:size(s,1), 1:size(s,1)],:L) for s in S_mats]
    F0 = cholesky(S0 - eig0/2*I)
    Fs = cholesky.(S_mats)
    Ls = [F0.L, [f.L for f in Fs]...]
    max_val = maximum(norm.([Ls...,eig0], Inf))
    scaling_factor = 1e15/max_val
    intLs = [round.(BigInt,scaling_factor*l) for l in Ls]
    D0 = round.(BigInt,eig0/2*scaling_factor^2)
    return (intLs, (D0*I)(size(Ls[1],1)))
end

function calc_integer_polynomial_residual(Ls, D0, eq_coeffs,max_half_degree::Integer,  x::Vector{<:Variable}, G::Vector{<:AbstractPolynomial}, H::Vector{<:AbstractPolynomial})
    d = 2*max_half_degree

    S0 = Ls[1]*Ls[1]'
    Ss = [l*l' for l in Ls[2:end]]
    mono_basis = [polynomial(p, BigInt) for p in maxdegree_basis(MonomialBasis,x,2*max_half_degree)]


    ineq_polys = [g*((mono_basis[1:size(Ss[ind],1)])'*Ss[ind]*mono_basis[1:size(Ss[ind],1)]) for (ind, g) in enumerate(G)]
    eq_polys = [h * dot(eq_coeffs[ind], mono_basis[1:length(eq_coeffs[ind])]) for (ind, h) in enumerate(H)]
    sos_poly = (mono_basis[1:size(S0,1)])'*(S0+D0)*mono_basis[1:size(S0,1)]
    sum_poly = (sos_poly + sum(eq_polys) + sum(ineq_polys))
    constant_term = MultivariatePolynomials.coefficient(sum_poly, constant_monomial(sum_poly))
    residual = constant_term - sum_poly
    residual_mat = create_canonical_gram(residual, x, max_half_degree)
    return (residual, residual_mat, constant_term)
end

function calc_certificate_residual(cert::pos_certificate)
    if isnothing(cert.eq_polynomials)
        return calc_integer_polynomial_residual_ineq(adjoint.(cert.SOS_U_factors), cert.diagonally_dominant_gram, cert.max_half_degree, cert.x, cert.ineq_polynomials)
    else
        return calc_integer_polynomial_residual(adjoint.(cert.SOS_U_factors), cert.diagonally_dominant_gram, cert.equality_coefficients, cert.max_half_degree, cert.x, cert.ineq_polynomials, cert.eq_polynomials)
    end
end

function calc_integer_polynomial_residual_ineq(Ls, D0,max_half_degree::Integer,  x::Vector{<:Variable}, G::Vector{<:AbstractPolynomial})
    d = 2*max_half_degree

    S0 = Ls[1]*Ls[1]'
    Ss = [l*l' for l in Ls[2:end]]
    mono_basis = [polynomial(p, BigInt) for p in maxdegree_basis(MonomialBasis,x,2*max_half_degree)]


    ineq_polys = [g*((mono_basis[1:size(Ss[ind],1)])'*Ss[ind]*mono_basis[1:size(Ss[ind],1)]) for (ind, g) in enumerate(G)]
    sos_poly = (mono_basis[1:size(S0,1)])'*(S0+D0)*mono_basis[1:size(S0,1)]
    sum_poly = (sos_poly + sum(ineq_polys))
    constant_term = MultivariatePolynomials.coefficient(sum_poly, constant_monomial(sum_poly))
    residual = constant_term - sum_poly
    residual_mat = create_canonical_gram(residual, x, max_half_degree)
    return (residual, residual_mat, constant_term)
end

function pos_refutation(max_half_degree::Integer, x::Vector{<:Variable}, G::Vector{<:AbstractPolynomial}, H::Vector{<:AbstractPolynomial})
    floatG = [polynomial(g, Float64) for g in G]
    floatH = [polynomial(h, Float64) for h in H]
    (model, P0, Ps, Vs, ineq_vals, eq_vals, c0, c,lc)= create_pos_sos_dual_problem(2*max_half_degree,x, floatG, floatH)
    optimize!(model)
    if !is_solved_and_feasible(model)
        return nothing
    end
    (Ls, D0, eq_coeffs) = extract_integer_matrix_reps(c0, c, P0, Ps, dual.(lc),2*max_half_degree,x)
    int_gs= [polynomial(g, BigInt) for g in G]
    int_hs = [polynomial(h, BigInt) for h in H]
    (residual, residual_mat, constant_term) = calc_integer_polynomial_residual(Ls, D0, eq_coeffs, max_half_degree, x, int_gs, int_hs)
    
    D = Symmetric(rationalize.(BigInt, D0) + residual_mat)
    return pos_certificate(max_half_degree, adjoint.(Ls), D, constant_term, eq_coeffs, x, int_gs, int_hs)
end

function pos_refutation(max_half_degree::Integer, x::Vector{<:Variable}, G::Vector{<:AbstractPolynomial})
    floatG = [polynomial(g, Float64) for g in G]
    (model, P0, Ps, ineq_vals, c0, c) = create_pos_sos_dual_problem_ineq_only(2*max_half_degree,x, floatG)
    optimize!(model)
    if !is_solved_and_feasible(model)
        return nothing
    end
    (Ls, D0) = extract_integer_matrix_reps_ineq(c0, c, P0, Ps, 2*max_half_degree,x)
    int_gs= [polynomial(g, BigInt) for g in G]
    (residual, residual_mat, constant_term) = calc_integer_polynomial_residual_ineq(Ls, D0, max_half_degree, x, int_gs)
    D = Symmetric(rationalize.(BigInt, D0) + residual_mat)
    return pos_certificate(max_half_degree, adjoint.(Ls), D, constant_term, nothing, x, int_gs, nothing)
end


function test_diagonal_dominance(A::AbstractMatrix)
    A = BigFloat.(A)
    return all(sum(abs,A;dims=2) .< 2abs.(diag(A)))
end

#=
@polyvar x[1:3] monomial_order = Graded{Reverse{LexOrder}}
g1 = 1.0 - x[3]^2 - (x[1] + 4)^2 - (x[2])^2 
example_h = [(x[2]^2) + (0.0),]
g2 = 1.0 - x[3]^2 - (x[1] - 4)^2 - (x[2])^2
g3 = (1.0+x[1]^2+x[2]^2+x[3]^2)
example_gs = [g1,g2,g3 ]
certificate = pos_refutation(4, x, example_gs)
calc_certificate_residual(certificate)
=#
#=
(model, P0, Ps, Vs, ineq_vals, eq_vals, c0, c,lc)= create_pos_sos_dual_problem(8,x, example_gs, example_h)
optimize!(model)
r = -1 .- (Vs[1]*dual(lc[1])).*eq_vals[1,:] .-(dual(c0) .+ dual(c[1]).*ineq_vals[1, :] .+ dual(c[2]).*ineq_vals[2, :].+ dual(c[3]).*ineq_vals[3, :]    ) 

(Ls, D0, eq_coeffs) = extract_integer_matrix_reps(c0, c, P0, Ps, dual.(lc),8,x)
int_gs= [polynomial(g, BigInt) for g in example_gs]
int_hs = [polynomial(h, BigInt) for h in example_h]
(residual, residual_mat, constant_term) = calc_integer_polynomial_residual(Ls, D0, eq_coeffs, 4, x, int_gs, int_hs)
D = Symmetric(rationalize.(BigInt, D0) + residual_mat)

mono_basis = [polynomial(p, BigInt) for p in maxdegree_basis(MonomialBasis,x,8)]

typeof(adjoint.(Ls))
=#