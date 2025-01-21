#Steps:
#1. Create Tensor product of chebyshev first-kind grid over [-1, 1]^n
#2. Create Vandermonde matrix for degree-d type-1 cheb polynomials
#3. Compute QR and return Q 

#Note, Polynomial basis from Hypatia is in Graded{Reverse{LexOrder}}



using LinearAlgebra
import DynamicPolynomials: @polyvar
using Random
using Hypatia
using Random
using MultivariateBases
using DynamicPolynomials
using StaticPolynomials


function sel_random_chebyshev_points(T::Type{<:Real}, n::Integer, d::Integer, sample_size::Integer, tolerance::Real = 1e-4)
    possible_indices = CartesianIndices(Tuple([2d+j for j = 1:n]))
    selection = randperm(length(possible_indices))[1:sample_size]
    cand_pts = zeros(T, length(selection), n)
    selected = Tuple.(possible_indices[selection])
    selected_indices = vcat(selected...)
    index_matrix = reinterpret(reshape, Int, selected_indices)'
    @views for col_index in axes(cand_pts,2)
        cand_pts[:, col_index] = @. -cospi((index_matrix[:, col_index]-1)/(2d+col_index-1))
    end
    return cand_pts
end


function approx_fekete_data_3D(T::Type{<:Real}, n_vars::Integer, max_half_degree::Integer)
    d = max_half_degree
    U = Hypatia.PolyUtils.get_U(n_vars, d)
    L = Hypatia.PolyUtils.get_L(n_vars, d)
    sample_size = min(4*U, prod([2d+j for j = 1:n_vars]))
    sampled_points = sel_random_chebyshev_points(T, n_vars, max_half_degree, sample_size)
    V = Hypatia.PolyUtils.make_chebyshev_vandermonde(sampled_points, max_half_degree*2)
    if U <= 1500
        Vc = copy(V')
        F = qr!(Vc, ColumnNorm())
    else
        Vc = copy(V)
        F = lu!(Vc, RowMaximum())
    end
    keep_pts = F.p[1:U]
    V = V[keep_pts, :]
    final_points = sampled_points[keep_pts, :]
    return (U, final_points,V)
end

function fast_SOS_data(T::Type{<:Real}, n_vars::Integer, max_half_degree::Integer)
    d = max_half_degree
    if n_vars == 1
        (U, pts, _, _, V, _) = Hypatia.PolyUtils.cheb2_data(T, d, true, false)
    elseif n_vars == 2
        (U, pts, _, _, V, _) = Hypatia.PolyUtils.padua_data(T, d, true, false)
    elseif n_vars > 2
        (U, pts, V) =  approx_fekete_data_3D(T, n_vars, d)
    end
    P0 = V[:, 1:Hypatia.PolyUtils.get_L(n_vars, d)]
    return (U, pts, [P0,], V)
end