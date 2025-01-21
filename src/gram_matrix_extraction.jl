#Note, Polynomial basis from Hypatia is in Graded{Reverse{LexOrder}}
using DynamicPolynomials
import MultivariatePolynomials: coefficient
using MultivariateBases
function homog_multinomial(exponents::Vector{<:Integer}, d::Integer)
    numerator = factorial(d)
    denominator = prod(factorial.(exponents))*factorial(d - sum(exponents))
    return numerator//denominator
end


function create_canonical_gram(p::AbstractPolynomialLike, x::Vector{<:Variable}, max_half_degree)
    b = maxdegree_basis(MonomialBasis, x, max_half_degree)
    bvec = [poly for poly in b]
    monomial_matrix = bvec*bvec'
    multinomial_vec = homog_multinomial.(DynamicPolynomials.exponents.(bvec),max_half_degree)
    scaling_matrix = (multinomial_vec*multinomial_vec').//( homog_multinomial.(DynamicPolynomials.exponents.(monomial_matrix),2*max_half_degree))
    gram_matrix = zeros(Rational{BigInt}, size(monomial_matrix))
    for (index, mono) in enumerate(monomial_matrix)
        gram_matrix[index] = MultivariatePolynomials.coefficient(p, mono, x)*scaling_matrix[index]
    end
    return gram_matrix
end
