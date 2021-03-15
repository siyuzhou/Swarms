#pragma once

#include <valarray>
#include "particle.hpp"

template <unsigned int N>
class Goal : public Particle<N>
{
public:
    Goal(const std::valarray<double> &p,
         const std::valarray<double> &v = std::valarray<double>(N),
         const std::valarray<double> &a = std::valarray<double>(N),
         double max_v = -1., double max_a = -1.) : Particle<N>{p, v, a, max_v, max_a} {}
};