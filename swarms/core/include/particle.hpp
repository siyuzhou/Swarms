#pragma once

#include <valarray>
#include <memory>
#include "entity.hpp"

template <unsigned int N>
class Particle : public Entity<N>
{
public:
    Particle(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
             double max_v = -1., double max_a = -1., double size = 0.);

    double distance(std::shared_ptr<const Entity<N>> other) const override;
    double distance(const std::valarray<double> &x) const override;
    std::valarray<double> direction(std::shared_ptr<const Entity<N>> other) const override;
    std::valarray<double> direction(const std::valarray<double> &x) const override;
};

// Template implementation

template <unsigned int N>
Particle<N>::Particle(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                      double max_v, double max_a, double size)
    : Entity<N>(p, v, a, max_v, max_a, size) {}

template <unsigned int N>
double Particle<N>::distance(std::shared_ptr<const Entity<N>> other) const
{
    return other->distance(this->position_);
}

template <unsigned int N>
double Particle<N>::distance(const std::valarray<double> &x) const
{
    auto diff = x - this->position_;
    return Entity<N>::norm(diff);
}

template <unsigned int N>
std::valarray<double> Particle<N>::direction(std::shared_ptr<const Entity<N>> other) const
{
    return -other->direction(this->position_);
}

template <unsigned int N>
std::valarray<double> Particle<N>::direction(const std::valarray<double> &x) const
{
    auto diff = x - this->position_;
    return diff / Entity<N>::norm(diff);
}