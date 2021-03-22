#pragma once

#include <valarray>
#include <vector>
#include <memory>
#include "particle.hpp"

template <unsigned int N>
class Chaser : public Particle<N>
{
public:
    explicit Chaser(const std::valarray<double> &p,
                    const std::valarray<double> &v = std::valarray<double>(N),
                    const std::valarray<double> &a = std::valarray<double>(N),
                    double max_v = -1., double max_a = -1.);
    void addTarget(std::shared_ptr<const Chaser<N>> other);
    void decide();

private:
    std::vector<std::shared_ptr<const Chaser<N>>> targets_;
};

// Template implementation

template <unsigned int N>
Chaser<N>::Chaser(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                  double max_v, double max_a)
    : Particle<N>(p, v, a, max_v, max_a) {}

template <unsigned int N>
void Chaser<N>::addTarget(std::shared_ptr<const Chaser<N>> other)
{
    targets_.push_back(other);
}

template <unsigned int N>
void Chaser<N>::decide()
{
    std::valarray<double> disp(N);
    for (auto &target : targets_)
        disp += target->position() - this->position_;

    this->setAcceleration(disp);
}