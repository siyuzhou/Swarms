#pragma once

#include <valarray>
#include <vector>
#include <memory>
#include "entity.hpp"
#include "particle.hpp"
#include "goal.hpp"
#include "obstacles.hpp"
#include "environment.hpp"

template <unsigned int N>
class Environment;

template <unsigned int N>
class Agent : public Particle<N>
{
public:
    explicit Agent(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                   double max_v = -1., double max_a = -1., double size = 0., double vision = -1.);

    void setGoal(std::shared_ptr<const Goal<N>> goal);
    bool canSee(std::shared_ptr<const Entity<N>> other) const;
    void observe(const Environment<N> &env);

    double size() const;

    virtual void decide() = 0;

protected:
    std::vector<std::shared_ptr<const Agent<N>>> neighbors_;
    std::vector<std::shared_ptr<const Obstacle<N>>> obstacles_;

    std::shared_ptr<const Goal<N>> goal_;

    const double size_;
    const double vision_;
};

// Template implementation

template <unsigned int N>
Agent<N>::Agent(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                double max_v, double max_a, double size, double vision)
    : Particle<N>{p, v, a, max_v, max_a}, size_(size), vision_((vision > 0 ? vision : std::numeric_limits<double>::max()))
{
    this->setMaxVelocity(max_v);
    this->setMaxAcceleration(max_a);
}

template <unsigned int N>
double Agent<N>::size() const { return size_; }

template <unsigned int N>
void Agent<N>::setGoal(std::shared_ptr<const Goal<N>> goal)
{
    goal_ = goal;
}

template <unsigned int N>
bool Agent<N>::canSee(std::shared_ptr<const Entity<N>> other) const
{
    return (this->distance(other) < vision_);
}

template <unsigned int N>
void Agent<N>::observe(const Environment<N> &env)
{
    neighbors_.clear();
    for (auto &other : env.population())
    {
        if ((this != other.get()) && canSee(other))
            neighbors_.push_back(other);
    }
    obstacles_.clear();
    for (auto &obstacle : env.obstacles())
        obstacles_.push_back(obstacle);
}