#pragma once

#include <valarray>
#include <memory>
#include "agent.hpp"

struct Config
{
    double tau = 1.0;
    double A = 1.0;
    double B = 2.0;
    double k = 2.0;
    double kappa = 1.0;
}

template <unsigned int N>
class Vicsek : public Agent<N>
{
public:
    explicit Vicsek(const std::valarray<double> &p,
                    const std::valarray<double> &v = std::valarray<double>(N),
                    const std::valarray<double> &a = std::valarray<double>(N),
                    double max_v = -1., double max_a = -1., double size = 0., double vision = -1.);

    void decide() override;

    static void setModel(const Config &config);

private:
    std::valarray<double> interaction_(std::shared_ptr<Entity<N>> other);
    std::valarray<double> goalSeaking_();

    static Config config_;
};

template <unsigned int N>
Config Vicsek<N>::config_ = Config();

template <unsigned int N>
Vicsek<N>::Vicsek(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                  double max_v, double max_a, double size, double vision) : Agent<N>{p, v, a, max_v, max_a, size, vision} {}

template <unsigned int N>
void Vicsek<N>::setModel(const Config &config)
{
    Vicsek<N>::config_ = config;
}

template <unsigned int N>
std::valarray<double> Vicsek<N>::interaction_(std::shared_ptr<const Entity<N>> other)
{
    double r = size_ + other->size();
    double d = this->distance(other);

    auto n = this->direction(other);
    auto repulsion = config_.A * std::exp((r - d) / config_.B) * n;

    std::valarray<double> friction(N);
    if (r > d)
    {
        repulsion += config_.k * (r - d) * n;

        auto delta_v = other.velocity() - this->velocity_;
        friction += config_.kappa * (r - d) * (delta_v - Entity<N>::dot(delta_v, n) * n);
    }
    return friction + repulsion;
}

template <unsigned int N>
std::valarray<double> Vicsek<N>::goalSeeking_()
{
    // If no explicit goal is preent, accelerate along velocity.
    if (this->goal_ == nullptr)
    {
        double speed = this->speed();
        if (speed > 0)
            return this->velocity_ / speed;
        else
            return std::valarray<double>(N);
    }

    // The farther the goal, the stronger the attraction.
    auto offset = goal_.position() - this->position_;
    double d = Entity<N>::norm(offset);
    auto target_speed = max_speed_ * std::min(1, d / 20);
    auto target_velocity = target_speed * offset / d;
    return target_velocity - this->velocity_;
}

template <unsigned int N>
void Vicsek<N>::decide()
{
    std::valarray<double> inter(N);
    for (const auto entity : this->obstacles_)
        inter += interaction_(entity);
    for (const auto entity : this->neighbors_)
        inter += interaction_(entity);

    this->setAcceleration(inter + goalSteering_());
}