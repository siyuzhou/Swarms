#pragma once

#include <valarray>
#include "agent.hpp"

enum Mode
{
    SUM,
    AVG
};

struct Config
{
    double cohesion = 0.2;
    double separation = 2;
    double alignment = 0.2;
    double obstacle_avoidance = 2;
    double goal_steering = 0.5;
    Mode neighbor_interaction_mode = AVG;
};

template <unsigned int N>
class Boid : public Agent<N>
{
public:
    explicit Boid(const std::valarray<double> &p,
                  const std::valarray<double> &v = std::valarray<double>(N),
                  const std::valarray<double> &a = std::valarray<double>(N),
                  double max_v = -1., double max_a = -1., double size = 0., double vision = -1.);

    void decide() override;

    static void setModel(const Config &config);

private:
    std::valarray<double> cohesion_();
    std::valarray<double> separation_();
    std::valarray<double> alignment_();
    std::valarray<double> obstacleAvoidance_();
    std::valarray<double> goalSeeking_();

    static Config config_;
};

template <unsigned int N>
Config Boid<N>::config_ = Config();

template <unsigned int N>
Boid<N>::Boid(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
              double max_v, double max_a, double size, double vision) : Agent<N>{p, v, a, max_v, max_a, size, vision} {}

template <unsigned int N>
void Boid<N>::setModel(const Config &config)
{
    Boid<N>::config_ = config;
}

template <unsigned int N>
std::valarray<double> Boid<N>::cohesion_()
{
    if (this->neighbors_.empty())
        return std::valarray<double>(N);

    std::valarray<double> center(N);
    for (auto &neighbor : this->neighbors_)
        center += neighbor->position();
    center /= this->neighbors_.size();

    return center - this->position_;
}

template <unsigned int N>
std::valarray<double> Boid<N>::separation_()
{
    std::valarray<double> repel(N);
    for (auto &neighbor : this->neighbors_)
    {
        double d = this->distance(neighbor);
        if (d < this->size_)
        {
            d = std::max(d, 0.01);
            repel += (this->position_ - neighbor->position()) / d / d;
        }
    }
    if (!this->neighbors_.empty() && config_.neighbor_interaction_mode == AVG)
        repel /= this->neighbors_.size();
    return repel;
}

template <unsigned int N>
std::valarray<double> Boid<N>::alignment_()
{
    if (this->neighbors_.empty())
        return std::valarray<double>(N);

    std::valarray<double> avg_velocity(N);
    for (auto &neighbor : this->neighbors_)
        avg_velocity += neighbor->velocity();
    avg_velocity /= this->neighbors_.size();

    return avg_velocity - this->velocity_;
}

template <unsigned int N>
std::valarray<double> Boid<N>::obstacleAvoidance_()
{
    // No obstacle.
    if (this->obstacles_.empty())
        return std::valarray<double>(N);

    // Assume there is always enough space between obstacles,
    // find the nearest obstacle in the front.
    double min_d = std::numeric_limits<double>::max();

    std::shared_ptr<const Obstacle<N>> closest = nullptr;
    for (auto &obs : this->obstacles_)
    {
        double d = this->distance(obs);
        if (d < min_d && Entity<N>::dot(this->direction(obs), this->velocity_) > 0)
        {
            closest = obs;
            min_d = d;
        }
    }
    if (closest == nullptr)
        return std::valarray<double>(N);

    std::shared_ptr<const Obstacle<N>> obs = closest;

    auto obs_direction = this->direction(obs);

    std::valarray<double> v_direction(N);
    double speed = this->speed();
    if (speed > 0)
        v_direction = this->velocity_ / speed;

    double cos_theta = Entity<N>::dot(obs_direction, v_direction);
    double sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    double normal_d = (min_d + obs->size()) * sin_theta - obs->size();

    // Decide if on course of collision.
    if (normal_d < this->size_)
    {
        std::valarray<double> turn_direction = v_direction * cos_theta - obs_direction;
        turn_direction /= Entity<N>::norm(turn_direction);

        double overlap = this->size_ - normal_d;
        return turn_direction * overlap * overlap / std::max(min_d, this->size_);
    }

    // Return 0 if obstacle does not obstruct.
    return std::valarray<double>(N);
}

template <unsigned int N>
std::valarray<double> Boid<N>::goalSeeking_()
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
    std::valarray<double> offset = this->goal_->position() - this->position_;
    double d = Entity<N>::norm(offset);
    double target_speed = this->max_speed_ * std::min(1., d / 20.);
    std::valarray<double> target_velocity = target_speed * offset / d;

    return target_velocity - this->velocity_;
}

template <unsigned int N>
void Boid<N>::decide()
{
    this->setAcceleration(config_.cohesion * this->cohesion_() +
                          config_.separation * this->separation_() +
                          config_.alignment * this->alignment_() +
                          config_.obstacle_avoidance * this->obstacleAvoidance_() +
                          config_.goal_steering * this->goalSeeking_());
}