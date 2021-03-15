#pragma once

#include <valarray>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>
#include "agent.hpp"
#include "obstacles.hpp"
#include "goal.hpp"

template <unsigned int N>
class Agent;

template <unsigned int N>
class Environment
{
public:
    Environment(const std::vector<double> &lbs, const std::vector<double> &ubs);

    unsigned int ndim() const;
    const std::vector<std::shared_ptr<const Obstacle<N>>> &obstacles() const;
    const std::vector<std::shared_ptr<const Agent<N>>> &population() const;

    void addAgent(std::shared_ptr<Agent<N>> agent);
    void addGoal(std::shared_ptr<Goal<N>> goal);
    void addObstacle(std::shared_ptr<Obstacle<N>> obstacle);
    void update(double dt);

private:
    void moveAgents_(double dt);
    void moveGoals_(double dt);
    void moveObstacles_(double dt);

    std::vector<std::shared_ptr<const Agent<N>>> population_;
    std::vector<std::shared_ptr<const Obstacle<N>>> obstacles_;

    std::vector<std::shared_ptr<Agent<N>>> agents_;
    std::vector<std::shared_ptr<Goal<N>>> goals_;
    std::vector<std::shared_ptr<Obstacle<N>>> free_obstacles_;
    std::vector<std::shared_ptr<Wall<N>>> boundaries_;
};

// Implementation

template <unsigned int N>
Environment<N>::Environment(const std::vector<double> &lbs, const std::vector<double> &ubs)
{
    if (lbs.size() != N || ubs.size() != N)
        throw std::invalid_argument("lbs or ubs not matching N");

    for (int d = 0; d < N; d++)
    {
        std::valarray<double> o1(N), o2(N), p1(N), p2(N);
        o1[d] = 1;
        o2[d] = -1;

        if (lbs[d] > ubs[d])
            throw std::invalid_argument("lbs[d] is greater than ubs[d]");

        p1[d] = lbs[d];
        p2[d] = ubs[d];

        std::shared_ptr<Wall<N>> wall1(new Wall<N>(o1, p1)), wall2(new Wall<N>(o2, p2));
        boundaries_.push_back(wall1);
        boundaries_.push_back(wall2);
        obstacles_.push_back(std::const_pointer_cast<const Wall<N>>(wall1));
        obstacles_.push_back(std::const_pointer_cast<const Wall<N>>(wall2));
    }
}

template <unsigned int N>
unsigned int Environment<N>::ndim() const { return N; }

template <unsigned int N>
const std::vector<std::shared_ptr<const Obstacle<N>>> &Environment<N>::obstacles() const
{
    return obstacles_;
}

template <unsigned int N>
const std::vector<std::shared_ptr<const Agent<N>>> &Environment<N>::population() const
{
    return population_;
}

template <unsigned int N>
void Environment<N>::addAgent(std::shared_ptr<Agent<N>> agent)
{
    agents_.push_back(agent);
    population_.push_back(std::const_pointer_cast<const Agent<N>>(agent));
}

template <unsigned int N>
void Environment<N>::addGoal(std::shared_ptr<Goal<N>> goal)
{
    goals_.push_back(move(goal));
}

template <unsigned int N>
void Environment<N>::addObstacle(std::shared_ptr<Obstacle<N>> obstacle)
{
    free_obstacles_.push_back(obstacle);
    obstacles_.push_back(std::const_pointer_cast<const Obstacle<N>>(obstacle));
}

template <unsigned int N>
void Environment<N>::moveAgents_(double dt)
{
    for (auto &agent : agents_)
    {
        agent->observe(*this);
        agent->decide();
    }

    for (auto &agent : agents_)
        agent->move(dt);
}

template <unsigned int N>
void Environment<N>::moveGoals_(double dt)
{
    for (auto &goal : goals_)
        goal->move(dt);
}

template <unsigned int N>
void Environment<N>::moveObstacles_(double dt)
{
    for (auto &obstacle : free_obstacles_)
        obstacle->move(dt);
}

template <unsigned int N>
void Environment<N>::update(double dt)
{
    moveAgents_(dt);
    moveGoals_(dt);
    moveObstacles_(dt);
}