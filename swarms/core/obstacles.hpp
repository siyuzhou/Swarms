#pragma once

#include <valarray>
#include <memory>
#include "entity.hpp"

template <unsigned int N>
class Obstacle : public Entity<N>
{
public:
    explicit Obstacle(const std::valarray<double> &p, const std::valarray<double> &v = std::valarray<double>(N), double size = 0);
    double size() const;

protected:
    const double size_;
};

template <unsigned int N>
class Wall : public Obstacle<N>
{
public:
    Wall(const std::valarray<double> &o, const std::valarray<double> &p);

    double distance(std::shared_ptr<const Entity<N>> other) const override;
    double distance(const std::valarray<double> &x) const override;
    std::valarray<double> direction(std::shared_ptr<const Entity<N>> other) const override;
    std::valarray<double> direction(const std::valarray<double> &x) const override;

private:
    void setDirection_(const std::valarray<double> &o);
    std::valarray<double> direction_;
};

template <unsigned int N>
class Sphere : public Obstacle<N>
{
public:
    explicit Sphere(double size, const std::valarray<double> &p, const std::valarray<double> &v = std::valarray<double>(N));

    double distance(const std::valarray<double> &x) const override;
    double distance(std::shared_ptr<const Entity<N>> other) const override;
    std::valarray<double> direction(std::shared_ptr<const Entity<N>> other) const override;
    std::valarray<double> direction(const std::valarray<double> &x) const override;
};

// Template implementations

template <unsigned int N>
Obstacle<N>::Obstacle(const std::valarray<double> &p, const std::valarray<double> &v, double size)
    : Entity<N>{p, v, std::valarray<double>(N)}, size_(size) {}

template <unsigned int N>
double Obstacle<N>::size() const { return size_; }

template <unsigned int N>
Wall<N>::Wall(const std::valarray<double> &o, const std::valarray<double> &p)
    : Obstacle<N>{p}
{
    setDirection_(o);
}

template <unsigned int N>
double Wall<N>::distance(std::shared_ptr<const Entity<N>> other) const
{
    return distance(other->position());
}

template <unsigned int N>
double Wall<N>::distance(const std::valarray<double> &x) const
{
    std::valarray<double> dis = x - this->position_;
    return Entity<N>::dot(dis, direction_);
}

template <unsigned int N>
std::valarray<double> Wall<N>::direction(std::shared_ptr<const Entity<N>> other) const
{
    return direction(other->position());
}

template <unsigned int N>
std::valarray<double> Wall<N>::direction(const std::valarray<double> &x) const
{
    return direction_;
}

template <unsigned int N>
void Wall<N>::setDirection_(const std::valarray<double> &o)
{
    if (o.size() != this->ndim())
        throw "valarray size not matched.";
    direction_ = o / Entity<N>::norm(o);
}

template <unsigned int N>
Sphere<N>::Sphere(double size, const std::valarray<double> &p, const std::valarray<double> &v)
    : Obstacle<N>{p, v, size} {}

template <unsigned int N>
double Sphere<N>::distance(const std::valarray<double> &x) const
{
    auto diff = x - this->position_;
    double d = Entity<N>::norm(diff) - this->size();
    return d;
}

template <unsigned int N>
double Sphere<N>::distance(std::shared_ptr<const Entity<N>> other) const
{
    return distance(other->position());
}

template <unsigned int N>
std::valarray<double> Sphere<N>::direction(std::shared_ptr<const Entity<N>> other) const
{
    return direction(other->position());
}

template <unsigned int N>
std::valarray<double> Sphere<N>::direction(const std::valarray<double> &x) const
{
    auto diff = x - this->position_;
    return diff / Entity<N>::norm(diff);
}