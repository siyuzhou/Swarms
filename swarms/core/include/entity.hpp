#pragma once

#include <valarray>
#include <numeric>
#include <memory>
#include "entity.hpp"

template <unsigned int N>
class Entity
{
public:
    Entity(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
           double max_v = -1, double max_a = -1, double size = 0.);

    void reset(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a);

    unsigned int ndim() const;
    double size() const;
    std::valarray<double> position() const;
    std::valarray<double> velocity() const;
    double speed() const;
    std::valarray<double> acceleration() const;
    double maxVelocity() const;
    double maxAcceleration() const;

    void setPosition(const std::valarray<double> &p);
    void setVelocity(const std::valarray<double> &v);
    void setAcceleration(const std::valarray<double> &a);
    void setMaxVelocity(double max_v);
    void setMaxAcceleration(double max_a);

    virtual double distance(std::shared_ptr<const Entity<N>> other) const = 0;
    virtual std::valarray<double> direction(std::shared_ptr<const Entity<N>> other) const = 0;
    virtual double distance(const std::valarray<double> &x) const = 0;
    virtual std::valarray<double> direction(const std::valarray<double> &x) const = 0;

    virtual void move(double dt);

    static double norm(const std::valarray<double> &x);
    static double dot(const std::valarray<double> &x, const std::valarray<double> &y);

protected:
    void regularizeV_();
    void regularizeA_();

    double max_speed_ = -1.;
    double max_acceleration_ = -1.;

    const double size_;

    std::valarray<double> position_;
    std::valarray<double> velocity_;
    std::valarray<double> acceleration_;
};

// Template implementation

template <unsigned int N>
Entity<N>::Entity(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a,
                  double max_v, double max_a, double size)
    : max_speed_(max_v), max_acceleration_(max_a), size_(size)
{
    reset(p, v, a);
}

template <unsigned int N>
void Entity<N>::reset(const std::valarray<double> &p, const std::valarray<double> &v, const std::valarray<double> &a)
{
    setPosition(p);
    setVelocity(v);
    setAcceleration(a);
}

template <unsigned int N>
void Entity<N>::setPosition(const std::valarray<double> &p)
{
    if (p.size() != N)
        throw "p size not matched.";
    position_ = p;
}

template <unsigned int N>
void Entity<N>::setVelocity(const std::valarray<double> &v)
{
    if (v.size() != N)
        throw "v size not matched.";
    velocity_ = v;
    regularizeV_();
}

template <unsigned int N>
void Entity<N>::setAcceleration(const std::valarray<double> &a)
{
    if (a.size() != N)
        throw "a size not matched.";

    acceleration_ = a;
    regularizeA_();
}

template <unsigned int N>
void Entity<N>::setMaxVelocity(double max_v) { max_speed_ = max_v; }

template <unsigned int N>
void Entity<N>::setMaxAcceleration(double max_a) { max_acceleration_ = max_a; }

template <unsigned int N>
unsigned int Entity<N>::ndim() const { return N; }

template <unsigned int N>
double Entity<N>::size() const { return size_; }

template <unsigned int N>
std::valarray<double> Entity<N>::position() const
{
    return position_;
}

template <unsigned int N>
std::valarray<double> Entity<N>::velocity() const { return velocity_; }

template <unsigned int N>
std::valarray<double> Entity<N>::acceleration() const { return acceleration_; }

template <unsigned int N>
double Entity<N>::speed() const { return Entity<N>::norm(velocity_); }

template <unsigned int N>
double Entity<N>::maxVelocity() const { return max_speed_; }

template <unsigned int N>
double Entity<N>::maxAcceleration() const { return max_acceleration_; }

template <unsigned int N>
double Entity<N>::norm(const std::valarray<double> &x)
{
    return Entity<N>::dot(x, x);
}

template <unsigned int N>
double Entity<N>::dot(const std::valarray<double> &x, const std::valarray<double> &y)
{
    return std::sqrt(std::inner_product(std::begin(x), std::end(x), std::begin(y), 0.));
}

template <unsigned int N>
void Entity<N>::regularizeV_()
{
    if (max_speed_ > 0)
    {
        double v = speed();
        if (v > max_speed_)
            velocity_ *= max_speed_ / v;
    }
}

template <unsigned int N>
void Entity<N>::regularizeA_()
{
    if (max_acceleration_ > 0)
    {
        double a = Entity<N>::norm(acceleration_);
        if (a > max_acceleration_)
            acceleration_ *= max_acceleration_ / a;
    }
}

template <unsigned int N>
void Entity<N>::move(double dt)
{
    velocity_ += acceleration_ * dt;
    regularizeV_();
    position_ += velocity_ * dt;
}