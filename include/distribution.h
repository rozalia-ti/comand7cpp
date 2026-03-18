#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <vector>
#include <random>
#include <concepts>
#include "point.h"

template<typename T>
concept RealDistribution = requires(T dist, std::default_random_engine& g) {
    { dist(g) } -> std::convertible_to<double>;
};

template<typename T>
concept DiscreteDistribution = requires(T dist, std::default_random_engine& g) {
    { dist(g) } -> std::convertible_to<uint8_t>;
};

template <typename Engine, RealDistribution PositionDistribution, DiscreteDistribution ClassDistribution>
requires std::uniform_random_bit_generator<std::remove_reference_t<Engine>>
std::vector<Point> generate_random(
    Engine&& engine,
    std::size_t amount,
    PositionDistribution&& position_distribution,
    ClassDistribution&& class_distribution
) {
    std::vector<Point> output;
    output.reserve(amount);

    for (std::size_t i = 0; i < amount; i++) {
        output.push_back(Point {
            position_distribution(engine),
            position_distribution(engine),
            class_distribution(engine)
        });
    }

    return output;
}

template <typename Engine, RealDistribution PositionDistribution, RealDistribution EquationDistribution>
requires std::uniform_random_bit_generator<std::remove_reference_t<Engine>>
std::vector<Point> generate_linear(
    Engine&& engine,
    std::size_t amount,
    PositionDistribution&& position_distribution,
    EquationDistribution&& equation_distribution
) {
    std::vector<Point> output;
    output.reserve(amount);

    // y = kx + b
    double k = equation_distribution(engine);
    double b = equation_distribution(engine);

    std::uniform_int_distribution<uint8_t> class_distribution(0, 1);
    auto choice = class_distribution(engine);

    for (std::size_t i = 0; i < amount; i++) {
        auto x = position_distribution(engine);
        auto y = position_distribution(engine);

        auto fn = k * x + b;

        output.push_back(Point {
            x,
            y,
            fn < y ? (uint8_t)(1 ^ choice) : (uint8_t)(0 ^ choice)
        });
    }

    return output;
}

#endif
