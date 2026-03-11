#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <vector>
#include <random>
#include "point.h"

template <typename Engine, typename PositionDistribution, typename ClassDistribution>
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

template <typename Engine, typename PositionDistribution, typename EquationDistribution>
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
