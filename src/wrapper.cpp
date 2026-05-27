#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <iostream>

#include "ai.h"
#include "distribution.h"
#include "point.h"

namespace py = pybind11;

class PyModel {
private:
    AiMatrix ai;
    double current_k = 0.0;
    double current_b = 0.0;
    bool class_1_is_above = true; // Флаг для запоминания стороны

    std::vector<double> format_input(const std::vector<Point>& points) {
        std::vector<double> ai_input(128 * 4, 0.0);
        size_t limit = std::min(points.size(), (size_t)128);
        for (size_t i = 0; i < limit; i++) {
            ai_input[i * 4 + 0] = points[i].x / 100.0;
            ai_input[i * 4 + 1] = points[i].y / 100.0;
            ai_input[i * 4 + 2] = (double)points[i].cls;
            ai_input[i * 4 + 3] = 1.0;
        }
        return ai_input;
    }

public:
    PyModel() {
        auto linear = [](double x) { return x; };
        auto linear_df = [](double) { return 1.0; };
        auto relu = [](double x) { return x > 0 ? x : x * 0.01; };
        auto relu_df = [](double x) { return x > 0 ? 1.0 : 0.01; };

        ai.configure({
            { 128 * 4, 0, relu, relu_df },
            { 64 * 4, 1, relu, relu_df },
            { 32 * 4, 1, relu, relu_df },
            { 2, 1, linear, linear_df }
        });
        ai.randomize();
    }

    // Первый аргумент - количество батчей, второй - количество проходов (эпох) по батчу
    void train_arbitraty(int batches, int epochs) {
        std::ranlux48 engine(std::random_device{}());
        // Расширили диапазон, чтобы сеть понимала отрицательные числа
        std::uniform_real_distribution<double> position_distribution(-100.0, 100.0);
        std::uniform_real_distribution<double> equation_distribution(-3.0, 3.0);

        for (int i = 0; i < batches; i++) {
            // Генерируем 100 точек, как в твоем оригинальном main.cpp
            auto points_result = generate_linear(engine, 100, position_distribution, equation_distribution);
            std::vector<double> ai_input = format_input(points_result.points);
            std::vector<double> ai_target = {points_result.k, points_result.b};

            for (int e = 0; e < epochs; e++) {
                ai.train(ai_input, ai_target, 0.00001);
            }
        }
    }

    std::pair<double, double> predict_line(const std::vector<std::vector<double>>& raw_points) {
        std::vector<Point> points;
        for (const auto& p : raw_points) {
            points.push_back({p[0], p[1], (uint8_t)p[2]});
        }
        
        std::vector<double> ai_input = format_input(points);
        auto output = ai.process(ai_input);
        
        current_k = output[0];
        current_b = output[1];
        
        // Магия: динамически определяем, с какой стороны прямой лежит класс 1
        int class_1_above_count = 0;
        int class_1_below_count = 0;
        
        for (const auto& p : points) {
            if (p.y > current_k * p.x + current_b) {
                if (p.cls == 1) class_1_above_count++;
            } else {
                if (p.cls == 1) class_1_below_count++;
            }
        }
        
        class_1_is_above = (class_1_above_count > class_1_below_count);
        
        return {current_k, current_b};
    }

    int predict_point_class(double x, double y) {
        bool is_above = (y > current_k * x + current_b);
        // Если точка сверху и класс 1 сверху, то это 1. Иначе 0.
        return (is_above == class_1_is_above) ? 1 : 0;
    }

    void save(const std::string& filename) { ai.save(filename); }
    void load(const std::string& filename) { ai.load(filename); }
    void train(const std::string& filename) { (void)filename; }
};

PYBIND11_MODULE(comand7, m) {
    py::class_<PyModel>(m, "Model")
        .def(py::init<>())
        .def("train_arbitraty", &PyModel::train_arbitraty)
        .def("predict_line", &PyModel::predict_line)
        .def("predict_point_class", &PyModel::predict_point_class)
        .def("save", &PyModel::save)
        .def("load", &PyModel::load)
        .def("train", &PyModel::train);
}