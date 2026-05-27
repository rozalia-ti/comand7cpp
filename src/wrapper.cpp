#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>

#include "ai.h"
#include "distribution.h"
#include "point.h"

namespace py = pybind11;

class PyModel {
private:
    AiMatrix ai;
    bool line_mode = true;
    size_t input_size = 512;
    double current_k = 0.0;
    double current_b = 0.0;
    bool class_1_is_above = true; // Флаг для запоминания стороны

    void configure_line_mode() {
        line_mode = true;
        input_size = 128 * 4;

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

    void configure_classification_mode(size_t in_size, size_t hidden1, size_t hidden2) {
        if (in_size == 0) {
            throw std::runtime_error("input_size must be > 0");
        }

        line_mode = false;
        input_size = in_size;

        auto relu = [](double x) { return x > 0 ? x : x * 0.01; };
        auto relu_df = [](double x) { return x > 0 ? 1.0 : 0.01; };
        auto linear = [](double x) { return x; };
        auto linear_df = [](double) { return 1.0; };
        auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        auto sigmoid_df = [](double y) { return y * (1.0 - y); };

        ai.configure({
            { in_size, 0, linear, linear_df },
            { hidden1, 1, relu, relu_df },
            { hidden2, 1, relu, relu_df },
            { 1, 1, sigmoid, sigmoid_df }
        });
        ai.randomize();
    }

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
        configure_line_mode();
    }

    PyModel(size_t in_size, size_t hidden1 = 32, size_t hidden2 = 16) {
        configure_classification_mode(in_size, hidden1, hidden2);
    }

    // Первый аргумент - количество батчей, второй - количество проходов (эпох) по батчу
    void train_arbitraty(int batches, int epochs) {
        if (!line_mode) {
            throw std::runtime_error("train_arbitraty is available only in line mode");
        }

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
        if (!line_mode) {
            throw std::runtime_error("predict_line is available only in line mode");
        }

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
        if (!line_mode) {
            throw std::runtime_error("predict_point_class is available only in line mode");
        }

        bool is_above = (y > current_k * x + current_b);
        // Если точка сверху и класс 1 сверху, то это 1. Иначе 0.
        return (is_above == class_1_is_above) ? 1 : 0;
    }

    void fit(const std::vector<std::vector<double>>& x_batch, const std::vector<double>& y_batch, double lr) {
        if (line_mode) {
            throw std::runtime_error("fit is available only in classification mode");
        }

        if (x_batch.size() != y_batch.size()) {
            throw std::runtime_error("x_batch and y_batch have different lengths");
        }

        for (size_t i = 0; i < x_batch.size(); i++) {
            const auto& sample = x_batch[i];
            if (sample.size() != input_size) {
                throw std::runtime_error("feature vector has wrong size");
            }

            std::vector<double> target{ y_batch[i] };
            std::vector<double> train_input = sample;
            ai.train(train_input, target, lr);
        }
    }

    double predict_proba(const std::vector<double>& sample) {
        if (line_mode) {
            throw std::runtime_error("predict_proba is available only in classification mode");
        }

        if (sample.size() != input_size) {
            throw std::runtime_error("feature vector has wrong size");
        }

        std::vector<double> input = sample;
        return ai.process(input)[0];
    }

    int predict(const std::vector<double>& sample) {
        return predict_proba(sample) >= 0.5 ? 1 : 0;
    }

    std::vector<double> predict_proba_batch(const std::vector<std::vector<double>>& x_batch) {
        std::vector<double> out;
        out.reserve(x_batch.size());
        for (const auto& sample : x_batch) {
            out.push_back(predict_proba(sample));
        }
        return out;
    }

    std::vector<int> predict_batch(const std::vector<std::vector<double>>& x_batch) {
        std::vector<int> out;
        out.reserve(x_batch.size());
        for (const auto& sample : x_batch) {
            out.push_back(predict(sample));
        }
        return out;
    }

    void save(const std::string& filename) { ai.save(filename); }
    void load(const std::string& filename) { ai.load(filename); }
    void train(const std::string& filename) {
        throw std::runtime_error("method train(filename) is not used in this build");
        (void)filename;
    }
};

PYBIND11_MODULE(comand7, m) {
    py::class_<PyModel>(m, "Model")
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t>())
        .def("train_arbitraty", &PyModel::train_arbitraty)
        .def("predict_line", &PyModel::predict_line)
        .def("predict_point_class", &PyModel::predict_point_class)
        .def("fit", &PyModel::fit)
        .def("predict_proba", &PyModel::predict_proba)
        .def("predict", &PyModel::predict)
        .def("predict_proba_batch", &PyModel::predict_proba_batch)
        .def("predict_batch", &PyModel::predict_batch)
        .def("save", &PyModel::save)
        .def("load", &PyModel::load)
        .def("train", &PyModel::train);
}
