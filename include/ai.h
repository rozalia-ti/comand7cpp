#ifndef AI_H
#define AI_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using activation_fn_t = std::function<double(double)>;

struct AiLayer {
    std::vector<double> neurons;
    std::vector<std::vector<double>> synapses;
    std::vector<double> deltas;
    activation_fn_t activation_fn;
    activation_fn_t activation_fn_df;
};

struct AiLayerDescriptor {
    size_t layer_size, density;
    activation_fn_t activation_fn;
    activation_fn_t activation_fn_df;
};

class AiMatrix {
private:
    std::vector<AiLayer> layers;

    void clear_neurons() {
        for (auto& layer : layers) {
            layer.neurons.assign(layer.neurons.size(), 0.0);
            layer.deltas.assign(layer.deltas.size(), 0.0);
        }
    }

    void set_input(std::vector<double>& input) {
        layers[0].neurons = input;
    }

    void propagate() {
        for (size_t i = 0; i < layers.size(); i++) {
            for (size_t j = 0; j < layers[i].neurons.size(); j++) {
                auto& neuron = layers[i].neurons[j];

                for (size_t s = 0; s < layers[i].synapses.size(); s++) {
                    for (size_t k = 0; k < layers[i - s - 1].neurons.size(); k++) {
                        neuron +=
                            layers[i - s - 1].neurons[k]
                            * layers[i].synapses[s][j * layers[i - s - 1].neurons.size() + k];
                    }
                }

                neuron = layers[i].activation_fn(neuron);
            }
        }
    }

    std::vector<double> get_output() {
        return layers[layers.size() - 1].neurons;
    }

public:
    void configure(std::vector<AiLayerDescriptor>&& descriptors) {
        layers.clear();

        for (size_t i = 0; i < descriptors.size(); i++) {
            AiLayer layer;
            layer.neurons = std::vector<double>(descriptors[i].layer_size, 0);
            layer.synapses = std::vector<std::vector<double>>(descriptors[i].density, std::vector<double>());
            layer.deltas = std::vector<double>(descriptors[i].layer_size, 0);
            layer.activation_fn = descriptors[i].activation_fn;
            layer.activation_fn_df = descriptors[i].activation_fn_df;
            for (size_t j = descriptors[i].density; j > 0; j--) {
                if (i < j) {
                    continue;
                }

                layer.synapses[j - 1].resize(
                    descriptors[i - j].layer_size *
                    descriptors[i].layer_size,
                    0
                );
            }
            layers.push_back(layer);
        }
    }

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        for (auto& layer : layers) {
            for (auto& synapse : layer.synapses) {
                for (auto& w : synapse) {
                    w = dist(gen);
                }
            }
        }
    }

    std::vector<double> process(std::vector<double>& input) {
        clear_neurons();
        set_input(input);
        propagate();
        return get_output();
    }

    void train(std::vector<double>& input, std::vector<double>& target, double lr) {
        process(input);

        auto& last = layers[layers.size() - 1];

        for (size_t i = 0; i < last.neurons.size(); i++) {
            last.deltas[i] = (last.neurons[i] - target[i]) * last.activation_fn_df(last.neurons[i]);
        }

        for (int i = (int)layers.size() - 2; i >= 0; i--) {
            for (size_t j = 0; j < layers[i].neurons.size(); j++) {
                double error_sum = 0;

                for (size_t k = i + 1; k < layers.size(); k++) {
                    size_t dist = k - i;
                    if (layers[k].synapses.size() < dist) continue;

                    size_t s_idx = dist - 1;
                    const auto& weights = layers[k].synapses[s_idx];

                    for (size_t n_next = 0; n_next < layers[k].neurons.size(); n_next++) {
                        double w = weights[n_next * layers[i].neurons.size() + j];
                        error_sum += w * layers[k].deltas[n_next];
                    }
                }
                layers[i].deltas[j] = error_sum * layers[i].activation_fn_df(layers[i].neurons[j]);
            }
        }

        for (size_t i = 1; i < layers.size(); i++) {
            for (size_t s = 0; s < layers[i].synapses.size(); s++) {
                size_t prev_idx = i - s - 1;
                for (size_t j = 0; j < layers[i].neurons.size(); j++) {
                    for (size_t k = 0; k < layers[prev_idx].neurons.size(); k++) {
                        size_t w_idx = j * layers[prev_idx].neurons.size() + k;
                        double gradient = layers[i].deltas[j] * layers[prev_idx].neurons[k];
                        layers[i].synapses[s][w_idx] -= lr * gradient;
                    }
                }
            }
        }
    }
};

class MlpClassifier {
private:
    std::vector<size_t> layer_sizes;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> pre_activations;

    static double leaky_relu(double x) {
        return x > 0.0 ? x : 0.01 * x;
    }

    static double leaky_relu_df(double x) {
        return x > 0.0 ? 1.0 : 0.01;
    }

    static double sigmoid(double x) {
        if (x >= 40.0) return 1.0;
        if (x <= -40.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }

    void check_input(const std::vector<double>& input) const {
        if (layer_sizes.empty()) {
            throw std::runtime_error("network is not configured");
        }
        if (input.size() != layer_sizes.front()) {
            throw std::runtime_error("input size does not match network input layer");
        }
    }

    void forward(const std::vector<double>& input) {
        check_input(input);
        activations[0] = input;

        for (size_t layer = 1; layer < layer_sizes.size(); layer++) {
            const size_t prev_size = layer_sizes[layer - 1];
            const size_t curr_size = layer_sizes[layer];
            const size_t storage_idx = layer - 1;

            for (size_t neuron = 0; neuron < curr_size; neuron++) {
                double z = biases[storage_idx][neuron];
                for (size_t prev = 0; prev < prev_size; prev++) {
                    z += weights[storage_idx][neuron * prev_size + prev] * activations[layer - 1][prev];
                }

                pre_activations[layer][neuron] = z;
                activations[layer][neuron] = layer + 1 == layer_sizes.size() ? sigmoid(z) : leaky_relu(z);
            }
        }
    }

public:
    MlpClassifier() = default;

    MlpClassifier(size_t input_size, const std::vector<size_t>& hidden_layers, unsigned int seed = 1) {
        configure(input_size, hidden_layers, seed);
    }

    void configure(size_t input_size, const std::vector<size_t>& hidden_layers, unsigned int seed = 1) {
        if (input_size == 0) {
            throw std::runtime_error("input layer size must be positive");
        }

        layer_sizes.clear();
        layer_sizes.push_back(input_size);
        for (auto size : hidden_layers) {
            if (size == 0) {
                throw std::runtime_error("hidden layer size must be positive");
            }
            layer_sizes.push_back(size);
        }
        layer_sizes.push_back(1);

        weights.clear();
        biases.clear();
        activations.clear();
        pre_activations.clear();

        activations.reserve(layer_sizes.size());
        pre_activations.reserve(layer_sizes.size());
        for (auto size : layer_sizes) {
            activations.push_back(std::vector<double>(size, 0.0));
            pre_activations.push_back(std::vector<double>(size, 0.0));
        }

        std::mt19937 generator(seed);
        for (size_t layer = 1; layer < layer_sizes.size(); layer++) {
            const size_t prev_size = layer_sizes[layer - 1];
            const size_t curr_size = layer_sizes[layer];
            const double scale = layer + 1 == layer_sizes.size()
                ? std::sqrt(1.0 / static_cast<double>(prev_size))
                : std::sqrt(2.0 / static_cast<double>(prev_size));
            std::normal_distribution<double> dist(0.0, scale);

            std::vector<double> layer_weights(prev_size * curr_size, 0.0);
            for (auto& w : layer_weights) {
                w = dist(generator);
            }

            weights.push_back(layer_weights);
            biases.push_back(std::vector<double>(curr_size, 0.0));
        }
    }

    size_t input_size() const {
        return layer_sizes.empty() ? 0 : layer_sizes.front();
    }

    double predict_proba(const std::vector<double>& input) {
        forward(input);
        return activations.back()[0];
    }

    int predict(const std::vector<double>& input, double threshold = 0.5) {
        return predict_proba(input) >= threshold ? 1 : 0;
    }

    double train_one(const std::vector<double>& input, double target, double lr, double sample_weight = 1.0) {
        if (target != 0.0 && target != 1.0) {
            throw std::runtime_error("target must be 0 or 1");
        }
        if (lr <= 0.0) {
            throw std::runtime_error("learning rate must be positive");
        }

        forward(input);
        const double eps = 1e-12;
        const double proba = std::clamp(activations.back()[0], eps, 1.0 - eps);
        const double loss = -sample_weight * (target * std::log(proba) + (1.0 - target) * std::log(1.0 - proba));

        std::vector<std::vector<double>> deltas(layer_sizes.size());
        for (size_t layer = 1; layer < layer_sizes.size(); layer++) {
            deltas[layer] = std::vector<double>(layer_sizes[layer], 0.0);
        }

        deltas.back()[0] = sample_weight * (activations.back()[0] - target);

        for (size_t layer = layer_sizes.size() - 2; layer > 0; layer--) {
            const size_t next_storage_idx = layer;
            const size_t curr_size = layer_sizes[layer];
            const size_t next_size = layer_sizes[layer + 1];

            for (size_t neuron = 0; neuron < curr_size; neuron++) {
                double error = 0.0;
                for (size_t next = 0; next < next_size; next++) {
                    error += weights[next_storage_idx][next * curr_size + neuron] * deltas[layer + 1][next];
                }
                deltas[layer][neuron] = error * leaky_relu_df(pre_activations[layer][neuron]);
            }
        }

        for (size_t layer = 1; layer < layer_sizes.size(); layer++) {
            const size_t prev_size = layer_sizes[layer - 1];
            const size_t curr_size = layer_sizes[layer];
            const size_t storage_idx = layer - 1;

            for (size_t neuron = 0; neuron < curr_size; neuron++) {
                for (size_t prev = 0; prev < prev_size; prev++) {
                    weights[storage_idx][neuron * prev_size + prev] -= lr * deltas[layer][neuron] * activations[layer - 1][prev];
                }
                biases[storage_idx][neuron] -= lr * deltas[layer][neuron];
            }
        }

        return loss;
    }

    void fit(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<int>& targets,
        size_t epochs,
        double lr,
        unsigned int seed = 1,
        bool shuffle = true,
        bool balance_classes = true
    ) {
        if (inputs.empty()) {
            throw std::runtime_error("training data is empty");
        }
        if (inputs.size() != targets.size()) {
            throw std::runtime_error("inputs and targets have different lengths");
        }

        for (const auto& input : inputs) {
            check_input(input);
        }

        size_t positives = 0;
        for (auto target : targets) {
            if (target != 0 && target != 1) {
                throw std::runtime_error("targets must contain only 0 and 1");
            }
            positives += target == 1 ? 1 : 0;
        }

        const size_t negatives = targets.size() - positives;
        double positive_weight = 1.0;
        double negative_weight = 1.0;
        if (balance_classes && positives > 0 && negatives > 0) {
            positive_weight = static_cast<double>(targets.size()) / (2.0 * static_cast<double>(positives));
            negative_weight = static_cast<double>(targets.size()) / (2.0 * static_cast<double>(negatives));
        }

        std::vector<size_t> order(inputs.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 generator(seed);

        for (size_t epoch = 0; epoch < epochs; epoch++) {
            if (shuffle) {
                std::shuffle(order.begin(), order.end(), generator);
            }

            for (auto idx : order) {
                const double weight = targets[idx] == 1 ? positive_weight : negative_weight;
                train_one(inputs[idx], static_cast<double>(targets[idx]), lr, weight);
            }
        }
    }

    std::vector<double> predict_proba_batch(const std::vector<std::vector<double>>& inputs) {
        std::vector<double> output;
        output.reserve(inputs.size());
        for (const auto& input : inputs) {
            output.push_back(predict_proba(input));
        }
        return output;
    }
};

#endif
