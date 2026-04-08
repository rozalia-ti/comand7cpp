#ifndef AI_H
#define AI_H

#include <vector>
#include <functional>
#include <random>

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

#endif
