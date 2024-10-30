#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include <nlohmann/json.hpp>
//#include "nlohmann/json.hpp"

//#include <nlohmann/json.hpp>  // Include JSON parsing library
using json = nlohmann::json;

class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate);

    // Member functions
    std::vector<double> forward(const std::vector<double>& input_data);
    std::vector<double> forward(const json& user_data); // Overloaded function to handle JSON input

    void backpropagate(const std::vector<double>& expected_output);
    void updateWeights();

    // Data preprocessing
    std::vector<double> normalizeInput(const json& user_data);
    json generateRecommendations(const json& user_data);

private:
    // Neural network parameters
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    // Network layers and weights
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> hidden_layer_output;
    std::vector<double> output_layer;

    // Activation functions
    double relu(double x);
    double relu_derivative(double x);
    std::vector<double> softmax(const std::vector<double>& input);

    // Helper functions
    void initializeWeights();
};

#endif // NEURALNETWORK_H
