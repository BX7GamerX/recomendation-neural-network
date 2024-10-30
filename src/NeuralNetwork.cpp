#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "NeuralNetwork.h"

// Constructor
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
    // Initialize weights
    initializeWeights();
}

// Initialize weights with random values between -1 and 1
void NeuralNetwork::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    // Initialize weights for input to hidden layer
    weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights_input_hidden[i][j] = dist(gen);
        }
    }

    // Initialize weights for hidden to output layer
    weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights_hidden_output[i][j] = dist(gen);
        }
    }
}
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input_data) {
    // Calculate hidden layer output
    hidden_layer_output.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j) {
        hidden_layer_output[j] = 0.0;
        for (int i = 0; i < input_size; ++i) {
            hidden_layer_output[j] += input_data[i] * weights_input_hidden[i][j];
        }
        // Apply ReLU activation
        hidden_layer_output[j] = relu(hidden_layer_output[j]);
    }

    // Calculate output layer
    output_layer.resize(output_size);
    for (int k = 0; k < output_size; ++k) {
        output_layer[k] = 0.0;
        for (int j = 0; j < hidden_size; ++j) {
            output_layer[k] += hidden_layer_output[j] * weights_hidden_output[j][k];
        }
    }

    // Apply softmax activation to the output layer
    return softmax(output_layer);
}
// ReLU activation function
double NeuralNetwork::relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of ReLU (useful for backpropagation)
double NeuralNetwork::relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Softmax activation function
std::vector<double> NeuralNetwork::softmax(const std::vector<double>& input) {
    std::vector<double> result(input.size());
    double sum_exp = 0.0;

    // Calculate the sum of exponentials
    for (double val : input) {
        sum_exp += std::exp(val);
    }

    // Calculate softmax values
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i]) / sum_exp;
    }

    return result;
}
// Backpropagation placeholder (for training implementation later)
void NeuralNetwork::backpropagate(const std::vector<double>& expected_output) {
    // Calculate output layer error
    std::vector<double> output_errors(output_size);
    for (int k = 0; k < output_size; ++k) {
        output_errors[k] = output_layer[k] - expected_output[k];
    }

    // Backpropagate error to hidden layer
    std::vector<double> hidden_errors(hidden_size, 0.0);
    for (int j = 0; j < hidden_size; ++j) {
        for (int k = 0; k < output_size; ++k) {
            hidden_errors[j] += output_errors[k] * weights_hidden_output[j][k];
        }
        hidden_errors[j] *= relu_derivative(hidden_layer_output[j]);
    }

    // Update weights from hidden to output layer
    for (int j = 0; j < hidden_size; ++j) {
        for (int k = 0; k < output_size; ++k) {
            weights_hidden_output[j][k] -= learning_rate * output_errors[k] * hidden_layer_output[j];
        }
    }

    // Update weights from input to hidden layer
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights_input_hidden[i][j] -= learning_rate * hidden_errors[j] * hidden_layer_output[j];
        }
    }
}

// Update weights placeholder
void NeuralNetwork::updateWeights() {
    // Placeholder code: Update weights after error correction during training.
}

double calculateMSE(const std::vector<double>& expected_output, const std::vector<double>& actual_output) {
    double mse = 0.0;
    for (size_t i = 0; i < expected_output.size(); ++i) {
        mse += pow(expected_output[i] - actual_output[i], 2);
    }
    return mse / expected_output.size();
}
#include <algorithm> // For std::min and std::max

std::vector<double> NeuralNetwork::normalizeInput(const json& user_data) {
    std::vector<double> normalized_data;

    // Example normalization of engagement metrics
    double like_count = user_data["likes"];
    double dislike_count = user_data["dislikes"];
    double duration = user_data["duration"];
    double shares = user_data["shares"];
    double comments = user_data["comments"];

    // Define normalization (assuming max values for each metric for simplicity)
    double max_likes = 1000.0, max_dislikes = 500.0, max_duration = 300.0; // in seconds
    double max_shares = 500.0, max_comments = 200.0;

    // Normalize each metric to be in the range [0, 1]
    normalized_data.push_back(std::min(like_count / max_likes, 1.0));
    normalized_data.push_back(std::min(dislike_count / max_dislikes, 1.0));
    normalized_data.push_back(std::min(duration / max_duration, 1.0));
    normalized_data.push_back(std::min(shares / max_shares, 1.0));
    normalized_data.push_back(std::min(comments / max_comments, 1.0));

    return normalized_data;
}
std::vector<double> NeuralNetwork::forward(const json& user_data) {
    // Normalize JSON input
    std::vector<double> input_data = normalizeInput(user_data);

    // Use the existing forward function with the normalized input
    return forward(input_data);
}

// Function to generate recommendations
#include <nlohmann/json.hpp>
#include "NeuralNetwork.h"

using json = nlohmann::json;

// Function to generate recommendations
json NeuralNetwork::generateRecommendations(const json& user_data) {
    // Normalize the input JSON data
    std::vector<double> input_data = normalizeInput(user_data);

    // Perform forward pass to get output probabilities
    std::vector<double> output = forward(input_data);

    // Assuming output probabilities correspond to content IDs (e.g., 0, 1)
    // Generate recommendations based on the highest probability
    json recommendations;
    recommendations["recommended_content_ids"] = json::array();

    // Select the top 10 recommendations based on highest probabilities
    std::vector<size_t> sorted_indices(output.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&output](size_t i, size_t j) { return output[i] > output[j]; });

    for (size_t i = 0; i < std::min<size_t>(10, sorted_indices.size()); ++i) {
        recommendations["recommended_content_ids"].push_back(sorted_indices[i]);
    }

    return recommendations;
}


