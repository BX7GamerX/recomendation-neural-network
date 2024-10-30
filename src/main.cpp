#include <iostream>
#include "NeuralNetwork.h"
// Function to test the NeuralNetwork classg++ -fdiagnostics-color=always -g -std=c++20 -I/home/him/Desktop/gainchain/repos/neural_network/include -o /home/him/Desktop/gainchain/repos/neural_network/bin/main /home/him/Desktop/gainchain/repos/neural_network/src/main.cpp /home/him/Desktop/gainchain/repos/neural_network/src/NeuralNetwork.cpp

void testNeuralNetwork() {
    // Define network parameters
    int input_size = 5;         // Example input size
    int hidden_size = 3;        // Example hidden layer size
    int output_size = 2;        // Example output size (e.g., two content recommendations)
    double learning_rate = 0.01; // Example learning rate

    // Initialize the neural network
    NeuralNetwork nn(input_size, hidden_size, output_size, learning_rate);

    // Create dummy input data (normalized between 0 and 1)
    std::vector<double> input_data = {0.5, 0.1, 0.2, 0.7, 0.9};

    // Perform a forward pass
    std::vector<double> output = nn.forward(input_data);

    // Print the output values
    std::cout << "Neural Network Output:" << std::endl;
    for (double value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}


int main() {
    testNeuralNetwork();
       return 0;
}
