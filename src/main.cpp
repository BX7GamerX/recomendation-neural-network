#include <iostream>
#include "NeuralNetwork.h"
// Function to test the NeuralNetwork classg++ -fdiagnostics-color=always -g -std=c++20 -I/home/him/Desktop/gainchain/repos/neural_network/include -o /home/him/Desktop/gainchain/repos/neural_network/bin/main /home/him/Desktop/gainchain/repos/neural_network/src/main.cpp /home/him/Desktop/gainchain/repos/neural_network/src/NeuralNetwork.cpp

void testNeuralNetwork() {
    int input_size = 5;
    int hidden_size = 3;
    int output_size = 2;
    double learning_rate = 0.01;

    NeuralNetwork nn(input_size, hidden_size, output_size, learning_rate);

    std::vector<double> input_data = {0.5, 0.1, 0.2, 0.7, 0.9};
    std::vector<double> expected_output = {0.0, 1.0}; // Dummy expected output for testing

    for (int i = 0; i < 1000; ++i) {  // Run multiple training iterations
        std::vector<double> output = nn.forward(input_data);
        nn.backpropagate(expected_output);

        if (i % 100 == 0) {  // Print output every 100 iterations
            std::cout << "Iteration " << i << " - Output: ";
            for (double value : output) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }
}



int main() {
    testNeuralNetwork();
       return 0;
}
