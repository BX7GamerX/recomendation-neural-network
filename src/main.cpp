#include <iostream>
#include "NeuralNetwork.h"
// Function to test the NeuralNetwork classg++ -fdiagnostics-color=always -g -std=c++20 -I/home/him/Desktop/gainchain/repos/neural_network/include -o /home/him/Desktop/gainchain/repos/neural_network/bin/main /home/him/Desktop/gainchain/repos/neural_network/src/main.cpp /home/him/Desktop/gainchain/repos/neural_network/src/NeuralNetwork.cpp

void testRecommendationAPI() {
    int input_size = 5;
    int hidden_size = 3;
    int output_size = 2;
    double learning_rate = 0.01;

    NeuralNetwork nn(input_size, hidden_size, output_size, learning_rate);

    // Sample JSON data
    json user_data = {
        {"likes", 300},
        {"dislikes", 25},
        {"duration", 150},
        {"shares", 90},
        {"comments", 45}
    };

    // Generate recommendations
    json recommendations = nn.generateRecommendations(user_data);

    // Print recommendations
    std::cout << "Recommended Content IDs: ";
    for (auto& id : recommendations["recommended_content_ids"]) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
}

int main() {
    testRecommendationAPI();
    return 0;
}
