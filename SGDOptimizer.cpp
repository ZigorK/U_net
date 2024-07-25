#include "SGDOptimizer.h"

SGDOptimizer::SGDOptimizer(double learningRate) : learningRate(learningRate) {}

void SGDOptimizer::updateWeights(std::vector<std::vector<double>>& weights,
                                 const std::vector<std::vector<double>>& gradients) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learningRate * gradients[i][j];
        }
    }
}
