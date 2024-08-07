#include "SGDOptimizer.h"

SGDOptimizer::SGDOptimizer(double learningRate) : learningRate(learningRate) {}

void SGDOptimizer::updateWeights(std::vector<std::vector<std::vector<std::vector<double>>>>& weights,
                                 const std::vector<std::vector<std::vector<std::vector<double>>>>& gradients) {
    for (size_t oc = 0; oc < weights.size(); ++oc) {
        for (size_t ic = 0; ic < weights[oc].size(); ++ic) {
            for (size_t ki = 0; ki < weights[oc][ic].size(); ++ki) {
                for (size_t kj = 0; kj < weights[oc][ic][ki].size(); ++kj) {
                    weights[oc][ic][ki][kj] -= learningRate * gradients[oc][ic][ki][kj];
                }
            }
        }
    }
}
