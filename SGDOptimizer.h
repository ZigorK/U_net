#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H

#include "Optimizer.h"
#include <vector>

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(double learningRate);
    void updateWeights(std::vector<std::vector<std::vector<std::vector<double>>>>& weights,
                       const std::vector<std::vector<std::vector<std::vector<double>>>>& gradients) override;

private:
    double learningRate;
};

#endif // SGDOPTIMIZER_H
