#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

class Optimizer {
public:
    virtual void updateWeights(std::vector<std::vector<std::vector<std::vector<double>>>>& weights,
                               const std::vector<std::vector<std::vector<std::vector<double>>>>& gradients) = 0;
};

#endif // OPTIMIZER_H
