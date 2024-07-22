#ifndef CONCATLAYER_H
#define CONCATLAYER_H

#include <vector>

class ConcatLayer {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input1, const std::vector<std::vector<std::vector<double>>>& input2);
};

#endif // CONCATLAYER_H
