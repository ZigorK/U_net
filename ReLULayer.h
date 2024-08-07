#ifndef RELU_H
#define RELU_H

#include <vector>
#include <fstream>

class ReLU {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);

    void save(std::ofstream& file) const;
    void load(std::ifstream& file);

private:
    std::vector<std::vector<std::vector<double>>> input;
};

#endif // RELU_H
