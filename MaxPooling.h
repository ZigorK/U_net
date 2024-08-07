#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include <vector>
#include <fstream>

class MaxPooling {
public:
    MaxPooling(int kernel_size, int stride);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);

    void save(std::ofstream& file) const;
    void load(std::ifstream& file);

private:
    int kernel_size;
    int stride;
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<std::vector<std::vector<int>>> mask;
};

#endif // MAXPOOLING_H
