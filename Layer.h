#ifndef LAYER_H
#define LAYER_H

#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;

    // Добавьте виртуальные методы save и load
    virtual void save(std::ofstream& file) const = 0;
    virtual void load(std::ifstream& file) = 0;

    // Остальные методы класса
};

#endif // LAYER_H
