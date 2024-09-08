#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <opencv2/opencv.hpp>

class UNet; // Предварительное объявление класса UNet

class MainWindow : public Fl_Window {
public:
    MainWindow(int w, int h, const char* title = 0);
    ~MainWindow();

    static void predictCallback(Fl_Widget* widget, void* data);

    // Установить объект модели
    void setModel(UNet* model);

private:
    Fl_Button* predictButton;
    cv::Mat inputImage;    // Образцы входного изображения
    cv::Mat predictedMask; // Образцы предсказанной маски
    UNet* unetModel;       // Объект модели UNet

    // Преобразование std::vector в cv::Mat
    cv::Mat vectorToMat(const std::vector<std::vector<std::vector<double>>>& vec);
};

#endif // MAINWINDOW_H
