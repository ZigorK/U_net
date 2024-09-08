#include "MainWindow.h"
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <vector>
#include "UNet.h"
#include <FL/fl_ask.H>

// Конструктор
MainWindow::MainWindow(int w, int h, const char* title)
    : Fl_Window(w, h, title), predictButton(nullptr), unetModel(nullptr) {

    // Инициализация кнопки
    predictButton = new Fl_Button(10, 10, 100, 30, "Predict");
    predictButton->callback(predictCallback, this);

    end();
}

// Деструктор
MainWindow::~MainWindow() {
    // Освобождение ресурсов, если требуется
}

// Установка объекта модели
void MainWindow::setModel(UNet* model) {
    unetModel = model;
}

// Функция для преобразования std::vector в cv::Mat
cv::Mat MainWindow::vectorToMat(const std::vector<std::vector<std::vector<double>>>& vec) {
    int height = vec.size();
    int width = vec[0].size();
    int channels = vec[0][0].size();

    cv::Mat mat(height, width, CV_64FC(channels));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                mat.at<cv::Vec<double, 3>>(i, j)[c] = vec[i][j][c];
            }
        }
    }

    return mat;
}

// Callback функция для обработки нажатия на кнопку
void MainWindow::predictCallback(Fl_Widget* widget, void* data) {
    MainWindow* mw = static_cast<MainWindow*>(data);

    if (mw->unetModel) {
        // Предположим, что unetModel.forward() возвращает std::vector<std::vector<std::vector<double>>>
        std::vector<std::vector<std::vector<double>>> result = mw->unetModel->forward(mw->inputImage);

        // Преобразуем результат в cv::Mat
        mw->predictedMask = mw->vectorToMat(result);

        // Здесь можно добавить код для отображения результата
    } else {
        // Обработка случая, когда модель не установлена
        fl_alert("UNet model not set!");
    }
}
