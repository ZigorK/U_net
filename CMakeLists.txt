cmake_minimum_required(VERSION 3.10)

project(unet)

# Установка типа сборки на Debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Поиск пакета OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Поиск пакета FLTK
find_package(FLTK REQUIRED)
include_directories(${FLTK_INCLUDE_DIRS})

# Добавление исходных файлов проекта
set(SOURCES
    main.cpp
    ConvolutionalLayer.cpp
    ReLULayer.cpp
    MaxPooling.cpp
    DeconvLayer.cpp
    ConcatLayer.cpp
    UNet.cpp
    Utils.cpp
    SGDOptimizer.cpp
    MainWindow.cpp
)

# Добавление заголовочных файлов проекта
set(HEADERS
    ConvolutionalLayer.h
    ReLULayer.h
    MaxPooling.h
    DeconvLayer.h
    ConcatLayer.h
    UNet.h
    Utils.h
    Optimizer.h
    SGDOptimizer.h
    Layer.h
    MainWindow.h
)

# Создание исполняемого файла
add_executable(unet ${SOURCES} ${HEADERS})

# Линковка с библиотеками
target_link_libraries(unet ${OpenCV_LIBS} ${FLTK_LIBRARIES})
