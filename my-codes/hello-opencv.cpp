#include <opencv2/opencv.hpp>
#include <iostream>

// Compilar com: g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
// g++ hello-opencv.cpp -o hello `pkg-config --cflags --libs opencv4`

int main() {
    // Caminho da imagem
    std::string caminho = "../assets/lenna.png";

    // Lê a imagem
    cv::Mat img = cv::imread(caminho);

    // Verifica se carregou
    if (img.empty()) {
        std::cerr << "Erro: não consegui abrir a imagem!" << std::endl;
        return 1;
    }

    // Mostra a imagem em uma janela
    cv::imshow("Janela OpenCV", img);

    // Espera uma tecla (0 = infinito)
    cv::waitKey(0);

    return 0;
}