#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string caminho = "../assets/lenna.png";

    // 1️⃣ Ler a imagem
    cv::Mat img = cv::imread(caminho);

    if (img.empty()) {
        std::cerr << "Erro: não foi possível carregar a imagem: " << caminho << std::endl;
        return 1;
    }

    // 2️⃣ Criar nova imagem com metade da dimensão
    cv::Mat img_reduzida(img.rows / 2, img.cols / 2, img.type());

    // 3️⃣ Percorrer imagem original e copiar apenas pixels pares
    for (int y = 0; y < img.rows; y += 2) {
        for (int x = 0; x < img.cols; x += 2) {
            // A posição (y, x) da original vai para (y/2, x/2) na reduzida
            img_reduzida.at<cv::Vec3b>(y / 2, x / 2) = img.at<cv::Vec3b>(y, x);
        }
    }

    // 4️⃣ Mostrar resultados
    cv::imshow("Original", img);
    cv::imshow("Imagem Reduzida (pares apenas)", img_reduzida);
    cv::waitKey(0);

    return 0;
}
