#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>
#include <cmath>

// Compilar com: g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
// Compilar e executar: g++ main.cpp -o main `pkg-config --cflags --libs opencv4` && ./main

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::string;

using std::pow;
using std::sqrt;

int main() {
    // Caminho da imagem
    string caminho = "../assets/lenna.png";

    // Lê a imagem
    cv::Mat img = cv::imread(caminho);

    // Verifica se a imagem foi lida corretamente
    if (img.empty()) {
        std::cerr << "Erro: não foi possível carregar a imagem (" << caminho << ")." << std::endl;
        return 1;
    }

    // ---------------------------------
    // Questão 1
    // ---------------------------------

    // Imagem vazia com a mesma dimensão e tipo da imagem original
    cv::Mat img2 = cv::Mat::zeros(img.size(), img.type());

    // Imagem vazia com a metade da dimensão e o mesmo tipo da imagem original
    cv::Mat img5(img.rows / 2, img.cols / 2, img.type()); 

    // Mostra informações da imagem
    cout << "\n---------------------------------" << endl;
    cout << "Imagem carregada: " << caminho.substr(caminho.find_last_of('/') + 1) << endl;
    cout << "img.cols: " << img.cols << endl;
    cout << "img.rows: " << img.rows << endl;
    cout << "img.size(): " << img.size() << endl;
    cout << "img.type(): " << img.type() << endl;

    uint32_t totalPixels = img.cols * img.rows;
    cout << "Total de pixels: " << totalPixels << endl;

    cout << "---------------------------------\n" << endl;
    
    // Percorre os pixels da imagem de 2 em 2
    for (size_t y = 0; y < img.rows; y += 2) {
        for (size_t x = 0; x < img.cols; x += 2) {
           
            // Copia os pixels pares para uma imagem do mesmo tamanho
            img2.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(y, x);

            // Copia os pixels pares para uma imagem com metade do tamanho
            img5.at<cv::Vec3b>(y / 2, x / 2) = img.at<cv::Vec3b>(y, x);

        }   
    }

    cv::Mat img3 = img2.clone();

    // Filtrando por linha
    for( size_t y = 0; y < img3.rows; y++ ){ // Todas as linhas
        for( size_t x = 1; x < img3.cols; x+=2 ){ // Somente colunas pares (pixel vazio)
            if( x < img3.cols - 1) {

                // Vizinhos
                cv::Vec3b left  = img2.at<cv::Vec3b>(y, x - 1);
                cv::Vec3b right = img2.at<cv::Vec3b>(y, x + 1);

                // Calcula a média
                cv::Vec3b media;

                for (size_t i = 0; i < 3; i++) {
                    media[i] = (left[i] + right[i]) / 2;
                }

                img3.at<cv::Vec3b>(y,x) = media;
            } else { // Para o último pixel
                img3.at<cv::Vec3b>(y,x) = img2.at<cv::Vec3b>(y,x-1);
            }
        }
    }

    cv::Mat img4 = img3.clone();

    // Filtrando por coluna
    for( size_t y = 1; y < img4.rows; y+=2 ){    // Somente linhas pares
        for( size_t x = 0; x < img4.cols; x++ ){ // Todas as colunas
            if( y < img4.rows - 1) {

                // Vizinhos
                cv::Vec3b up   = img3.at<cv::Vec3b>(y - 1, x);
                cv::Vec3b down = img3.at<cv::Vec3b>(y + 1, x);

                // Calcula a média
                cv::Vec3b media;

                for (size_t i = 0; i < 3; i++) {
                    media[i] = (up[i] + down[i]) / 2;
                }

                img4.at<cv::Vec3b>(y,x) = media;
            } else { // Para o último pixel
                img4.at<cv::Vec3b>(y,x) = img3.at<cv::Vec3b>(y-1,x);
            }
        }
    }

    // Mostra as imagens
    cv::imshow(caminho.substr(caminho.find_last_of('/') + 1), img);
    cv::imshow(caminho.substr(caminho.find_last_of('/') + 1) + " reduzida", img5);
    // cv::imshow("Image 2", img2);
    // cv::imshow("Image 3", img3);
    cv::imshow("Image 4", img4);

    // Calculando a Media Quadratica do Erro entre referencia e imagem processada
    double mse = 0.0;

    for (size_t y = 0; y < img.rows; y++) {
        for (size_t x = 0; x < img.cols; x++) {

            // Pixels a serem comparados
            cv::Vec3b p1 = img.at<cv::Vec3b>(y, x);
            cv::Vec3b p2 = img4.at<cv::Vec3b>(y, x);

            // Calcula o erro quadratico para cada canal de cor
            for (size_t i = 0; i < 3; i++) {
                double diff = static_cast<double>(p1[i]) - static_cast<double>(p2[i]);
                mse += diff * diff;
            }
        }
    }

    // Divide pelo numero total de pixels vezes os canais de cor
    mse /= (totalPixels * 3); 

    cout << "\n---------------------------------" << endl;
    cout << "MSE (Mean Squared Error)" << "\n > Questao 1: " << mse << endl;

    // ---------------------------------
    // Questão 2
    // ---------------------------------
/*
    // Aplicando o filtro de media 5x5 no dominio espacial
    cv::Mat img6 = img.clone();
    cv::blur(img, img6, cv::Size(5, 5));
    cv::imshow("Image 6 - Filtro 5x5", img6);

    // > Calculando o filtro Gaussiano no dominio da frequencia <
    
    // Convertendo para float
    cv::Mat imgFloat;
    img.convertTo(imgFloat, 5);

    // Calcular a DFT (Transformada Discreta de Fourier)
    cv::Mat dftImg;
    cv::dft(imgFloat, dftImg, cv::DFT_COMPLEX_OUTPUT);

    // Criar a mascara do filtro
    cv::Mat gaussFilter(dftImg.size(), 5);
    
    double sigma = 10.0;
    uint32_t cx = img.cols / 2;
    uint32_t cy = img.rows / 2;

    for( uint32_t y = 0; y < img.rows; y++ ) {
        for ( uint32_t x = 0; x < img.cols;x++ ) {

            double D = sqrt(double((x - cx) * (x- cx) + (y - cy) * (y - cy)));
            gaussFilter.at<float>(y, x) = static_cast<float>( exp(-(D * D) / (2.0 * sigma * sigma)));

        }
    }

    // Aplicar o filtro em ambos os canais
    cv::Mat planes[2];
    cv::split(dftImg, planes);

    planes[0] = planes[0].mul(gaussFilter);
    planes[1] = planes[1].mul(gaussFilter);

    cv::merge(planes, 2, dftImg);

    // Transformar de volta para ao dominio espacial
    cv::Mat imgBack;
    cv::idft(dftImg, imgBack, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Converter de volta para 8 bits
    cv::Mat img7;
    imgBack.convertTo(img7, 0);

    // Calculando o MSE entre o filtro 5x5 e o filtro Gaussiano
    double mse2 = 0.0;

    for (size_t y = 0; y < img.rows; y++) {
        for (size_t x = 0; x < img.cols; x++) {

            // Pixels a serem comparados
            cv::Vec3b p1 = img6.at<cv::Vec3b>(y, x);
            cv::Vec3b p2 = img7.at<cv::Vec3b>(y, x);

            // Calcula o erro quadratico para cada canal de cor
            for (size_t i = 0; i < 3; i++) {
                double diff = static_cast<double>(p1[i]) - static_cast<double>(p2[i]);
                mse2 += diff * diff;
            }
        }
    }

    cv::imshow("Image 7 - Filtro Gaussiano", img7);

    // Divide pelo numero total de pixels vezes os canais de cor
    mse2 /= (totalPixels * 3);

    cout << " > Questao 2: " << mse2 << endl;
    cout << "---------------------------------\n" << endl;
*/
  
    // Espera uma tecla
    cv::waitKey(0);

    return 0;
}