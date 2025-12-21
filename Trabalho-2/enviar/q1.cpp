#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdint>
#include <cmath>

// Compilar com: g++ q1.cpp -o q1 `pkg-config --cflags --libs opencv4`
// Compilar e executar: g++ q1.cpp -o q1 `pkg-config --cflags --libs opencv4` && ./q1

using std::string;
using std::cout;
using std::cerr;
using std::cin;
using std::endl;

using std::pow;
using std::sqrt;

int main() {

    // caminho da imagem
    string path = "../assets/brain.png";

    cout << "\n > Lendo " << path << "...\n" << endl;

    // lendo a imagem
    cv::Mat img = cv::imread(path);
    img = img(cv::Rect(0,0, 440, 420));

    // copia a original
    cv::Mat img_original = img(cv::Rect(0,0, 440, 420));

    // checa se a imagem foi carregada
    if (img.empty()) {
        cerr << "\n > Error: Could not load image (" << path << ")." << endl;
        return 1;
    } else {
        cout << "\n > Imagem OK" << endl;
    }

    // garante que a imagem esteja em preto e branco
    cv::Mat img_mono = cv::Mat::zeros(img.size(), img.type());
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // aplicando filtros para ajudar
    int32_t GB_size = 3;
    cv::GaussianBlur(img, img, cv::Size(GB_size, GB_size), 0); // low

    int32_t B_size = 3;
    cv::medianBlur(img, img, B_size); // mid

    // histograma
    int32_t         histSize = 256;
    float           range[] = {0, 256};
    const float*    histRange = {range};
    bool            uniform = true, acc = false;
    cv::Mat         hist;

    cout << "\n > Calculando Histograma..." << endl;

    cv::calcHist (
        &img,       // entrada
        1,          // numero de imagens
        0,          // canal 0
        cv::Mat(),  // sem mascara
        hist,       // saida
        1,          // 1 dimensao
        &histSize,  // tamanho
        &histRange, // alcance
        uniform,
        acc
    );

    cout << "\n > Histograma OK" << endl;

    // aplicando o threshold OTSU
    cv::Mat bin;

    double th = cv::threshold(
        img,        // entrada
        bin,        // saida binaria
        0,          // valor inicial (ignored)
        255,        // pixels > threshold
        cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    cout << "\n > Limiar escolhido pelo OTSU: " << th << endl;
    cout << endl;

    // ajustando o valor do OTSU
    double ajustado = th + 95; 
    cv::Mat bin2;
    cv::threshold(img, bin2, ajustado, 255, cv::THRESH_BINARY);

    // operacoes
    int32_t kernelOpenValue = 3;
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelOpenValue, kernelOpenValue));

    int32_t kernelCloseValue = 9;
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelCloseValue, kernelCloseValue));

    // fechamento
    cv::Mat closed;
    cv::morphologyEx(bin2, closed, cv::MORPH_CLOSE, kernelClose);
    
    // abertura
    cv::Mat opened;
    cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernelOpen);

    // elementos conexos
    cv::Mat labels, stats, centroids;

    int numComp = cv::connectedComponentsWithStats(
        opened,
        labels,
        stats,
        centroids
    );

    int largestLabel = 1;
    int largestArea  = stats.at<int>(1, cv::CC_STAT_AREA);

    for (int i = 2; i < numComp; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > largestArea) {
            largestArea = area;
            largestLabel = i;
        }
    }

    cv::Mat tumorMask = (labels == largestLabel);
    cv::Mat tumorMask8U;
    tumorMask.convertTo(tumorMask8U, CV_8U, 255);

    // mostra as imagens
    cv::imshow("Original Image", img);
    // cv::imshow("Histogram", bin);
    // cv::imshow("Histogram Ajustado", bin2);
    // cv::imshow("Open", opened);
    // cv::imshow("Close", closed);
    // cv::imshow("Maior componente conexo (Tumor)", tumorMask8U);

    // cria uma imagem com a original e o tumor destacado
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::Mat resultado = imgColor.clone();

    // cor do tumor
    cv::Vec3b cor(0, 0, 255);

    // aplicar máscara colorida
    for (int y = 0; y < resultado.rows; y++) {
        for (int x = 0; x < resultado.cols; x++) {
            if (tumorMask8U.at<uchar>(y,x) == 255) {
                resultado.at<cv::Vec3b>(y,x) = cor;
            }
        }
    }

    // mostrar e salvar
    cv::imshow("Tumor Destacado", resultado);
    cv::imwrite("resultado_tumor.png", resultado);

    // salva a original com o corte
    cv::imwrite("brain_crop.png", img_original);

    cv::waitKey(0);

    return 0;
}