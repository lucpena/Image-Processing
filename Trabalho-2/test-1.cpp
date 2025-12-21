#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string path = "../assets/brain.png";
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    img = img(cv::Rect(0,0, 440, 420));

    if (img.empty()) {
        std::cerr << "Erro ao carregar imagem." << std::endl;
        return -1;
    }

    // garante que a imagem esteja em preto e branco
    cv::Mat img_mono = cv::Mat::zeros(img.size(), img.type());
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // aplicando filtros para ajudar
    int32_t GB_size = 3;
    cv::GaussianBlur(img, img, cv::Size(GB_size, GB_size), 0); // low

    int32_t B_size = 3;
    cv::medianBlur(img, img, B_size); // mid

    // equalização para aumentar contraste
    cv::Mat eq;
    cv::equalizeHist(img, eq);

    // threshold (limiar) forte para pegar só a região mais brilhante
    cv::Mat bin;
    cv::threshold(eq, bin, 250, 255, cv::THRESH_BINARY);

    // --- 4) Componentes conexos: escolher aquele com maior brilho médio ---
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(opened, labels, stats, centroids);

    int bestLabel = -1;
    double bestMean = 0;
    
    // começa em 1 (0 = fundo)
    for (int i = 1; i < n; i++) { 
        cv::Mat mask = (labels == i);

        // intensidade média no original
        double meanVal = cv::mean(img, mask)[0];

        if (meanVal > bestMean) {
            bestMean = meanVal;
            bestLabel = i;
        }
    }

    cv::Mat tumorMask = (labels == bestLabel);

    cv::Mat tumorMask8U;
    tumorMask.convertTo(tumorMask8U, CV_8U, 255);

    // --- 5) Mostrar ---
    cv::imshow("Original", img);
    cv::imshow("Equalized", eq);
    cv::imshow("Threshold 250", bin);
    cv::imshow("Opened", opened);
    cv::imshow("Tumor Detectado", tumorMask8U);

    cv::waitKey(0);
    return 0;
}
