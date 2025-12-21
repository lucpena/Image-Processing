#include <opencv2/opencv.hpp>
#include <iostream>
#include <iostream>
#include <cstdint>

using namespace cv;
using namespace std;

// executa o k-means e retorna labels e centros
void run_kmeans(const Mat& img, int32_t K, Mat& labels, Mat& centers) {
    Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);
    imgFloat = imgFloat.reshape(1, img.rows * img.cols);

    kmeans(imgFloat, K, labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
           3, KMEANS_PP_CENTERS, centers);

    centers = centers.reshape(3, centers.rows);
}

// converte um centro para HSV e retorna Vec3b
Vec3b centerRGBtoHSV(const Vec3f& centerRGB) {
    Mat px(1,1,CV_32FC3);
    px.at<Vec3f>(0) = centerRGB;

    Mat px8;
    px.convertTo(px8, CV_8UC3); // para cvtColor aceitar

    Mat pxHSV;
    cvtColor(px8, pxHSV, COLOR_BGR2HSV); // precisa estar em BGR
    
    return pxHSV.at<Vec3b>(0,0);
}

// tenta encontrar a pimenta
Mat extract_pepper(const Mat& img, int32_t K) {
    Mat blurImg;
    GaussianBlur(img, blurImg, Size(5,5), 0);

    Mat labels, centers;
    run_kmeans(blurImg, K, labels, centers);

    // labels: (N x 1) int, centers: (K x 3) float (B,G,R order)
    labels = labels.reshape(1, img.rows); // reshape para rows x cols
    labels.convertTo(labels, CV_32S);

    // identifica o vermelho
    vector<int32_t> redClusters;
    for (int i = 0; i < centers.rows; ++i) {
        Vec3f c = centers.at<Vec3f>(i); // float no formato BGR
        Vec3b hsv = centerRGBtoHSV(c);  // H:0-180, S,V:0-255

        int H = hsv[0];
        int S = hsv[1];
        int V = hsv[2];

        // ajustar para encontrar
        bool isRed = ( (H <= 3 || H >= 179) && (S >= 150 && S <= 255) && (V >= 100 && V <= 255) );
        if (isRed) redClusters.push_back(i);
    }

    // constroi uma mascara com o resultado
    Mat mask = Mat::zeros(img.size(), CV_8U);
    for (int32_t r = 0; r < img.rows; ++r) {
        for (int32_t c = 0; c < img.cols; ++c) {

            int32_t lab = labels.at<int32_t>(r,c);
            for (int32_t rc : redClusters) {
                if (lab == rc) { mask.at<uchar>(r,c) = 255; break; }
            }

        }
    }

    // pós-processamento morfológico
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 1);

    // mantem o maior componente
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {

        int32_t idxMax = 0;
        double  areaMax = 0;
        
        // encontrar maior contorno
        for (size_t i=0;i<contours.size();++i) {
            double a = contourArea(contours[i]);
            if (a > areaMax) { areaMax = a; idxMax = (int)i; }
        }

        Mat mask2 = Mat::zeros(mask.size(), CV_8U);
        drawContours(mask2, contours, idxMax, Scalar(255), FILLED);
        mask = mask2;
    }

    return mask;
}

// tenta encontrar a cebola
Mat extract_onion(const Mat& img, int32_t K) {
    Mat blurImg;
    GaussianBlur(img, blurImg, Size(5,5), 0);

    Mat labels, centers;
    run_kmeans(blurImg, K, labels, centers);

    // labels: (N x 1) int, centers: (K x 3) float (B,G,R order)
    labels = labels.reshape(1, img.rows); // reshape para rows x cols
    labels.convertTo(labels, CV_32S);

    // identifica o branco
    vector<int32_t> redClusters;
    for (int i = 0; i < centers.rows; ++i) {
        Vec3f c = centers.at<Vec3f>(i); // float no formato BGR
        Vec3b hsv = centerRGBtoHSV(c);  // H:0-180, S,V:0-255

        int H = hsv[0];
        int S = hsv[1];
        int V = hsv[2];

        // ajustar para encontrar
        bool isOnion = ( (H >= 10 && H <= 20) && (S >= 0 && S <= 255) && (V >= 0 && V <= 255) );
        if (isOnion) redClusters.push_back(i);
    }

    // constroi uma mascara com o resultado
    Mat mask = Mat::zeros(img.size(), CV_8U);
    for (int32_t r = 0; r < img.rows; ++r) {
        for (int32_t c = 0; c < img.cols; ++c) {

            int32_t lab = labels.at<int32_t>(r,c);
            for (int32_t rc : redClusters) {
                if (lab == rc) { mask.at<uchar>(r,c) = 255; break; }
            }

        }
    }

    // pós-processamento morfológico
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 1);

    // mantem o maior componente
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {

        int32_t idxMax = 0;
        double  areaMax = 0;
        
        // encontrar maior contorno
        for (size_t i=0;i<contours.size();++i) {
            double a = contourArea(contours[i]);
            if (a > areaMax) { areaMax = a; idxMax = (int)i; }
        }

        Mat mask2 = Mat::zeros(mask.size(), CV_8U);
        drawContours(mask2, contours, idxMax, Scalar(255), FILLED);
        mask = mask2;
    }

    return mask;
}

// tenta encontrar os vegetais
Mat extract_vegetables(const Mat& img, int32_t K) {
    Mat blurImg;
    GaussianBlur(img, blurImg, Size(5,5), 0);

    Mat labels, centers;
    run_kmeans(blurImg, K, labels, centers);

    // labels: (N x 1) int, centers: (K x 3) float (B,G,R order)
    labels = labels.reshape(1, img.rows); // reshape para rows x cols
    labels.convertTo(labels, CV_32S);

    // identifica os vegetais (cores basicamente)
    vector<int32_t> redClusters;
    for (int i = 0; i < centers.rows; ++i) {
        Vec3f c = centers.at<Vec3f>(i); // float no formato BGR
        Vec3b hsv = centerRGBtoHSV(c);  // H:0-180, S,V:0-255

        int H = hsv[0];
        int S = hsv[1];
        int V = hsv[2];

        // ajustar para encontrar
        bool isOnion = ( (H <= 50) && (S >= 0 && S <= 255) && (V >= 0 && V <= 255) );
        if (isOnion) redClusters.push_back(i);
    }

    // constroi uma mascara com o resultado
    Mat mask = Mat::zeros(img.size(), CV_8U);
    for (int32_t r = 0; r < img.rows; ++r) {
        for (int32_t c = 0; c < img.cols; ++c) {

            int32_t lab = labels.at<int32_t>(r,c);
            for (int32_t rc : redClusters) {
                if (lab == rc) { mask.at<uchar>(r,c) = 255; break; }
            }

        }
    }

    // pós-processamento morfológico
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 2);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,-1), 1);

    // mantem o maior componente
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {

        int32_t idxMax = 0;
        double  areaMax = 0;
        
        // encontrar maior contorno
        for (size_t i=0;i<contours.size();++i) {
            double a = contourArea(contours[i]);
            if (a > areaMax) { areaMax = a; idxMax = (int)i; }
        }

        Mat mask2 = Mat::zeros(mask.size(), CV_8U);
        drawContours(mask2, contours, idxMax, Scalar(255), FILLED);
        mask = mask2;
    }

    return mask;
}

int main() {
    Mat img = imread("../assets/onion.png");

    // roda k-means para valores de K
    for (int K : {3,4,5,6,7,8}) {

        // pimenta
        Mat mask = extract_pepper(img, K);
        Mat pepper;
        img.copyTo(pepper, mask);

        // imshow("Pepper - K=" + to_string(K), pepper);
        imwrite("./q2-img/Pepper - K = " + std::to_string(K) + ".png", pepper);

        // cebola
        mask = extract_onion(img, K);
        Mat onion;
        img.copyTo(onion, mask);

        // imshow("Onion - K=" + to_string(K), onion);
        imwrite("./q2-img/Onion - K = " + std::to_string(K) + ".png", onion);

        // verduras
        mask = extract_vegetables(img, K);
        Mat vegetables;
        img.copyTo(vegetables, mask);

        // imshow("Onion - K=" + to_string(K), vegetables);
        imwrite("./q2-img/Veggy - K = " + std::to_string(K) + ".png", vegetables);

    }

    imwrite("./q2-img/onion.png", img);

    waitKey(0);

    return 0;
}