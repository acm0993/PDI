#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::microseconds us;
typedef std::chrono::duration<float> fsec;

ofstream outputFile_sepfilter_space, outputFile_filter_space;
ofstream csv_sepfilter_space, csv_filter_space;
std::string file_format = ".txt";
std::string filename_filter_space = "time_filter_space_n_";
std::string filename_sepfilter_space = "time_sepfilter_space_n_";

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat kernel1d, kernel2d, img_filter2d_espacio, img_sepfilter2d_espacio, img_filter2d_freq, img_sepfilter2d_freq;
    int s;
    double sigma;
    high_resolution_clock::time_point t0, t1;
    fsec elapsed_time;
    string n;

    //csv_sepfilter_space.open(outputFile_sepfilter_space,filename_sepfilter_space);
    //csv_filter_space.open(outputFile_filter_space,filename_filter_space);

    s = 5;
    //n = std::to_string(s);
    //std::cout << filename_sepfilter_space + n + file_format << '\n';
    sigma = (s+2)/6;
    kernel1d = cv::getGaussianKernel(s, sigma, CV_32F);

    kernel2d = kernel1d * kernel1d.t();

    t0 = Time::now();
    cv::sepFilter2D(image, img_sepfilter2d_espacio, -1, kernel1d.t(), kernel1d, Point(-1,-1), 0, BORDER_DEFAULT);
    t1 = Time::now();

    elapsed_time = t1 - t0;

    std::cout << elapsed_time.count() << "s\n";

    t0 = Time::now();
    cv::filter2D(image, img_filter2d_espacio, -1, kernel2d, Point(-1,-1), 0, BORDER_DEFAULT);
    t1 = Time::now();

    elapsed_time = t1 - t0;

    std::cout << elapsed_time.count() << "s\n";

    //std::cout << "sizekernel1d: " << kernel1d.size()<<'\n';
    //std::cout << "sizekernel2d: " << kernel2d.size()<<'\n';
    namedWindow("Original Image", WINDOW_AUTOSIZE );
    imshow("Original Image", image);
    namedWindow("Image Filtered: sepFilter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: sepFilter2D", img_sepfilter2d_espacio);
    namedWindow("Image Filtered: Filter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: Filter2D", img_filter2d_espacio);
    waitKey(0);
    return 0;
}
