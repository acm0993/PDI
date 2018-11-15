#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::microseconds us;
typedef std::chrono::duration<float> fsec;


void compute_dft(const Mat &img_src, Mat &img_dst)
{
    Mat padded;
    int m = getOptimalDFTSize( img_src.rows );
    int n = getOptimalDFTSize( img_src.cols ); // on the border add zero values
    copyMakeBorder(img_src, padded, 0, m - img_src.rows, 0, n - img_src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[2] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, img_dst, DFT_COMPLEX_OUTPUT);
}

void visualize_dft(const Mat &img_src, Mat &img_dst){
    std::cout << "size: " << img_src.size() << '\n';
    Mat planes [2] = {Mat::zeros(img_src.size(), CV_32F), Mat::zeros(img_src.size(), CV_32F)}; ;
    split(img_src, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    img_dst = planes[0];

    img_dst += Scalar::all(1);                    // switch to logarithmic scale
    log(img_dst, img_dst);
    // crop the spectrum, if it has an odd number of rows or columns
    img_dst = img_dst(Rect(0, 0, img_dst.cols & -2, img_dst.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = img_dst.cols/2;
    int cy = img_dst.rows/2;

    Mat q0(img_dst, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(img_dst, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(img_dst, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(img_dst, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(img_dst, img_dst, 0, 1, CV_MINMAX);
}
void compute_idft(const Mat &img_src, Mat &img_dst)
{
    dft(img_src, img_dst, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    normalize(img_dst, img_dst, 0, 1, CV_MINMAX);
}


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
    ofstream sepfilter_space_file, filter_space_file;
    std::string filename_filter_space = "time_filter_space.csv";
    std::string filename_sepfilter_space = "time_sepfilter_space.csv";
    cv::Mat kernel1d, kernel2d, img_filter2d_espacio, img_sepfilter2d_espacio, img_filter2d_freq, img_sepfilter2d_freq;
    int s;
    double sigma;
    high_resolution_clock::time_point t0, t1;
    fsec elapsed_time;

    s = 1;
    uint num_kernels = 25;
    uint num_interactions = 25;


    float data_array_sepfilter [num_kernels][num_interactions];
    float data_array_filter [num_kernels][num_interactions];

    for (uint i = 0; i < num_kernels; i++) {
      sigma = (s+2)/6;
      kernel1d = cv::getGaussianKernel(s, sigma, CV_32F);
      kernel2d = kernel1d * kernel1d.t();

      data_array_sepfilter[i][0] = (float)s;
      data_array_filter[i][0] = (float)s;
      for (uint j = 1; j < num_interactions + 1; j++) {
        t0 = Time::now();
        cv::sepFilter2D(image, img_sepfilter2d_espacio, -1, kernel1d.t(), kernel1d, Point(-1,-1), 0, BORDER_DEFAULT);
        t1 = Time::now();

        elapsed_time = t1 - t0;

        data_array_sepfilter[i][j] = elapsed_time.count();

        t0 = Time::now();
        cv::filter2D(image, img_filter2d_espacio, -1, kernel2d, Point(-1,-1), 0, BORDER_DEFAULT);
        t1 = Time::now();

        elapsed_time = t1 - t0;

        data_array_filter[i][j] = elapsed_time.count();

      }
      s = s + 2;
    }

    sepfilter_space_file.open(filename_sepfilter_space, std::ofstream::app);
    filter_space_file.open(filename_filter_space, std::ofstream::out | std::ofstream::app);

    if ((sepfilter_space_file.is_open()) && (filter_space_file.is_open())) {
      for (uint i = 0; i < num_interactions; i++) {
        for (uint j = 0; j < num_kernels; j++) {
          sepfilter_space_file << data_array_sepfilter[j][i] << ",";
          filter_space_file << data_array_filter[j][i] << ",";

        }
        sepfilter_space_file << "\n";
        filter_space_file << "\n";
      }
    }
    filter_space_file.close();
    sepfilter_space_file.close();

    Mat img_inv_fourier,tmp_dst, img_inverted;
    compute_dft(image,img_inv_fourier);
    visualize_dft(img_inv_fourier,tmp_dst);
    compute_idft(img_inv_fourier,img_inverted);


    namedWindow("Freq Image", WINDOW_AUTOSIZE );
    imshow("Freq Image", tmp_dst);

    namedWindow("Freq Inv Image", WINDOW_AUTOSIZE );
    imshow("Freq Inv Image", img_inverted);

    namedWindow("Original Image", WINDOW_AUTOSIZE );
    imshow("Original Image", image);
    namedWindow("Image Filtered: sepFilter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: sepFilter2D", img_sepfilter2d_espacio);
    namedWindow("Image Filtered: Filter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: Filter2D", img_filter2d_espacio);
    waitKey(0);
    return 0;
}
