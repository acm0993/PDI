#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::microseconds us;
typedef std::chrono::duration<float> fsec;

//ofstream outputFile_sepfilter_space, outputFile_filter_space;


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
    // char* data_array_sepfilter [num_kernels][num_interactions];
    // char* data_array_filter [num_kernels][num_interactions];

    //std::cout << "hola1" << '\n';

    //sepfilter_space_file.open(filename_sepfilter_space, std::ofstream::out | std::ofstream::app);
    //filter_space_file.open(filename_filter_space, std::ofstream::out | std::ofstream::app);

    for (uint i = 0; i < num_kernels; i++) {
      sigma = (s+2)/6;
      kernel1d = cv::getGaussianKernel(s, sigma, CV_32F);
      kernel2d = kernel1d * kernel1d.t();

      //sprintf(data_array_sepfilter[i][0], "%d", s);
      //sprintf(data_array_filter[i][0], "%d", s);
      data_array_sepfilter[i][0] = (float)s;
      data_array_filter[i][0] = (float)s;

      //std::cout << "s: " << s << " sigma: "<< sigma << '\n';


      for (uint j = 1; j < num_interactions + 1; j++) {
        t0 = Time::now();
        cv::sepFilter2D(image, img_sepfilter2d_espacio, -1, kernel1d.t(), kernel1d, Point(-1,-1), 0, BORDER_DEFAULT);
        t1 = Time::now();

        elapsed_time = t1 - t0;

        //sprintf(data_array_sepfilter[i][j], "%f", elapsed_time.count());
        data_array_sepfilter[i][j] = elapsed_time.count();

        //std::cout << "data_array_filter " << data_array_filter[i][j] << " " << elapsed_time.count() << "s\n";

        //std::cout << elapsed_time.count() << "s\n";

        t0 = Time::now();
        cv::filter2D(image, img_filter2d_espacio, -1, kernel2d, Point(-1,-1), 0, BORDER_DEFAULT);
        t1 = Time::now();

        elapsed_time = t1 - t0;

        //sprintf(data_array_filter[i][j], "%f", elapsed_time.count());
        data_array_filter[i][j] = elapsed_time.count();

        //std::cout << "data_array_sepfilter " << data_array_sepfilter[i][j] << " " << elapsed_time.count() << "s\n";
      }
      s = s + 2;
    }

    //std::cout << "hola" << '\n';
    sepfilter_space_file.open(filename_sepfilter_space, std::ofstream::app);
    filter_space_file.open(filename_filter_space, std::ofstream::out | std::ofstream::app);

    if ((sepfilter_space_file.is_open()) && (filter_space_file.is_open())) {
      for (uint i = 0; i < num_interactions; i++) {
        for (uint j = 0; j < num_kernels; j++) {
          sepfilter_space_file << data_array_sepfilter[j][i] << ",";
          //std::cout << "data_array_sepfilter " << j << " " << i << " = " << data_array_sepfilter[j][i] <<'\n';
          filter_space_file << data_array_filter[j][i] << ",";
          //std::cout << "data_array_filter " << j << " " << i << " = " << data_array_filter[j][i] <<'\n';
        }
        sepfilter_space_file << "\n";
        filter_space_file << "\n";
      }
    }

    // sepfilter_space_file.write((char*)data_array_sepfilter,num_kernels*num_interactions*sizeof(float));
    // filter_space_file.write((char*)data_array_filter,num_kernels*num_interactions*sizeof(float));

    filter_space_file.close();
    sepfilter_space_file.close();


    // for (uint i = 0; i < num_kernels; i++) {
    //   for (uint j = 0; i < num_interactions; j++) {
    //     csv_sepfilter_space.write((char*)&data_array_sepfilter[i][j])
    //   }
    // }



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
