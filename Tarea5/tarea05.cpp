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

void compute_dft_fil(const Mat &src, Mat &dst, Size &dft_size)
{
  Mat tempA(dft_size, src.type(), Scalar::all(0));
  //Mat tempB(dftSize, B.type(), Scalar::all(0));

  // copy A and B to the top-left corners of tempA and tempB, respectively
  Mat roiA(tempA, Rect(0,0,src.cols,src.rows));
  src.copyTo(roiA);
  //Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
  //B.copyTo(roiB);

  Mat planes[2] = {Mat_<float>(tempA), Mat::zeros(tempA.size(), CV_32F)};
  Mat complexI;
  merge(planes, 2, complexI);

  dft(complexI, dst, DFT_COMPLEX_OUTPUT, src.rows);

}

void visualize_dft(const Mat &img_src, Mat &img_dst){
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

void applyfilter_freq(const Mat &src, Mat &dst, const Mat &kernel){
    Mat src_amplitude, kernel_amplitude, kernel_freq;
    Mat tmp_src_split [2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
    Mat tmp_kernel_split [2] = {Mat::zeros(kernel.size(), CV_32F), Mat::zeros(kernel.size(), CV_32F)};
    split(src, tmp_src_split);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    split(kernel, tmp_kernel_split);
    magnitude(tmp_src_split[0], tmp_src_split[1], src_amplitude);// planes[0] = magnitude
    magnitude(tmp_kernel_split[0], tmp_kernel_split[1], kernel_amplitude);// planes[0] = magnitude
    phase(tmp_src_split[0], tmp_src_split[1], tmp_src_split[1], false);

    kernel_freq = Mat::zeros(src.size(), CV_32F);
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = kernel.cols/2;
    int cy = kernel.rows/2;

    Mat q0(kernel_amplitude, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(kernel_amplitude, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(kernel_amplitude, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(kernel_amplitude, Rect(cx, cy, cx, cy)); // Bottom-Right

    int cx_k = kernel_freq.cols;
    int cy_k = kernel_freq.rows;

    Mat q0_k(kernel_freq, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1_k(kernel_freq, Rect(cx_k - cx, 0, cx, cy));  // Top-Right
    Mat q2_k(kernel_freq, Rect(0, cy_k - cy, cx, cy));  // Bottom-Left
    Mat q3_k(kernel_freq, Rect(cx_k - cx, cy_k - cy, cx, cy)); // Bottom-Right

    //Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(q0_k);
    q3.copyTo(q3_k);

    q1.copyTo(q1_k);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q2_k);
    Mat src_filtered, tmp[2];
    mulSpectrums(src_amplitude, kernel_freq, src_filtered, 0, false );


    polarToCart(src_filtered, tmp_src_split[1], tmp_src_split[0], tmp_src_split[1], false);
    merge(tmp_src_split,2,dst);
    namedWindow("tmp", WINDOW_AUTOSIZE );
    imshow("tmp", kernel_freq);



}

void compute_idft(const Mat &img_src, Mat &img_dst){
    dft(img_src, img_dst, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    normalize(img_dst, img_dst, 0, 1, CV_MINMAX);
}

void applyfilter_freq2(const Mat &src, Mat &dst, const Mat &kernel){
  // int dft_cols = getOptimalDFTSize(src.cols + kernel.cols - 1);
  // int dft_rows = getOptimalDFTSize(src.rows + kernel.rows - 1);

  Mat src_freq, kernel_freq, tmp_dst;
  compute_dft(src, src_freq);
  cv::Size tmp = src_freq.size();
  compute_dft_fil(kernel, kernel_freq, tmp);
  //compute_dft_conv(kernel, kernel_freq, dft_rows, dft_cols);

  mulSpectrums(src_freq, kernel_freq, src_freq,0,false);

  compute_idft(src_freq, tmp_dst);
  std::cout << "hola8" << '\n';

  tmp_dst(Rect(0, 0, src.cols, src.rows)).copyTo(dst);
  std::cout << "hola9" << '\n';

}


void GaussianFilterFreq(const Size &filter_size, Mat &kernel, int uX, int uY, float sigmaX, float sigmaY){
  Mat tmp_kernel = Mat(filter_size, CV_32F);
  float amplitude, x, y, tmp_value;

  amplitude = 1.0f;

  for (uint i = 0; i < filter_size.height; i++) {
    for (uint j = 0; j < filter_size.width; j++) {
      x = (((float)j - uX)*((float)j - uX))/(2.0f * sigmaX * sigmaX);
      y = (((float)i - uY)*((float)i - uY))/(2.0f * sigmaY * sigmaY);
      tmp_value = amplitude * exp(-(x + y));
      tmp_kernel.at<float>(i,j) = tmp_value;
    }
  }
  //normalize(tmp_kernel, tmp_kernel, 0.0f, 1.0f, NORM_MINMAX);
  kernel = tmp_kernel;
}

void convolveDFT(Mat &A, Mat &B, Mat &C){
    // reallocate the output array if needed
    //C.create(A.rows + B.rows - 1, A.cols + B.cols - 1, A.type());
    //std::cout << "size img freq: " << C.size() << " size img freq: " << C.size() <<'\n';
    Size dftSize;
    // calculate the size of DFT transform
    dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

    // allocate temporary buffers and initialize them with 0's
    Mat tempA(dftSize, A.type(), Scalar::all(0));
    Mat tempB(dftSize, B.type(), Scalar::all(0));

    // copy A and B to the top-left corners of tempA and tempB, respectively
    Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
    A.copyTo(roiA);
    Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
    B.copyTo(roiB);

    Mat planes1[2] = {Mat_<float>(tempA), Mat::zeros(tempA.size(), CV_32F)};
    Mat complexI1;
    merge(planes1, 2, complexI1);         // Add to the expanded another plane with zeros

    Mat planes2[2] = {Mat_<float>(tempB), Mat::zeros(tempB.size(), CV_32F)};
    Mat complexI2;
    merge(planes2, 2, complexI2);         // Add to the expanded another plane with zeros

    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    std::cout << "hola1" << '\n';
    dft(complexI1, complexI1, DFT_COMPLEX_OUTPUT, A.rows);
    std::cout << "hola2" << '\n';
    dft(complexI2, complexI2, 0, B.rows);

    // multiply the spectrums;
    // the function handles packed spectrum representations well
    mulSpectrums(complexI1, complexI2, complexI1,0,false);

    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    dft(complexI1, tempA, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE, C.rows);

    // now copy the result back to C.
    tempA(Rect((B.cols-1)/2, (B.rows-1)/2, A.cols, A.rows)).copyTo(C);
    normalize(C, C, 0, 1, CV_MINMAX);

    // all the temporary buffers will be deallocated automatically
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

    Mat img_inv_fourier,tmp_dst, img_inverted, kernel_freq,img_filtered_freq;
    Mat kernel_freq_tmp, kernel_freq_resized, img_filtered_freq_inv, kernel_freq_func;

    s = 15;
    sigma = (s+2)/6;
    kernel1d = cv::getGaussianKernel(s, sigma, CV_32F);
    kernel2d = kernel1d * kernel1d.t();

    cv::sepFilter2D(image, img_sepfilter2d_espacio, -1, kernel1d.t(), kernel1d, Point(-1,-1), 0, BORDER_DEFAULT);
    cv::filter2D(image, img_filter2d_espacio, -1, kernel2d, Point(-1,-1), 0, BORDER_DEFAULT);

    Mat out_conv;
    applyfilter_freq2(image, out_conv, kernel2d);

    compute_dft(image,img_inv_fourier);
    visualize_dft(img_inv_fourier,tmp_dst);
    compute_idft(img_inv_fourier,img_inverted);
    //GaussianFilterFreq(img_inv_fourier.size(), kernel_freq, int uX, int uY, float sigmaX, float sigmaY);
    //GaussianFilterFreq(Size(s,s), kernel_freq_func, s/2, s/2, sigma, sigma);

    //compute_dft(image,kernel_freq);

    compute_dft(kernel2d,kernel_freq);
    visualize_dft(kernel_freq,kernel_freq_tmp);


    //applyfilter_freq(img_inv_fourier, img_filtered_freq, kernel_freq);
    //std::cout << "size img freq: " << img_inv_fourier.size() << " size img filt freq: " << img_filtered_freq.size() <<'\n';
    //std::cout << "type img freq: " << img_inv_fourier.type() << " type img freq: " << img_filtered_freq.type() <<'\n';
    //compute_idft(img_filtered_freq,img_filtered_freq_inv);

    Mat img_filtered_freq_fun, img_filtered_freq_inv_fun;

    //applyfilter_freq2(img_inv_fourier, img_filtered_freq_fun, kernel_freq_func);
    //std::cout << "size img freq: " << kernel_freq_func.size() << " size img freq: " << kernel_freq_func.size() <<'\n';
    //std::cout << "size img freq: " << img_inv_fourier.size() << " size img filt freq: " << img_filtered_freq.size() <<'\n';
    //std::cout << "type img freq: " << img_inv_fourier.type() << " type img freq: " << img_filtered_freq.type() <<'\n';
    //compute_idft(img_filtered_freq_fun,img_filtered_freq_inv_fun);

    //
    // namedWindow("Freq Image", WINDOW_AUTOSIZE );
    // imshow("Freq Image", tmp_dst);
    //
    // namedWindow("Freq Inv Image", WINDOW_AUTOSIZE );
    // imshow("Freq Inv Image", img_inverted);
    //

    //convolveDFT(image, kernel2d, out_conv);
    //std::cout << "size img freq: " << img_sepfilter2d_espacio.size() << " size img freq: " << out_conv.size() <<'\n';


    namedWindow("out conv", WINDOW_AUTOSIZE );
    imshow("out conv", out_conv);
    //
    // namedWindow("kernel fun", WINDOW_AUTOSIZE );
    // imshow("kernel fun", kernel_freq_func);
    //
    // namedWindow("img_filtered_freq_inv_fun", WINDOW_AUTOSIZE );
    // imshow("img_filtered_freq_inv_fun", img_filtered_freq_inv_fun);
    //
    // namedWindow("Image filtered freq", WINDOW_AUTOSIZE );
    // imshow("Image filtered freq", img_filtered_freq_inv);
    // //
    // namedWindow("kernel freq", WINDOW_AUTOSIZE );
    // imshow("kernel freq", kernel_freq_tmp);
    // //
    // namedWindow("kernel freq resized", WINDOW_AUTOSIZE );
    // imshow("kernel freq resized", kernel_freq_resized);

    namedWindow("Original Image", WINDOW_AUTOSIZE );
    imshow("Original Image", image);
    namedWindow("Image Filtered: sepFilter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: sepFilter2D", img_sepfilter2d_espacio);
    namedWindow("Image Filtered: Filter2D", WINDOW_AUTOSIZE );
    imshow("Image Filtered: Filter2D", img_filter2d_espacio);
    waitKey(0);
    return 0;
}
