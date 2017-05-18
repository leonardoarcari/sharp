#include "../include/Sharp.h"

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("usage: DisplayImage.out <ImagePath>\n");
    return -1;
  }

  aapp::sharp(argv[1]);

  //#pragma omp parallel num_threads(3)
  //  {
  //    std::cout << "Hello World! by thread #" << omp_get_thread_num() << "\n";
  //  }
  //
  //  auto image = cv::Mat();
  //  image = cv::imread(argv[1], cv::ImreadModes::IMREAD_UNCHANGED);
  //
  //  if (!image.data) {
  //    printf("No image data!\n");
  //    return -1;
  //  }
  //
  //  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  //  cv::imshow("Display Image", image);
  //
  //  cv::waitKey(0);

  return 0;
}