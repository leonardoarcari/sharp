//
// Author: Leonardo Arcari
// Mail: leonardo1[dot]arcari[at]gmail[dot]com
// Date: 22/05/17.
//

#include "../include/SharpSupport.h"

#include <boost/filesystem.hpp>
#include <iomanip>

namespace aapp {

cv::Mat detectEdges(const cv::Mat &src) {
  auto dst = cv::Mat(src.size(), src.type());
  auto detected_edges = cv::Mat();
  auto src_gray = cv::Mat();

  cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  blur(src_gray, detected_edges, cv::Size(3, 3));
  Canny(detected_edges, detected_edges, 50, 500, 3);

  dst = cv::Scalar::all(0);

  src.copyTo(dst, detected_edges);
  return dst;
}

void showTwoImages(const cv::Mat &img1, const cv::Mat &img2) {
  using namespace cv;

  namedWindow("Image 1", WINDOW_AUTOSIZE);
  namedWindow("Image 2", WINDOW_AUTOSIZE);
  imshow("Image 1", img1);
  imshow("Image 2", img2);

  waitKey(0);
}

std::unique_ptr<SharpContext::Score> buildScore(int orientations) {
  using size_type = std::vector<SharpContext::Score>::size_type;
  return std::make_unique<SharpContext::Score>(
      static_cast<size_type>(orientations), 0.0);
}

void edgeDetectionOption(const std::string &testShape) {

  // Load image
  auto tshape = cv::imread(testShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

  showTwoImages(tshape, binaryTShape);
}
}
