//
// Created by leonardoarcari on 17/05/17.
//

#include "sharp.h"
#include "Line.h"
#include <iostream>
#include <memory>
#include <omp.h>

namespace aapp {

static std::vector<std::vector<std::vector<std::unique_ptr<Line>>>>
buildSLHT(unsigned int orientations, unsigned int distances) {
  auto slht = std::vector<std::vector<std::vector<std::unique_ptr<Line>>>>(
      orientations);
  for (auto &orientation : slht) {
    orientation.reserve(distances);
  }
  return slht;
}

void sharp(const std::string &testShape) {
  // Load image
  auto tshape = cv::imread(testShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

  // Debugging purpose
  showTwoImages(tshape, binaryTShape);

  // Compute partial shlt
  // Compute partial signature
  // Iterate over reference shapes and match
}

cv::Mat detectEdges(const cv::Mat &src) {
  auto dst = cv::Mat(src.size(), src.type());
  auto detected_edges = cv::Mat();
  auto src_gray = cv::Mat();

  cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  blur(src_gray, detected_edges, cv::Size(3, 3));
  Canny(detected_edges, detected_edges, 50, 50 * 3, 3);

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

void partialSLHT(const cv::Mat &testShape, SharpContext &context) {
  auto processorId = omp_get_thread_num();

  auto thetaInterval = context.getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  auto orientations =
      static_cast<unsigned int>((max - min) / context.thetaStep());
  auto distances =
      static_cast<unsigned int>(context.maxDist() - context.minDist() + 1);

  auto slht = buildSLHT(orientations, distances);

  for (int x = 0; x < testShape.rows; ++x) {
    for (int y = 0; y < testShape.cols; ++y) {
      if (testShape.at<unsigned char>(x, y) != 0) {
        for (double theta = min; theta <= max; theta += context.thetaStep()) {
          unsigned int t =
              static_cast<unsigned int>((theta - min) / context.thetaStep());
          unsigned int r = static_cast<unsigned int>(x * std::cos(theta) +
                                                     y * std::sin(theta));

          bool pointAdded = false;
          for (auto &line : slht[t][r]) {
            if (line) {
              auto p = Point{x, y};
              if (line->isAdjacient(p)) {
                line->addPoint(p);
                pointAdded = true;
              }
            }
          }
          if (!pointAdded) {
            slht[t][r].push_back(std::make_unique<Line>(Point{x, y}));
          }
        }
      }
    }
  }
}

SharpContext::SharpContext(unsigned int shapeSize, double minTheta,
                           double maxTheta, int thetaStep, double lenThreshold)
    : _shapeSize(shapeSize), _minTheta(minTheta), _maxTheta(maxTheta),
      _thetaStep(thetaStep), _lenThreshold(lenThreshold) {
  _minDist = -_shapeSize * maxSumSinCos;
  _maxDist = _shapeSize * maxSumSinCos;
  _orientations = static_cast<unsigned int>(
      std::floor(std::abs(_maxTheta - _minTheta) / omp_get_num_threads()));
}

std::pair<double, double> SharpContext::getAnglesInterval(int processorNo) {
  auto thetaMin =
      processorNo * _thetaStep * (_orientations / omp_get_num_threads());
  auto thetaMax =
      (processorNo + 1) * _thetaStep * (_orientations / omp_get_num_threads()) -
      1;
  return std::pair<double, double>(thetaMin, thetaMax);
}
}
