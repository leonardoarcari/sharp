//
// Created by leonardoarcari on 17/05/17.
//

#ifndef SHARP_SHARP_H
#define SHARP_SHARP_H

#include <climits>
#include <opencv/cv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace aapp {
class SharpContext {
public:
  SharpContext(unsigned int shapeSize, double minTheta, double maxTheta,
               int thetaStep, double lenThreshold);

  std::pair<double, double> getAnglesInterval(int processorNo);
  int thetaStep() const { return _thetaStep; }
  double lenThreshold() const { return _lenThreshold; }

  double minDist() const { return _minDist; }
  double maxDist() const { return _maxDist; }

private:
  unsigned int _shapeSize;
  double _minTheta;
  double _maxTheta;
  double _minDist;
  double _maxDist;
  int _thetaStep;
  double _lenThreshold;
  unsigned int _orientations;

  constexpr static double maxSumSinCos = std::cos(45) + std::sin(45);
};

void sharp(const std::string &testShape);
void partialSLHT(const cv::Mat &testShape, SharpContext &context);

// Co-routines
static cv::Mat detectEdges(const cv::Mat &src);

// Utility procedures
static void showTwoImages(const cv::Mat &img1, const cv::Mat &img2);
}

#endif // SHARP_SHARP_H
