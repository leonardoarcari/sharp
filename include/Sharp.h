//
// Created by leonardoarcari on 17/05/17.
//

#ifndef SHARP_SHARP_H
#define SHARP_SHARP_H

#include <climits>
#include <opencv/cv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <memory>
#include "Line.h"

namespace aapp {

/**
 * A SharpContext is an instance of a SHARP algorithm execution. SHARP
 * algorithm can be tuned according to a number of parameters and
 * SharpContext is the place where you set them. It is also responsible to
 * carry data to allow processors communication.
 * The parameters available for customization are the following:
 *  - Shape size ~ An integer representing height and width of the Test input
 *      image.
 *  - Min Theta ~ The lower bound of the angles interval to consider for
 *      shape orientation [degrees].
 *  - Max Theta ~ The upper bound of the angles interval to consider for
 *      shape orientation [degrees].
 *  - Theta step ~ The difference between two consecutive angles in our
 *      discrete space.
 *  - Length threshold ~ Minimum length starting from which we consider a
 *      tangent segment to the shape valid to contribute to STIRS signature.
 *
 * In addition to input parameters a number of other SHARP parameters are
 * evaluated, as they are a function of the input ones:
 *  - Min Distance ~ The minimum distance a line can have from the Hough
 *      Space origin. It equals <i>-shapeSize * (cos45° + sin45°)</i>
 *  - Max Distance ~ The maximum distance a line can have from the Hough
 *      Space origin. It equals <i>shapeSize * (cos45° + sin45°)</i>
 *  - Orientations ~ The number of orientations a line can have given the
 *      input interval <i>[Min Theta, Max Theta]</i> and the Theta step. It
 *      equals <i>abs(MaxTheta - MinTheta)/ThetaStep</i>.
 */
class SharpContext {
public:
  using Slht = std::vector<std::vector<std::vector<std::unique_ptr<Line>>>>;
  using Acc = std::vector<std::vector<bool>>;
  using Stirs = std::vector<std::vector<bool>>;

  /**
   * Instantiates a SharpContext.
   * @param [in] shapeSize The Shape size
   * @param [in] minTheta The Minimum Theta
   * @param [in] maxTheta The Maximum Theta
   * @param [in] thetaStep The Theta step
   * @param [in] lenThreshold The length threshold
   */
  SharpContext(unsigned int shapeSize, double minTheta, double maxTheta,
               int thetaStep, double lenThreshold);

  /**
   * Evaluates the interval of angles that processor running with Id = \p
   * processorNo should elaborate. The interval is evaluated as the following:
   * \f[
   *    \left \[processorNo \cdot \delta_{\theta} \cdot \frac{m_{\theta}}{p}
   *        processorNo \cdot \delta_{\theta} \cdot \frac{m_{\theta}}{p},
   *        (processorNo + 1) \cdot \delta_{\theta} \cdot
   *            \frac{m_{\theta}}{p} - 1
   *    \right \]
   * \f]
   * @param [in] processorNo Processor Id to evaluate angles interval for.
   * @return A std::pair where first is the lower bound of the interval and
   *    second is the upper bound.
   */
  std::pair<double, double> getAnglesInterval(int processorNo);

  /**
   * Getter for Theta step.
   * @return Theta step.
   */
  int thetaStep() const { return _thetaStep; }

  /**
   * Getter for Length threshold.
   * @return Length threshold.
   */
  double lenThreshold() const { return _lenThreshold; }

  /**
   * Getter for Min distance.
   * @return Min distance.
   */
  double minDist() const { return _minDist; }

  /**
   * Getter for Max distance.
   * @return Max distance.
   */
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
static void partialSLHT(const cv::Mat &testShape, SharpContext &context);
static void partialSignature(const SharpContext::Slht &slht,
                             SharpContext &context);

// Co-routines

/**
 * Function to compute the edges of \p src image. It exploits OpenCV
 * and Canny algorithm.
 * @param [in] src image for which to detect edges.
 * @return an image whose non-zero pixels are the edges of shapes in \p src.
 */
static cv::Mat detectEdges(const cv::Mat &src);

template<typename T>
static T buildHough(unsigned int orientations, unsigned int distances) {
  auto hough = T(orientations);
  for (auto &orientation : hough) {
    orientation.reserve(distances);
  }
  return hough;
}

// Utility procedures
static void showTwoImages(const cv::Mat &img1, const cv::Mat &img2);
}

#endif // SHARP_SHARP_H
