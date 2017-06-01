//
// Created by leonardoarcari on 17/05/17.
//

#ifndef SHARP_SHARP_H
#define SHARP_SHARP_H

#include "Line.h"

#include <climits>
#include <memory>
#include <opencv/cv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace aapp {

class ReferenceShape;
class OmpLock;

/**
 * @return A double long representation of pi.
 */
constexpr double pi() { return std::atan(1) * 4; }

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
 *
 * A SharpContext contains some structures to carry data required by SHARP:
 *  - Reference shapes ~ A list of ReferenceShapes of which we are testing
 *      the matching level with the input shape.
 *  - Scores Vault ~ A structure for exchanging partial matching scores
 *      among threads.
 */
class SharpContext {
public:
  using Slht = std::vector<std::vector<std::vector<std::shared_ptr<Line>>>>;
  using Acc = std::vector<std::vector<bool>>;
  using Stirs = std::vector<std::vector<bool>>;
  using Score = std::vector<double>;

  /**
   * Instantiates a SharpContext.
   * @param [in] shapeSize The Shape size
   * @param [in] minTheta The Minimum Theta
   * @param [in] maxTheta The Maximum Theta
   * @param [in] thetaStep The Theta step
   * @param [in] lenThreshold The length threshold
   */
  SharpContext(int shapeSize,
               double minTheta,
               double maxTheta,
               int thetaStep,
               double lenThreshold,
               int threads);

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
   * @return Minimum Theta angle
   */
  double minTheta() const {
    return _minTheta;
  }

  /**
   * @return Maximum Theta angle
   */
  double maxTheta() const {
    return _maxTheta;
  }

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

  /**
   * @return number of Orientations.
   */
  int orientations() const { return _orientations; }

  int threads() const {
    return _threads;
  }

  void setThreads(int _threads) {
    SharpContext::_threads = _threads;
  }

  /**
   * @return list of available ReferenceShapes
   */
  const std::vector<aapp::ReferenceShape> &referenceShapes() const;

  /**
   * Add a ReferenceShape to the list of ReferenceShapes.
   * This method accepts both lvalue and rvalue as input, \p ref is
   * std::forwarded.
   * @param ref ReferenceShape to add to ReferenceShapes list.
   */
  void addReferenceShape(aapp::ReferenceShape &&ref) {
    _referenceShapes.push_back(std::forward<aapp::ReferenceShape>(ref));
  }

  /**
   * Moves \p score to Scores Vault at \p processorId place.
   * @param score Score to move
   * @param processorId Scores Vault spot where to move \p score to
   */
  void sendScoreTo(std::unique_ptr<SharpContext::Score> score,
                   int processorId);

  /**
   * Moves Score located at \p processorId index in Scores Vault.
   * @param processorId index in Scores Vault from which to read
   * @return Score located at \p processorId
   */
  std::unique_ptr<SharpContext::Score> receiveScore(int processorId);

private:
  // SHARP parameters
  int _shapeSize;
  double _minTheta;
  double _maxTheta;
  double _minDist;
  double _maxDist;
  int _thetaStep;
  double _lenThreshold;
  int _orientations;
  int _threads;

  // Data
  std::vector<aapp::ReferenceShape> _referenceShapes;
  std::vector<std::pair<std::unique_ptr<aapp::SharpContext::Score>, bool>>
      _scoresVault;
  std::vector<aapp::OmpLock> _locks;

  // Convenient constexpr
  constexpr static double
      maxSumSinCos = std::cos(pi() / 4) + std::sin(pi() / 4);
};

void sharp(const std::string &testShape,
           const std::string &referencePath,
           int shapeSize,
           double minTheta,
           double maxTheta,
           int thetaStep,
           double lenThresh,
           int threads);

std::unique_ptr<SharpContext::Slht>
partialSLHT(const cv::Mat &testShape,
            SharpContext &context,
            int processorId);

std::unique_ptr<SharpContext::Stirs>
partialSignature(const SharpContext::Slht &slht,
                 SharpContext &context,
                 int processorId);

std::unique_ptr<SharpContext::Score>
partialMatch(const SharpContext::Stirs &testStirs,
             const SharpContext::Stirs &refStirs,
             SharpContext &context,
             int processorId);

std::unique_ptr<SharpContext::Score>
participateInAdd(std::unique_ptr<
    SharpContext::Score> score, SharpContext &context, int processorId);

void buildReferenceDB(const std::string &refPath,
                      SharpContext &context);
}

#endif // SHARP_SHARP_H
