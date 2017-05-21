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

/**
 * @return A double long representation of pi.
 */
constexpr double pi() { return std::atan(1) * 4; }

class ReferenceShape;

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
  SharpContext(int shapeSize, double minTheta, double maxTheta,
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

  /**
   * @return list of available ReferenceShapes
   */
  const std::vector<ReferenceShape> &referenceShapes() const;

  /**
   * Add a ReferenceShape to the list of ReferenceShapes.
   * This method accepts both lvalue and rvalue as input, \p ref is
   * std::forwarded.
   * @param ref ReferenceShape to add to ReferenceShapes list.
   */
  void addReferenceShape(ReferenceShape &&ref) {
    _referenceShapes.push_back(std::forward<ReferenceShape>(ref));
  }

  /**
   * Moves \p score to Scores Vault at \p processorId place.
   * @param score Score to move
   * @param processorId Scores Vault spot where to move \p score to
   */
  void sendScoreTo(std::unique_ptr<SharpContext::Score> score,
                   int processorId) {
    _scoresVault[processorId] = std::move(score);
  }

  /**
   * Moves Score located at \p processorId index in Scores Vault.
   * @param processorId index in Scores Vault from which to read
   * @return Score located at \p processorId
   */
  std::unique_ptr<SharpContext::Score> receiveScore(int processorId) {
    return std::move(_scoresVault[processorId]);
  }

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

  // Data
  std::vector<ReferenceShape> _referenceShapes;
  std::vector<std::unique_ptr<SharpContext::Score>> _scoresVault;

  // Convenient constexpr
  constexpr static double
      maxSumSinCos = std::cos(pi() / 4) + std::sin(pi() / 4);
};

/**
 * Class representing a Reference Shape for SHARP. The purpose of a reference
 * shape is to be compared with the input test image and to evaluate a matching
 * score. Therefore a ReferenceShape is represented by:
 *  - its SharpContext::Stirs signature to be compared with the test shape's one
 *  - its file pathname
 */
class ReferenceShape {
public:
  using StirsPtr = std::unique_ptr<SharpContext::Stirs>;

  /**
   * Instantiate a ReferenceShape with \p path as file pathname and empty
   * SharpContext::Stirs signature.
   * @param path reference shape file pathname
   */
  explicit ReferenceShape(const std::string &path) : _path(path), _stirs() {}

  /**
   * Copy constructor that instantiates a ReferenceShape whose file pathname
   * is \p rs one and whose SharpContext::Stirs signature is copy-constructed
   * from \p rs one.
   * @param rs ReferenceShape to copy from
   */
  ReferenceShape(const ReferenceShape &rs)
      : _path(rs._path),
        _stirs(std::make_unique<SharpContext::Stirs>(*rs._stirs)) {}

  /**
   * Move constructor that instantiates a ReferenceShape whose file pathname
   * is \p rs one whose SharpContext::Stirs signature is move-constructed
   * from \p rs one.
   * @param rs ReferenceShape to move from
   */
  ReferenceShape(ReferenceShape &&rs)
      : _path(std::move(rs._path)), _stirs(std::move(rs._stirs)) {}

  /**
   * Sets this ReferenceShape's SharpContext::Stirs signature from \p stirs
   * @param stirs new SharpContext::Stirs signature of this ReferenceShape
   */
  void setStirs(StirsPtr stirs) { _stirs = std::move(stirs); }

  /**
   * @return a reference to this ReferenceShape's SharpContext::Stirs signature
   */
  const StirsPtr &Stirs() const { return _stirs; }

  /**
   * @return this ReferenceShape's file pathname
   */
  const std::string &path() const { return _path; }

private:
  StirsPtr _stirs;
  std::string _path;
};

void sharp(const std::string &testShape, const std::string &referencePath);

static std::unique_ptr<SharpContext::Slht>
partialSLHT(const cv::Mat &testShape, SharpContext &context);

static std::unique_ptr<SharpContext::Stirs>
partialSignature(const SharpContext::Slht &slht, SharpContext &context);

static std::unique_ptr<SharpContext::Score>
partialMatch(const SharpContext::Stirs &testStirs,
             const SharpContext::Stirs &refStirs,
             SharpContext &context);

static void participateInAdd(SharpContext &context);

// Co-routines

/**
 * Function to compute the edges of \p src image. It exploits OpenCV
 * and Canny algorithm.
 * @param [in] src image for which to detect edges.
 * @return an image whose non-zero pixels are the edges of shapes in \p src.
 */
static cv::Mat detectEdges(const cv::Mat &src);

/**
 * Handler for '--detect-edges' command-line option to detect edges of
 * \p testShape and display it in a window.
 * @param testShape
 */
void edgeDetectionOption(const std::string &testShape);

/**
 * Builds a two-dimensional vector of type T of size
 * \p orientations x \p distances.
 * @tparam T a default-constructable type
 * @param orientations first dimension cardinality
 * @param distances second dimension cardinality
 * @return a std::unique_ptr owning the allocated two-dimensional vector
 */
template<typename T>
static std::unique_ptr<T> buildHough(int orientations, int distances) {

  using distV = typename T::value_type;
  using lineV = typename distV::value_type;

  return std::make_unique<T>(orientations, distV(distances, lineV()));
}

/**
 * Builds a vector of SharpContext::Score of size \p orientations.
 * @param orientations vector cardinality
 * @return a std::unique_ptr owning the allocated vector
 */
static std::unique_ptr<SharpContext::Score> buildScore(int orientations);

// Utility procedures

/**
 * Displays two windows showing \p img1 and \p img2 that close on any key
 * pressing.
 * @param img1 First image to display
 * @param img2 Second image to display
 */
static void showTwoImages(const cv::Mat &img1, const cv::Mat &img2);
}

#endif // SHARP_SHARP_H
