//
// Author: Leonardo Arcari
// Mail: leonardo1[dot]arcari[at]gmail[dot]com
// Date: 17/05/17.
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
 *  - Threads ~ Number of parallel threads to use.
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
   * @param [in] threads The number of threads to use
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
   * <br>[Id * thetaStep * orientations / threads,
   *  (Id + 1) * thetaStep * orientations / threads - 1]<br>
   * Some adjustments are done to the above interval to ensure correct
   * discretization and non-overlapping intervals among threads.
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
   * Getter for number of threads to use.
   * @return number of threads used.
   */
  int threads() const {
    return _threads;
  }

  /**
   * Setter for number of threads to use.
   * @param [in] threads Number of threads to use.
   */
  void setThreads(int threads) {
    SharpContext::_threads = threads;
  }

  /**
   * @return list of available ReferenceShapes
   */
  const std::vector<aapp::ReferenceShape> &referenceShapes() const;

  /**
   * Add a ReferenceShape to the list of ReferenceShapes.
   * This method accepts both lvalue and rvalue as input, \p ref is
   * std::forwarded.
   * @param [in] ref ReferenceShape to add to ReferenceShapes list.
   */
  void addReferenceShape(aapp::ReferenceShape &&ref) {
    _referenceShapes.push_back(std::forward<aapp::ReferenceShape>(ref));
  }

  /**
   * Moves \p score to Scores Vault at \p processorId place.
   * @param [in] score Score to move
   * @param [in] processorId Scores Vault spot where to move \p score to
   */
  void sendScoreTo(std::unique_ptr<SharpContext::Score> score,
                   int processorId);

  /**
   * Moves Score located at \p processorId index in Scores Vault.
   * @param [in] processorId index in Scores Vault from which to read
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

/**
 * Top level function to run SHARP algorithm. This is what a user would call
 * if they wanted to execute SHARP on the \p testShape given a set of reference
 * shapes located at \p referencePath.
 * SHARP execution is composed of an initial sequential part.
 * In the first place, STIRS signature is computed for all the reference shapes
 * found in \p referencePath having file extension {.png, .jpg, .jpeg}. That
 * consists in detecting edges, computing SLHT and then STIRS signature out of
 * it.<br>
 * Then edge detection is performed (using Canny algorithm) on the test image.
 * From now on the algorithm runs parallelized on the number of threads
 * defined by \p threads execution the following steps:
 *  - SLHT computation.
 *  - STIRS signature computation.
 *  - Matching score evaluation towards all the reference shapes.
 *  - Merge partial results evaluated on each thread and compute final score.
 *
 * @param [in] testShape Image file {.png, .jpg, .jpeg} of the image to recognize
 * @param [in] referencePath Folder containing reference images files
 * @param [in] shapeSize An integer representing height and width of the Test
 *             input image
 * @param [in] minTheta The lower bound of the angles interval to consider for
 *                      shape orientation [degrees]
 * @param [in] maxTheta The upper bound of the angles interval to consider for
 *                      shape orientation [degrees]
 * @param [in] thetaStep The difference between two consecutive angles in our
 *                       discrete space
 * @param [in] lenThresh Minimum length starting from which we consider a
 *                       tangent segment to the shape valid to contribute to
 *                       STIRS signature
 * @param [in] threads Number of parallel threads to use
 */
void sharp(const std::string &testShape,
           const std::string &referencePath,
           int shapeSize,
           double minTheta,
           double maxTheta,
           int thetaStep,
           double lenThresh,
           int threads);

/**
 * Computes the Straight-Line Hough Transform of \p testShape for the angles
 * interval returned by SharpContext::getAnglesInterval for \p processorId.
 *
 * @param [in] testShape image which we compute SLHT for
 * @param [in] context SHARP execution context
 * @param [in] processorId Thread ID
 * @return a std::unique_ptr to a matrix of std::vector of aapp::Line
 */
std::unique_ptr<SharpContext::Slht>
partialSLHT(const cv::Mat &testShape,
            SharpContext &context,
            int processorId);

/**
 * Computes the STIRS signature of \p testShape for the angles
 * interval returned by SharpContext::getAnglesInterval for \p processorId.
 *
 * @param [in] slht SLHT out of which we evaluate STIRS signature.
 * @param [in] context SHARP execution context
 * @param [in] processorId Thread ID
 * @return a std::unique_ptr to a matrix of bool
 */
std::unique_ptr<SharpContext::Stirs>
partialSignature(const SharpContext::Slht &slht,
                 SharpContext &context,
                 int processorId);

/**
 * Computes a matching score of \p testStirs wrt \p refStirs for the angles
 * interval of testStirs returned by SharpContext::getAnglesInterval for
 * \p processorId.
 * @param [in] testStirs STIRS signature of test shape.
 * @param [in] refStirs STIRS signature to match \p testStirs to.
 * @param [in] context SHARP execution context
 * @param [in] processorId Thread ID
 * @return a std::unique_ptr to a vector of doubles. Each index represents the
 *         rotation angle of refStirs wrt which the matching point contained at
 *         that index is evaluated.
 */
std::unique_ptr<SharpContext::Score>
partialMatch(const SharpContext::Stirs &testStirs,
             const SharpContext::Stirs &refStirs,
             SharpContext &context,
             int processorId);

/**
 * Routine to merge each partial Score result from each thread. Since each
 * thread computes a matching score for an angles interval that does not
 * overlap with other threads, a simple vector sum is performed.
 * Here we use a binary-tree technique to bound the complexity of this process
 * to O(orientations * log_2(p)).
 * @param [in] score a moved std::unique_ptr (rvalue) to the vector of scores
 *             evaluated by \p processorId thread
 * @param [in] context SHARP execution context
 * @param [in] processorId Thread ID
 * @return a std::unique_ptr to the vector of scores. Only last thread returns
 *         a vector containing all the values collected by other threads.
 */
std::unique_ptr<SharpContext::Score>
participateInAdd(std::unique_ptr<SharpContext::Score> score,
                 SharpContext &context,
                 int processorId);

/**
 * Co-routine to evaluate STIRS signature for all the images {.png, .jpg, .jpeg}
 * located at \p refPath folder. The new STIRS signatures are added to
 * \p context set of reference shapes.
 * @param [in] refPath folder path in which to search for images
 * @param [in] context SHARP execution context
 */
void buildReferenceDB(const std::string &refPath,
                      SharpContext &context);
}

#endif // SHARP_SHARP_H
