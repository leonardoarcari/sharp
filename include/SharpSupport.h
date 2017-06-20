//
// Created by leonardoarcari on 22/05/17.
//

#ifndef SHARP_SHARP_SUPPORT_H
#define SHARP_SHARP_SUPPORT_H

#include "Line.h"
#include "Sharp.h"
#include <omp.h>
#include <memory>

namespace aapp {

/**
 * Function to compute the edges of \p src image. It exploits OpenCV
 * and Canny algorithm.
 * @param [in] src image for which to detect edges.
 * @return an image whose non-zero pixels are the edges of shapes in \p src.
 */
cv::Mat detectEdges(const cv::Mat &src);

/**
 * Displays two windows showing \p img1 and \p img2 that close on any key
 * pressing.
 * @param img1 First image to display
 * @param img2 Second image to display
 */
void showTwoImages(const cv::Mat &img1, const cv::Mat &img2);

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
  explicit ReferenceShape(const std::string &path)
      : _path(path), _stirs() {}

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
      : _path(move(rs._path)), _stirs(move(rs._stirs)) {}

  /**
   * Sets this ReferenceShape's SharpContext::Stirs signature from \p stirs
   * @param stirs new SharpContext::Stirs signature of this ReferenceShape
   */
  void setStirs(StirsPtr stirs) { _stirs = move(stirs); }

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

/**
 * RAII wrapper to OpenMP library's omp_lock_t. When a OmpLock object is
 * constructed an omp_lock_t is initialized. OmpLock class exposes methods
 * that wrap plain C function calls of OpenMP. Finally, when the object goes
 * out of scope, on destruction the omp_lock_t is destroyed.
 */
class OmpLock {
public:
  /**
   * Constructor. Initializes internal omp_lock_t.
   */
  OmpLock() { omp_init_lock(&_lock); }
  /**
   * Sets the OmpLock by calling omp_set_lock()
   */
  void set() { omp_set_lock(&_lock); }
  /**
   * Unsets the OmpLock by calling omp_unset_lock()
   */
  void unset() { omp_unset_lock(&_lock); }
  /**
   * Destructor. Destroys internal omp_lock_t by calling omp_destroy_lock()
   */
  virtual ~OmpLock() { omp_destroy_lock(&_lock); }
private:
  omp_lock_t _lock;
};

/**
 * Builds a vector of SharpContext::Score of size \p orientations.
 * @param orientations vector cardinality
 * @return a std::unique_ptr owning the allocated vector
 */
std::unique_ptr<SharpContext::Score> buildScore(int orientations);

/**
 * Builds a two-dimensional vector of type T of size
 * \p orientations x \p distances.
 * @tparam T a default-constructable type
 * @param orientations first dimension cardinality
 * @param distances second dimension cardinality
 * @return a std::unique_ptr owning the allocated two-dimensional vector
 */
template <typename T>
std::unique_ptr<T> buildHough(int orientations, int distances) {

  using distV = typename T::value_type;
  using lineV = typename distV::value_type;

  return std::make_unique<T>(orientations, distV(distances));
}

/**
 * Handler for '--detect-edges' command-line option to detect edges of
 * \p testShape and display it in a window.
 * @param testShape
 */
void edgeDetectionOption(const std::string &testShape);
}

#endif // SHARP_SHARP_SUPPORT_H
