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

class Point {
public:
  Point() : _x{0}, _y{0} {}
  Point(int x, int y) : _x(x), _y(y) {}
  int _x;
  int _y;
};

double distance(Point p, Point q);
double slope(Point p, Point q);

class Line {
public:
  explicit Line(Point point)
      : _start(point), _end{0, 0}, _length{0.0}, _degenerate{true} {}

  void addPoint(Point p) {
    // If this is already a non degenerate line check for adjacency and update
    // _start or _end
    if (isAdjacient(p)) {
      if (!_degenerate) {
        if (p._x < _start._x) {
          _start = p;
        } else if (p._x > _end._x) {
          _end = p;
        }
      } else if (_degenerate) {
        // Otherwise if it's a degenerate line (i.e. a Point) we need to make it
        // a non degenerate line by setting a real _start and _end
        if (p._x < _start._x) {
          _end = _start;
          _start = p;
        } else if (p._x > _start._x) {
          _end = p;
        } else if (p._x == _start._x && p._y > _start._y) {
          _end = _start;
          _start = p;
        } else {
          _end = p;
        }
        _slope = slope(_start, _end);
        _degenerate = false;
      }
      _length = distance(_start, _end);
    }
  }

  double length() const { return (_degenerate ? 0.0 : _length); }

  bool isAdjacient(Point p) const {
    if (!_degenerate) {
      // In case we have a proper Line we need to check whether p lies to the
      // same line and is adjacent to either the _start or the _end
      bool posSlope = _slope >= 0;

      if (p._x == _start._x - 1) { // Test if p is new _start
        if (p._y == posSlope ? (_start._y - _slope) : (_start._y + _slope)) {
          return true;
        }
      } else if (p._x == _end._x + 1) { // Test if p is new _end
        if (p._y == posSlope ? (_start._y + _slope) : (_start._y - _slope)) {
          return true;
        }
      } else if (_slope == DBL_MAX &&
                 (p._y == _start._y + 1 || p._y == _end._y - 1)) {
        // In case of a vertical line (approximated with _slope == DBL_MAX) a
        // point is adjacent if it comes right before _start (higher y value) or
        // right after _end (lower y value)
        return true;
      }
    } else if (_degenerate) {
      // Any point around the current one in Line is adjacent. The maximum
      // distance is the one on the diagonal which is sqrt(2) long.
      if (0 < distance(p, _start) && distance(p, _start) <= sqrt2)
        return true;
    }
    return false;
  }

private:
  Point _start;
  Point _end;
  double _slope;
  double _length;
  bool _degenerate;
  constexpr static double sqrt2 = std::sqrt(2);
};

void sharp(const std::string &testShape);
void partialSLHT(const cv::Mat &testShape, SharpContext &context);

// Co-routines
static cv::Mat detectEdges(const cv::Mat &src);

// Utility procedures
static void showTwoImages(const cv::Mat &img1, const cv::Mat &img2);
}

#endif // SHARP_SHARP_H
