//
// Author: Leonardo Arcari
// Mail: leonardo1[dot]arcari[at]gmail[dot]com
// Date: 18/05/17.
//

#include "../include/Line.h"

namespace aapp {

double distance(Point p, Point q) {
  return sqrt(std::pow((p._x - q._x), 2) + std::pow((p._y - q._y), 2));
}

double slope(Point p, Point q) {
  if (p._x - q._x == 0)
    return (p._y < q._y) ? -DBL_MAX : DBL_MAX;
  return (p._y - q._y) / (p._x - q._x);
}

void Line::addPoint(Point p) {
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
      _degenerate = false;
    }
    _slope = slope(_start, _end);
    _length = distance(_start, _end);
  }
}

bool Line::isAdjacient(Point p) const {
  if (!_degenerate) {
    // In case we have a proper Line we need to check whether p lies to the
    // same line and is adjacent to either the _start or the _end
    bool posSlope = _slope >= 0;

    if (p._x == _start._x - 1) { // Test if p is new _start
      if (p._y == posSlope ? (_start._y - _slope) : (_start._y + _slope)) {
        return true;
      }
    } else if (p._x == _end._x + 1) { // Test if p is new _end
      if (p._y == posSlope ? (_end._y + _slope) : (_end._y - _slope)) {
        return true;
      }
    } else if (_slope == std::abs(DBL_MAX) && p._x == _start._x && p._x == _end._x &&
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

double Line::length() const { return (_degenerate ? 0.0 : _length); }

bool Line::isDegenerate() const { return _degenerate; }
const Point &Line::getStart() const { return _start; }
const Point &Line::getEnd() const { return _end; }

std::ostream& operator<<(std::ostream& os, const aapp::Line& obj) {
  os << "(Start: " << obj._start << ", End: " << obj._end << ", Slope: " << obj._slope << ", Length: " << obj._length << ")";
  return os;
}

bool Line::operator==(const Line &ol) const {
  return _start == ol._start &&
      _end == ol._end &&
      _slope == ol._slope &&
      _length == ol._length;
}

std::ostream& operator<<(std::ostream& os, const aapp::Point& obj) {
  os << "(" << obj._x << ", " << obj._y << ")";
  return os;
}
}