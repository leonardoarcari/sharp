//
// Created by leonardoarcari on 18/05/17.
//

#ifndef SHARP_LINE_H
#define SHARP_LINE_H

#include <climits>
#include <opencv/cv.hpp>
namespace aapp {

class Point {
public:
  Point() : _x{0}, _y{0} {}
  Point(int x, int y) : _x(x), _y(y) {}
  int _x;
  int _y;
};

class Line {
public:
  explicit Line(Point point)
      : _start(point), _end{0, 0}, _length{0.0}, _degenerate{true} {}

  void addPoint(Point p);

  double length() const;

  bool isAdjacient(Point p) const;

private:
  Point _start;
  Point _end;
  double _slope;
  double _length;
  bool _degenerate;
  constexpr static double sqrt2 = std::sqrt(2);
};

double distance(Point p, Point q);
double slope(Point p, Point q);
}

#endif // SHARP_LINE_H
