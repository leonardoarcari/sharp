//
// Created by leonardoarcari on 18/05/17.
//

#ifndef SHARP_LINE_H
#define SHARP_LINE_H

#include <climits>
#include <opencv/cv.hpp>
namespace aapp {

/**
 * A convenient class to represent a point in a two-dimensional space.
 */
class Point {
public:
  /**
   * Default constructor. Instantiate a Point centered in (0, 0).
   */
  Point() : _x{0}, _y{0} {}
  /**
   * Instantiate a Point centered in (x, y)
   * @param x the value of X dimension
   * @param y the value of Y dimension
   */
  Point(int x, int y) : _x(x), _y(y) {}
  int _x; ///< Position on the X axis
  int _y; ///< Position on the Y axis
};

/**
 * A class representing a segment line in a two dimensional space.
 * A line segment is abstracted by its starting and ending point, its slope and
 * its length.
 * From the analytical geometry we know that the slope of a line varies from
 * [-inf, +inf]. For our purposes a vertical line is set to have a slope
 * equal to DBL_MAX. That's our infinite.
 *
 * We consider as a valid Line also a single Point. A single Point is a
 * degenerate Line, whose length is null.
 */
class Line {
public:
  /**
   * Instantiates a degenerate Line out of \p point. The new line has a null
   * length, while its slope is undefined.
   * @param point Starting point of Line segment.
   */
  explicit Line(Point point)
      : _start(point), _end{0, 0}, _length{0.0}, _degenerate{true} {}

  /**
   * Appends or prepends Point \p p to this Line segment. Point \p p must be
   * adjacent this Line (refer to Line::isAdjacent for a definition of
   * adjacent). If this was a degenerate Line, start and end points are set
   * and the segment line's slope and length are computed. If this was
   * already a proper Line only the length is updated.
   * @param p Point to prepend or append to this Line.
   */
  void addPoint(Point p);

  /**
   * @return the length of this Line segment.
   */
  double length() const;

  /**
   * Tests whether \p p is adjacent to this Line.
   * To define adjacency, it's important to understand the underlying space.
   * Our two-dimensional space is discrete and represented as a matrix whose
   * cells represent a (x, y) point. Two adjacent points on the same
   * horizontal line have same Y value and X values whose distance is 1.
   * Generalizing this consideration, a point is adjacent to a line segment
   * if lies on the same line and its X value is (start.X - 1) or (end.X + 1).
   *
   * If this is a degenerate line, every point "right around" the Line's one
   * is adjacent. "Right around" is analytically expressed by the distance
   * between the two points being less than sqrt(2).
   * @param p Point to test if it's adjacent to this Line.
   * @return true if \p is adjacent. False otherwise.
   */
  bool isAdjacient(Point p) const;

private:
  Point _start;
  Point _end;
  double _slope;
  double _length;
  bool _degenerate;
  constexpr static double sqrt2 = std::sqrt(2);
};

/**
 * Evaluates the euclidean distance between Point \p p and Point \p p.
 * @param p A Point
 * @param q A Point
 * @return euclidean distance between \p p and \p q.
 */
double distance(Point p, Point q);

/**
 * Evaluates the slope of a Line passing through Points \p p and \p q. If p.X
 * equals q.X theoretically the slope would be +inf or -inf. In our case,
 * it is set to +DBL_MAX or -DBL_MAX. That's our representation of infinite.
 * @param p A Point
 * @param q A Point
 * @return slope of a Line passing through \p p and \p q
 */
double slope(Point p, Point q);
}

#endif // SHARP_LINE_H
