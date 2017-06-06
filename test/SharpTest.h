//
// Created by leonardoarcari on 31/05/17.
//

#ifndef SHARP_SHARPTEST_H
#define SHARP_SHARPTEST_H

#include <string>
#include <sstream>
#include <memory>

#include "../include/Sharp.h"

using namespace aapp;

void sharpSequential(const std::string &testShape,
                     const std::string &referencePath, int shapeSize,
                     double minTheta, double maxTheta, int thetaStep,
                     double lenThresh, int threads);

template <typename T>
std::unique_ptr<T> combineHough(const T& a, const T& b) {
  using distV = typename T::value_type;
  using lineV = typename distV::value_type;

  auto orientations = a.size();
  auto distances = a[0].size();

  auto combined = std::make_unique<T>(orientations, distV(distances));

  for (auto orn = 0; orn < orientations; ++orn) {
    for (auto dist = 0; dist < distances; ++dist) {
      (*combined)[orn][dist] = a[orn][dist] + b[orn][dist];
    }
  }

  return combined;
}

template<> std::unique_ptr<SharpContext::Slht>
combineHough(const SharpContext::Slht &a,
             const SharpContext::Slht &b);

template <typename T>
std::string printHough(const T& hough) {
  auto orientations = hough.size();
  auto distances = hough[0].size();

  auto ss = std::stringstream();
  ss << "{";

  for (auto orn = 0; orn < orientations; ++orn) {
    ss << "[";
    for (auto dist = 0; dist < distances; ++dist) {
      ss << hough[orn][dist] << ((dist < distances - 1) ? ", " : "");
    }
    ss << "]";
    ss << ((orn < orientations - 1) ? ", " : "");
  }

  ss << "}";

  return ss.str();
}

template<> std::string
printHough(const SharpContext::Slht& hough);

std::string printScore(const SharpContext::Score& score);

template <typename T> std::string compareHough(const T &a, const T &b) {
  auto orientations = a.size();
  auto distances = a[0].size();

  auto ss = std::stringstream();

  for (auto orn = 0; orn < orientations; ++orn) {
    for (auto dist = 0; dist < distances; ++dist) {
      if (a[orn][dist] != b[orn][dist]) {
        ss << "Mismatch at [theta = " << orn << ", r = " << dist << "] -> ("
           << a[orn][dist] << ", " << b[orn][dist] << ")\n";
      }
    }
  }

  return ss.str();
}

template<> std::string compareHough(const SharpContext::Slht& a, const SharpContext::Slht& b);

#endif //SHARP_SHARPTEST_H
