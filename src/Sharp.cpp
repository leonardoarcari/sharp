//
// Created by leonardoarcari on 17/05/17.
//

#include "../include/Sharp.h"
#include <chrono>
#include <iomanip>
#include <omp.h>

INITIALIZE_EASYLOGGINGPP

namespace aapp {

static void nonDegenerateLines(const SharpContext::Slht &slht) {
  for (unsigned t = 0; t < slht.size(); ++t) {
    for (unsigned r = 0; r < slht[t].size(); ++r) {
      for (auto &l : slht[t][r]) {
        if (l->length() > 2.0) {
          LOG(DEBUG) << "Line in (" << t << ", " << r
                     << ") [start: " << l->getStart()
                     << ", end: " << l->getEnd() << ", length: " << l->length()
                     << "]";
        }
      }
    }
  }
}

void sharp(const std::string &testShape) {
  // Load image
  auto tshape = cv::imread(testShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

  // Debugging purpose
  showTwoImages(tshape, binaryTShape);

  auto context = std::make_shared<SharpContext>(256, 0.0, 180, 5, 2.0);

  // Compute partial shlt
  auto slht = partialSLHT(binaryTShape, context);
  nonDegenerateLines(slht);

  // Compute partial signature
  // Iterate over reference shapes and match
}

cv::Mat detectEdges(const cv::Mat &src) {
  auto dst = cv::Mat(src.size(), src.type());
  auto detected_edges = cv::Mat();
  auto src_gray = cv::Mat();

  cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  blur(src_gray, detected_edges, cv::Size(3, 3));
  Canny(detected_edges, detected_edges, 50, 500, 3);

  dst = cv::Scalar::all(0);

  src.copyTo(dst, detected_edges);
  return dst;
}

void showTwoImages(const cv::Mat &img1, const cv::Mat &img2) {
  using namespace cv;

  namedWindow("Image 1", WINDOW_AUTOSIZE);
  namedWindow("Image 2", WINDOW_AUTOSIZE);
  imshow("Image 1", img1);
  imshow("Image 2", img2);

  waitKey(0);
}

SharpContext::Slht partialSLHT(const cv::Mat &testShape,
                               std::shared_ptr<SharpContext> &context) {
  auto processorId = omp_get_thread_num();

  auto thetaInterval = context->getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  LOG(DEBUG) << "Theta : [" << min << ", " << max << "]";
  LOG(DEBUG) << "r: [" << context->minDist() << ", " << context->maxDist()
             << "]";

  auto orientations = static_cast<int>((max - min) / context->thetaStep() + 1);
  auto distances =
      static_cast<int>(context->maxDist() - context->minDist() + 1);

  LOG(DEBUG) << "SLHT matrix size: " << orientations << " x " << distances;
  auto slht = buildHough<SharpContext::Slht>(orientations, distances);

  for (int x = 0; x < testShape.rows; ++x) {
    for (int y = 0; y < testShape.cols; ++y) {
      if (testShape.at<unsigned char>(x, y) != 0) {
        for (double theta = min; theta <= max; theta += context->thetaStep()) {
          auto theta_rad = theta * pi() / 180;
          auto t = static_cast<int>((theta - min) / context->thetaStep());
          auto r = static_cast<int>(x * std::cos(theta_rad) +
                                    y * std::sin(theta_rad));

          // r may have negative value so we apply an offset such that the
          // lowest possible value (context.minDist()) is indexed by 0.
          auto rIndex = static_cast<int>(r + std::abs(context->minDist()));
          bool appendedPoint = false;
          for (auto &line : slht[t][rIndex]) {
            if (line) {
              auto p = Point{x, y};
              if (line->isAdjacient(p)) {
                line->addPoint(p);
                appendedPoint = true;
              }
            }
          }
          if (!appendedPoint) {
            slht[t][rIndex].push_back(std::make_shared<Line>(Point{x, y}));
          }
        }
      }
    }
  }

  return slht;
}

void partialSignature(const SharpContext::Slht &slht,
                      std::shared_ptr<SharpContext> &context) {
  auto processorId = omp_get_thread_num();

  auto thetaInterval = context->getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  auto orientations =
      static_cast<unsigned int>((max - min) / context->thetaStep()) + 1;
  auto distances =
      static_cast<unsigned int>(context->maxDist() - context->minDist()) + 1;

  auto acc = buildHough<SharpContext::Acc>(orientations, distances);
  auto stirs = buildHough<SharpContext::Stirs>(orientations, distances);

  for (auto theta = min; theta <= max; theta += context->thetaStep()) {
    for (auto r = 0.0; r < distances; ++r) {
      auto theta_i = static_cast<unsigned int>(theta);
      auto r_i = static_cast<unsigned int>(r);

      for (auto &line : slht[theta_i][r_i]) {
        if (line && line->length() > context->lenThreshold()) {
          acc[theta_i][r_i] = true;
        }
      }
    }

    for (auto r = 0.0; r < distances; ++r) {
      unsigned int theta_i = static_cast<unsigned int>(theta);
      unsigned int r_i = static_cast<unsigned int>(r);

      if (acc[theta_i][r_i]) {
        for (auto rPrime = r + 1; r < distances; ++r) {
          auto rPrime_i = static_cast<unsigned int>(rPrime);
          if (acc[theta_i][rPrime_i]) {
            stirs[theta_i][rPrime_i - r_i] = true;
          }
        }
      }
    }
  }
}

static void configureLogger() {
  using namespace el;
  using namespace std::chrono;

  auto defaultConf = Configurations();
  defaultConf.setToDefault();
  defaultConf.setGlobally(ConfigurationType::ToFile, "true");

  auto now = system_clock::to_time_t(system_clock::now());
  auto timestamp = std::stringstream();
  timestamp << std::put_time(std::localtime(&now), "%T");
  defaultConf.setGlobally(ConfigurationType::Filename,
                          "/tmp/logs/sharp_" + timestamp.str() + ".log");
  defaultConf.setGlobally(ConfigurationType::Enabled, "true");
  defaultConf.setGlobally(ConfigurationType::SubsecondPrecision, "6");

  Helpers::installCustomFormatSpecifier(
      CustomFormatSpecifier("%omp_tid", [](auto m) {
        return "Thread " + std::to_string(omp_get_thread_num());
      }));
  defaultConf.setGlobally(ConfigurationType::Format, "[%omp_tid] %msg");
  Loggers::reconfigureLogger("default", defaultConf);
}

SharpContext::SharpContext(int shapeSize, double minTheta, double maxTheta,
                           int thetaStep, double lenThreshold)
    : _shapeSize(shapeSize), _minTheta(minTheta), _maxTheta(maxTheta),
      _thetaStep(thetaStep), _lenThreshold(lenThreshold) {

  // SHARP parameters
  _minDist = -_shapeSize * maxSumSinCos;
  _maxDist = _shapeSize * maxSumSinCos;
  _orientations = static_cast<int>(
      std::floor(std::abs(_maxTheta - _minTheta) / _thetaStep));

  auto sincos = maxSumSinCos;

  // Logger
  configureLogger();
}

std::pair<double, double> SharpContext::getAnglesInterval(int processorNo) {
  auto thetaMin =
      processorNo * _thetaStep * (_orientations / omp_get_num_threads());
  auto thetaMax =
      (processorNo + 1) * _thetaStep * (_orientations / omp_get_num_threads()) -
      1;

  return std::pair<double, double>(thetaMin, thetaMax);
}
}
