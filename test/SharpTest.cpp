//
// Created by leonardoarcari on 31/05/17.
//

#include "SharpTest.h"

#include "../include/SharpSupport.h"

using namespace aapp;

std::unique_ptr<SharpContext::Slht> combineHough(const SharpContext::Slht &a,
                                                 const SharpContext::Slht &b) {
  using distV = typename SharpContext::Slht ::value_type;
  using lineV = typename distV::value_type;

  auto orientations = a.size();
  auto distances = a[0].size();

  auto combined = std::make_unique<SharpContext::Slht >(orientations, distV(distances));

  for (auto orn = 0; orn < orientations; ++orn) {
    for (auto dist = 0; dist < distances; ++dist) {
      (*combined)[orn][dist].insert((*combined)[orn][dist].end(), a[orn][dist].begin(), a[orn][dist].end());
      (*combined)[orn][dist].insert((*combined)[orn][dist].end(), b[orn][dist].begin(), b[orn][dist].end());
    }
  }

  return combined;
}

std::string printHough(const SharpContext::Slht &hough) {
  auto orientations = hough.size();
  auto distances = hough[0].size();

  auto ss = std::stringstream();
  ss << "{";

  for (auto orn = 0; orn < orientations; ++orn) {
    ss << "{";
    for (auto dist = 0; dist < distances; ++dist) {
      ss << "[";
      for (auto li = 0; li < hough[orn][dist].size(); ++li) {
        ss << *hough[orn][dist][li] << ((li < hough[orn][dist].size() - 1) ? ", " : "");
      }
      ss << "]";
      ss << ((dist < distances - 1) ? ", " : "");
    }
    ss << "}";
    ss << ((orn < orientations - 1) ? ", " : "");
  }

  ss << "}";

  return ss.str();
}

std::string printScore(const SharpContext::Score &score) {
  auto orientations = score.size();

  auto ss = std::stringstream();
  ss << "[";

  for (auto orn = 0; orn < orientations; ++orn) {
    ss << score[orn] << ((orn < orientations - 1) ? ", " : "");
  }
  ss << "]";

  return ss.str();
}

std::string compareHough(const SharpContext::Slht &a,
                         const SharpContext::Slht &b) {
  auto orientations = a.size();
  auto distances = a[0].size();

  auto ss = std::stringstream();

  for (auto orn = 0; orn < orientations; ++orn) {
    for (auto dist = 0; dist < distances; ++dist) {
      for (auto &line : a[orn][dist]) {
        auto found = std::find_if(b[orn][dist].begin(), b[orn][dist].end(),
                               [&line](auto &sp) { return *sp == *line; });
        if (found == std::end(b[orn][dist])) {
          ss << "Missing in b at [theta = " << orn << ", r = " << dist << "] -> "
             << *line << "\n";
        }
      }
      for (auto &line : b[orn][dist]) {
        auto found = std::find_if(a[orn][dist].begin(), a[orn][dist].end(),
                               [&line](auto &sp) { return *sp == *line; });
        if (found == std::end(a[orn][dist])) {
          ss << "Missing in a at [theta = " << orn << ", r = " << dist << "] -> "
             << *line << "\n";
        }
      }
    }
  }

  return ss.str();
}

bool compareSlht(const std::vector<std::unique_ptr<SharpContext::Slht>>& threadsv,
                 const std::vector<std::unique_ptr<SharpContext::Slht>>& halfv,
                 const SharpContext& context) {

  auto pow_k = static_cast<int>(std::pow(2, 0));
  auto pow_k1 = static_cast<int>(std::pow(2, 1));

  auto slhts = std::vector<std::unique_ptr<SharpContext::Slht>>();

  for (int pid = 0; pid < context.threads(); ++pid) {
    if (pid % pow_k1 == pow_k - 1) {
      LOG(DEBUG) << "Merging shlt from " << pid << " to " << pid+pow_k;
      slhts.push_back(combineHough(*threadsv[pid], *threadsv[pid+pow_k]));
    }
  }

  for (int pid = 0; pid < halfv.size(); ++pid) {
    LOG(DEBUG) << "[Interval " << pid << "]\n" << compareHough(*slhts[pid], *halfv[pid]);
  }
  return true;
}

void sharpSequential(const std::string &testShape,
                     const std::string &referencePath, int shapeSize,
                     double minTheta, double maxTheta, int thetaStep,
                     double lenThresh, int threads) {

  // Prepare algorithm context
  auto context = std::make_shared<SharpContext>(shapeSize, minTheta, maxTheta,
                                                thetaStep, lenThresh, threads);

  buildReferenceDB(referencePath, *context);

  LOG(DEBUG) << "Testing SHARP on test shape: " << testShape;

  // Load image
  auto tshape = cv::imread(testShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

  LOG(DEBUG) << "Binary test shape =\n" << cv::format(binaryTShape, cv::Formatter::FMT_PYTHON);

  auto slhts = std::vector<std::unique_ptr<SharpContext::Slht>>();
  auto stirss = std::vector<std::unique_ptr<SharpContext::Stirs>>();
  auto scores = std::vector<std::unique_ptr<SharpContext::Score>>();

  auto &ref = context->referenceShapes()[0];

  // Test for THREADS number of threads
  for (int pid = 0; pid < threads; ++pid) {
    slhts.push_back(partialSLHT(binaryTShape, *context, pid));
    LOG(DEBUG) << "[FakeThread " << pid << "] SLHT =\n" << printHough(*slhts[pid]);
  }

  for (int pid = 0; pid < threads; ++pid) {
    stirss.push_back(partialSignature(*(slhts[pid]), *context, pid));
    LOG(DEBUG) << "[FakeThread " << pid << "] STIRS =\n" << printHough(*stirss[pid]);
  }

  for (int pid = 0; pid < threads; ++pid) {
    scores.push_back(partialMatch(*(stirss[pid]), *ref.Stirs(), *context, pid));
    LOG(DEBUG) << "[FakeThread " << pid << "] Score =\n" << printScore(*scores[pid]);
  }

  // Test against THREADS/2 number of threads
  auto hslhts = std::vector<std::unique_ptr<SharpContext::Slht>>();
  auto hstirss = std::vector<std::unique_ptr<SharpContext::Stirs>>();
  auto hscores = std::vector<std::unique_ptr<SharpContext::Score>>();

  context->setThreads(threads/2);

  for (int pid = 0; pid < threads/2; ++pid) {
    hslhts.push_back(partialSLHT(binaryTShape, *context, pid));
    LOG(DEBUG) << "[FakeHalfThread " << pid << "] SLHT =\n" << printHough(*hslhts[pid]);
  }

  context->setThreads(threads);

  compareSlht(slhts, hslhts, *context);

  context->setThreads(threads/2);

  for (int pid = 0; pid < threads/2; ++pid) {
    hstirss.push_back(partialSignature(*(slhts[pid]), *context, pid));
    LOG(DEBUG) << "[FakeHalfThread " << pid << "] STIRS =\n" << printHough(*hstirss[pid]);
  }

  for (int pid = 0; pid < threads/2; ++pid) {
    hscores.push_back(partialMatch(*(stirss[pid]), *ref.Stirs(), *context, pid));
    LOG(DEBUG) << "[FakeHalfThread " << pid << "] Score =\n" << printScore(*hscores[pid]);
  }
}


