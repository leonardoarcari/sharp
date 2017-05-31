//
// Created by leonardoarcari on 17/05/17.
//

#include "../include/Sharp.h"
#include "../include/Sharp_support.h"

#include <boost/filesystem.hpp>
#include <iomanip>

INITIALIZE_EASYLOGGINGPP

namespace aapp {

int inline d2i(double n) { return static_cast<int>(n); }

static double nonDegenerateLines(const SharpContext::Slht &slht) {
  double maxLength = 0.0;

  for (unsigned t = 0; t < slht.size(); ++t) {
    for (unsigned r = 0; r < slht[t].size(); ++r) {
      for (auto &l : slht[t][r]) {
        maxLength = std::max(maxLength, l->length());

        if (l->length() > 3.0) {
          LOG(DEBUG) << "Line in (" << t << ", " << r
                     << ") [start: " << l->getStart()
                     << ", end: " << l->getEnd() << ", length: " << l->length()
                     << "]";
        }
      }
    }
  }

  return maxLength;
}

static void stirsPoints(const SharpContext::Stirs &stirs) {
  for (unsigned t = 0; t < stirs.size(); ++t) {
    for (unsigned r = 0; r < stirs[t].size(); ++r) {
      if (stirs[t][r]) {
        LOG(DEBUG) << "Two segments having angle: " << t << " are " << r
                   << " far";
      }
    }
  }
}

static void buildReference(const std::string &refShape, SharpContext &context) {
  LOG(DEBUG) << "Building STIRST signature for " << refShape;

  // Load image
  auto tshape = cv::imread(refShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

  // Compute partial shlt
  auto slht = partialSLHT(binaryTShape, context);

  // Compute partial signature
  auto stirs = partialSignature(*slht, context);

  auto ref = ReferenceShape(refShape);
  ref.setStirs(std::move(stirs));

  context.addReferenceShape(std::move(ref));
}

static void buildReferenceDB(const std::string &refPath,
                             SharpContext &context) {
  using namespace boost::filesystem;

  int refsNo = 0;
  auto p = path(refPath);

  // Let the context know that this special part of the algorithm is run
  // by a single thread only.
  auto oldThreads = context.threads();
  context.setThreads(1);

  auto extensions = std::array<std::string, 3>{".jpg", ".jpeg", ".png"};
  try {
    if (exists(p)) {
      if (is_regular_file(p)) {
        LOG(DEBUG) << p << " is a file. Insert a directory path.";
      } else if (is_directory(p)) {
        LOG(DEBUG) << p << " contains the following reference shapes:";

        for (directory_entry &x : directory_iterator(p)) {
          auto x_path = x.path();
          auto ext_s = x_path.extension().string();
          std::transform(ext_s.begin(), ext_s.end(), ext_s.begin(),
                         [](auto c) { return std::tolower(c); });

          for (auto &ext : extensions) {
            if (ext_s == ext) {
              buildReference(x_path.string(), context);
              LOG(DEBUG) << "  " << x_path;
              ++refsNo;
            }
          }
        }
      }
    } else {
      LOG(DEBUG) << p << " does not exist\n";
    }
  } catch (const filesystem_error &ex) {
    LOG(DEBUG) << ex.what();
  }

  // Restore input number of threads.
  context.setThreads(oldThreads);
}



void sharp(const std::string &testShape, const std::string &referencePath,
           int shapeSize, double minTheta, double maxTheta, int thetaStep,
           double lenThresh, int threads) {
  // Prepare algorithm context
  auto context = std::make_shared<SharpContext>(shapeSize, minTheta, maxTheta,
                                                thetaStep, lenThresh, threads);
  buildReferenceDB(referencePath, *context);
  omp_set_num_threads(context->threads());

  LOG(DEBUG) << "Running SHARP on test shape: " << testShape;

  // Load image
  auto tshape = cv::imread(testShape);
  // Perform edge detection
  auto binaryTShape = detectEdges(tshape);

// Debugging purpose
// showTwoImages(tshape, binaryTShape);

#pragma omp parallel default(shared) shared(context, binaryTShape)
  {
    using Clock = std::chrono::steady_clock;
    decltype(Clock::now()) start;
    // Last thread, that collects all partial data at last, starts a clock
    if (omp_get_thread_num() == context->threads() - 1) {
      start = Clock::now();
    }

    // Compute partial shlt
    auto slht = partialSLHT(binaryTShape, *context);
    //nonDegenerateLines(*slht);

    // Compute partial signature
    auto stirs = partialSignature(*slht, *context);
    // stirsPoints(*stirs);

    // #pragma omp barrier
    // Iterate over reference shapes and match
    for (auto &ref : context->referenceShapes()) {
      auto score = partialMatch(*stirs, *ref.Stirs(), *context);
      LOG(DEBUG) << "Matching score for " << ref.path() << "\n  " << *score;
      score = participateInAdd(std::move(score), *context);
      if (omp_get_thread_num() == context->threads() - 1) {
        auto max = 0.0;
        auto max_index = 0;
        for (auto theta = 0; theta < score->size(); ++theta) {
          if ((*score)[theta] > max) {
            max = (*score)[theta];
            max_index = theta;
          }
        }

        LOG(INFO) << ref.path() << "\n  Peak found at [theta = "
                  << max_index * context->thetaStep() << ", score = " << max
                  << "]";
      }
#pragma omp barrier
    }

    if (omp_get_thread_num() == context->threads() - 1) {
      auto end = Clock::now();
      auto diff =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      LOG(INFO) << "Time spent in shape recognition: " << diff.count()
                << "ms\n";
    }
  }
}

std::unique_ptr<SharpContext::Slht> partialSLHT(const cv::Mat &testShape,
                                                SharpContext &context) {
  auto processorId = omp_get_thread_num();

  auto thetaInterval = context.getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  LOG(DEBUG) << "Theta : [" << min << ", " << max << "]";
  LOG(DEBUG) << "r: [" << context.minDist() << ", " << context.maxDist() << "]";

  auto distances = d2i(context.maxDist() - context.minDist() + 1);

  LOG(DEBUG) << "SLHT matrix size: " << context.orientations() << " x "
             << distances;
  auto slht = buildHough<SharpContext::Slht>(context.orientations(), distances);

  for (int x = 0; x < testShape.rows; ++x) {
    for (int y = 0; y < testShape.cols; ++y) {
      if (testShape.at<unsigned char>(x, y) != 0) {
        for (double theta = min; theta <= max; theta += context.thetaStep()) {
          auto theta_rad = theta * pi() / 180;
          auto t = d2i(theta / context.thetaStep());
          auto r = d2i(x * std::cos(theta_rad) + y * std::sin(theta_rad));

          // r may have negative value so we apply an offset such that the
          // lowest possible value (context.minDist()) is indexed by 0.
          auto rIndex = d2i(r + std::abs(context.minDist()));
          bool appendedPoint = false;
          auto &slht_ref = *slht;
          for (auto &line : (*slht)[t][rIndex]) {
            if (line) {
              auto p = Point{x, y};
              if (line->isAdjacient(p)) {
                auto &l = *line;
                l.addPoint(p);
                appendedPoint = true;
              }
            }
          }
          if (!appendedPoint) {
            (*slht)[t][rIndex].push_back(std::make_shared<Line>(Point{x, y}));
          }
        }
      }
    }
  }

  return slht;
}

std::unique_ptr<SharpContext::Stirs>
partialSignature(const SharpContext::Slht &slht, SharpContext &context) {
  auto processorId = omp_get_thread_num();

  auto thetaInterval = context.getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  LOG(DEBUG) << "Evaluating signature for angles [" << min << ", " << max
             << "]";

  auto distances =
      static_cast<unsigned int>(context.maxDist() - context.minDist()) + 1;

  auto acc = buildHough<SharpContext::Acc>(context.orientations(), distances);
  auto stirs =
      buildHough<SharpContext::Stirs>(context.orientations(), distances);

  for (auto theta = min; theta < max; theta += context.thetaStep()) {
    auto t_i = d2i(theta / context.thetaStep());

    for (auto r = 0.0; r < distances; ++r) {
      auto r_i = d2i(r);

      for (auto &line : slht[t_i][r_i]) {
        if (line && line->length() > context.lenThreshold()) {
          (*acc)[t_i][r_i] = true;
        }
      }
    }

    for (auto r = 0.0; r < distances; ++r) {
      int r_i = d2i(r);

      if ((*acc)[t_i][r_i]) {
        for (auto rPrime = r + 1; rPrime < distances; ++rPrime) {
          auto rPrime_i = d2i(rPrime);
          if ((*acc)[t_i][rPrime_i]) {
            (*stirs)[t_i][rPrime_i - r_i] = true;
          }
        }
      }
    }
  }
  return stirs;
}

std::unique_ptr<SharpContext::Score>
partialMatch(const SharpContext::Stirs &testStirs,
             const SharpContext::Stirs &refStirs, SharpContext &context) {

  auto processorId = omp_get_thread_num();

  auto thetaInterval = context.getAnglesInterval(processorId);
  auto min = thetaInterval.first;
  auto max = thetaInterval.second;

  auto distances =
      static_cast<unsigned int>(context.maxDist() - context.minDist()) + 1;

  auto score = buildScore(context.orientations());

  for (auto theta_1 = context.minTheta(); theta_1 < context.maxTheta();
       theta_1 += context.thetaStep()) {
    int match = 0, approx = 0, miss = 0;
    auto theta_1_i = d2i(theta_1 / context.thetaStep());

    for (auto theta_2 = min; theta_2 < max; theta_2 += context.thetaStep()) {
      auto theta_2_i = d2i(theta_2 / context.thetaStep());
      auto t_i = (d2i((theta_1 + theta_2) / context.thetaStep()) %
                  context.orientations());

      for (auto r = 0.0; r < distances; ++r) {
        auto r_i = d2i(r);

        if (refStirs[t_i][r_i]) {
          if (testStirs[theta_2_i][r_i])
            match += 1;
          else if (r_i < distances - 1 && testStirs[theta_2_i][r_i + 1] ||
                   r_i > 0.0 && testStirs[theta_2_i][r_i - 1])
            approx += 1;
          else
            miss += 1;
        }
      }
    }

    (*score)[theta_1_i] = match + 0.5 * approx - miss;
  }

  return score;
}

std::unique_ptr<SharpContext::Score>
participateInAdd(std::unique_ptr<SharpContext::Score> score,
                 SharpContext &context) {

  auto processorId = omp_get_thread_num();
  auto logP = d2i((std::log2(context.threads())));

  auto localScore = std::move(score);

  for (int k = 0; k < logP; ++k) {
    auto pow_k = d2i(std::pow(2, k));
    auto pow_k1 = d2i(std::pow(2, k + 1));

    if (processorId >= pow_k - 1) {
      if (processorId % pow_k1 == pow_k - 1) {
        // Send score to processor i + 2^k
        LOG(DEBUG) << "Sending score to processor " << processorId + pow_k
                   << "\n  " << *localScore;
        context.sendScoreTo(std::move(localScore), processorId + pow_k);
      } else if (processorId % pow_k1 == pow_k1 - 1) {
        // Receive score from processor i - 2^k
        auto receivedScore = context.receiveScore(processorId);
        LOG(DEBUG) << "Received score:\n  " << *receivedScore;

        // update local score
        std::transform(localScore->begin(), localScore->end(),
                       receivedScore->begin(), localScore->begin(),
                       [](auto f, auto s) { return f + s; });
      }
    }
#pragma omp barrier
  }

  return localScore;
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
                           int thetaStep, double lenThreshold, int threads)
    : _shapeSize(shapeSize), _minTheta(minTheta), _maxTheta(maxTheta),
      _thetaStep(thetaStep), _lenThreshold(lenThreshold), _threads{threads} {

  // SHARP parameters
  _minDist = -_shapeSize * maxSumSinCos;
  _maxDist = _shapeSize * maxSumSinCos;
  _orientations = d2i(std::floor(std::abs(_maxTheta - _minTheta) / _thetaStep));

  // Initialize Scores Vault
  using svSize = decltype(_scoresVault.size());
  _scoresVault =
      std::vector<std::pair<std::unique_ptr<SharpContext::Score>, bool>>(
          static_cast<svSize>(_threads));

  for (auto &pair : _scoresVault) {
    pair.first = std::unique_ptr<Score>(nullptr);
    pair.second = false;
  }

  // Initialize Locks
  using lSize = decltype(_locks.size());
  _locks = std::vector<aapp::OmpLock>(static_cast<lSize>(_threads));

  // Logger
  configureLogger();
}

std::pair<double, double> SharpContext::getAnglesInterval(int processorNo) {
  double thetaMin = (processorNo * _thetaStep * _orientations) / _threads;
  double thetaMax =
      ((processorNo + 1) * _thetaStep * _orientations / _threads) - 1.0;

  // LOG(DEBUG) << "Min: " << thetaMin << " Max: " << thetaMax << " ProcessorNo:
  // " << processorNo << " ThetaStep: " << _thetaStep << " Orientations: " <<
  // _orientations << " Threads: " << _threads;

  return std::pair<double, double>(thetaMin, thetaMax);
}

const std::vector<aapp::ReferenceShape> &SharpContext::referenceShapes() const {
  return _referenceShapes;
}

void SharpContext::sendScoreTo(std::unique_ptr<SharpContext::Score> score,
                               int processorId) {
  _locks[processorId].set();
  _scoresVault[processorId].first = std::move(score);
  _scoresVault[processorId].second = true;
  _locks[processorId].unset();
}

std::unique_ptr<SharpContext::Score>
SharpContext::receiveScore(int processorId) {
  while (true) {
    _locks[processorId].set();
    if (_scoresVault[processorId].second) {
      _scoresVault[processorId].second = false;
      auto ret = std::move(_scoresVault[processorId].first);
      _locks[processorId].unset();
      return ret;
    }
    _locks[processorId].unset();
  }
}
}
