#include "../include/Sharp.h"
#include "../include/SharpSupport.h"
#include "../test/SharpTest.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// This source file builds a command line options parsers using
// boost::program_options. Then the main function calls the right routines
// according to those options and parameters.

std::tuple<po::variables_map, po::options_description>
buildParser(int argc, char **argv) {
  auto run_desc = "Run SHARP algorithm";
  auto run = po::options_description(run_desc);

  auto refShape_desc = "Path of reference images (Default: working directory)";
  auto testShape_desc = "Test shape image file";
  auto shapeSize_desc = "Test shape width in pixels";
  auto minTheta_desc = "Minimum angle to consider for shape rotation in degrees";
  auto maxTheta_desc = "Maximum angle to consider for shape rotation in degrees";
  auto thetaStep_desc = "Angular distance between two consecutive angles";
  auto lenThresh_desc = "Minimum accepted length of a shape-tangent segment";
  auto numThreads_desc = "Number of threads to run (Default: max available)";

  run.add_options()
      ("shape-size", po::value<int>()->default_value(256), shapeSize_desc)
      ("min-theta", po::value<double>()->default_value(0.0), minTheta_desc)
      ("max-theta", po::value<double>()->default_value(180.0), maxTheta_desc)
      ("theta-step", po::value<int>()->default_value(5), thetaStep_desc)
      ("length-thresh", po::value<double>()->default_value(2.0), lenThresh_desc)
      ("threads", po::value<int>(), numThreads_desc)
      ("reference-shapes,r", po::value<std::string>(), refShape_desc)
      ("test-shape,t", po::value<std::string>(), testShape_desc)
      ;

  auto demo_desc = "Demo options";
  auto demo = po::options_description(demo_desc);

  auto help_desc = "Print this help message";
  auto detectEdges_desc = "Run and display Edge Detection for input image";

  demo.add_options()
      ("help,h", help_desc)
      ("detect-edges,e", detectEdges_desc)
      ("test-sequential", "Simulate parallel SHARP sequentially")
      ;

  auto testShape_pos = po::positional_options_description();
  testShape_pos.add("test-shape", 1);

  auto generic_desc = "SHARP Algorithm driver. A set of demo facilities for SHARP";
  auto generic = po::options_description(generic_desc);
  generic.add(run).add(demo);

  auto vm = po::variables_map();
  po::store(po::command_line_parser(argc, argv)
                .options(generic)
                .positional(testShape_pos)
                .run(),
            vm);
  po::notify(vm);
  return std::make_tuple(vm, generic);
}

int main(int argc, char **argv) {

  auto parserPack = buildParser(argc, argv);
  auto vm = std::get<0>(parserPack);
  auto opt = std::get<1>(parserPack);

  if (vm.count("help")) {
    std::cout <<  opt << "\n";
    return 0;
  }

  if (vm.count("detect-edges")) {
    if (vm.count("test-shape")) {
      auto testShape = vm["test-shape"].as<std::string>();
      aapp::edgeDetectionOption(testShape);
      return 0;
    } else {
      std::cout << "Missing test-shape argument\nUsage: sharp_driver -e "
          "TEST_SHAPE\n";
      return 0;
    }
  }

  if (vm.count("reference-shapes")) {
    auto refShapes = vm["reference-shapes"].as<std::string>();
    auto shapeSize = vm["shape-size"].as<int>();
    auto minTheta = vm["min-theta"].as<double>();
    auto maxTheta = vm["max-theta"].as<double>();
    auto thetaStep = vm["theta-step"].as<int>();
    auto lenThresh = vm["length-thresh"].as<double>();

    auto testShape = std::string{};
    int threads = omp_get_max_threads();

    if (vm.count("test-shape")) {
      testShape = vm["test-shape"].as<std::string>();
    } else {
      std::cout << "Missing test-shape argument\nUsage: " << argv[0] << " -r "
                   "REFERENCE_FOLDER TEST_SHAPE\n";
      return 0;
    }

    if (vm.count("threads")) {
      threads = vm["threads"].as<int>();
    }

    if (vm.count("test-sequential")) {
      sharpSequential(testShape,
                      refShapes,
                      shapeSize,
                      minTheta,
                      maxTheta,
                      thetaStep,
                      lenThresh,
                      threads);
      return 0;
    }

    aapp::sharp(testShape,
                refShapes,
                shapeSize,
                minTheta,
                maxTheta,
                thetaStep,
                lenThresh,
                threads);
    return 0;
  }

  std::cout << "Usage: " << argv[0] << " [options]\n";
  std::cout << opt << "\n";
  return 0;
}