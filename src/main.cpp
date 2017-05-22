#include "../include/Sharp.h"
#include "../include/Sharp_support.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

std::tuple<po::variables_map, po::options_description>
buildParser(int argc, char **argv) {
  auto generic_desc = "SHARP Algorithm driver. A set of demo facilities for SHARP";
  auto generic = po::options_description(generic_desc);

  auto help_desc = "Print this help message";
  auto refShape_desc = "Path of reference images (Default: working directory)";
  auto testShape_desc = "Test shape image file";
  auto detectEdges_desc = "Run and display Edge Detection for input image";
  generic.add_options()
      ("help,h", help_desc)
      ("reference-shapes,r", po::value<std::string>(), refShape_desc)
      ("test-shape,t", po::value<std::string>(), testShape_desc)
      ("detect-edges,e", detectEdges_desc)
      ;

  auto testShape_pos = po::positional_options_description();
  testShape_pos.add("test-shape", 1);

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
    if (vm.count("test-shape")) {
      auto refShapes = vm["reference-shapes"].as<std::string>();
      auto testShape = vm["test-shape"].as<std::string>();
      aapp::sharp(testShape, refShapes);
      return 0;
    } else {
      std::cout << "Missing test-shape argument\nUsage: sharp_driver -r "
                   "REFERENCE_FOLDER TEST_SHAPE\n";
      return 0;
    }
  }

  std::cout << "Usage: sharp_driver [options]\n";
  std::cout << opt << "\n";
  return 0;
}