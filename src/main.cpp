#include "../include/Sharp.h"

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("usage: DisplayImage.out <ImagePath>\n");
    return -1;
  }

  aapp::sharp(argv[1]);
  return 0;
}