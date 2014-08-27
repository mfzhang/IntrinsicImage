#ifndef INTRINSIC_H
#define INTRINSIC_H

#include <iostream>
#include <vector>
#include <cmath>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cstdio>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"

typedef Mat_<Vec3b> CVImage;

using namespace std;
using namespace cv;

image<rgb> ImageToMat(const CVImage& input);
CVImage MatToImage(const image<rgb>& input);


#endif


