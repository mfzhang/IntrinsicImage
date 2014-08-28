#ifndef INTRINSIC_H
#define INTRINSIC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "segment-image.h"

using namespace std;
using namespace cv;
typedef Mat_<Vec3b> CVImage;

image<rgb> MatToImage(const CVImage& input){
    int height = input.rows;
    int width = input.cols;
    image<rgb> output(width,height);

    for(int x = 0;x < height;x++){
        for(int y = 0;y < width;y++){
            rgb color;
            color.b = input(x,y)[0];
            color.g = input(x,y)[1];
            color.r = input(x,y)[2];  
            output.access[x][y] = color; 
        }
    } 
    return output;
}

CVImage ImageToMat(const image<rgb>& input){
    int height = input.height();
    int width = input.width();
    CVImage output(height,width, Vec3b(0,0,0));

    for(int x = 0;x < height;x++){
        for(int y = 0;y < width;y++){
            output(x,y)[0] = input.access[x][y].b;
            output(x,y)[1] = input.access[x][y].g;
            output(x,y)[2] = input.access[x][y].r;
        }
    }
    return output;
}



#endif


