#ifndef INTRINSIC_H
#define INTRINSIC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "segment-image.h"
#incldue <map>

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

class ReflectanceCluster{
    public:
        ReflectanceCluster(){
            cluster_size = 0;
            cluster_center = Point2i(-1,-1); 
        }
        Point2i GetClusterCenter(){
            if(cluster_center_.x >= 0 && cluster_center_.y >= 0){
                return cluster_center_; 
            } 
            double x = 0;
            double y = 0;
            for(int i = 0;i < pixel_locations_.size();i++){
                x += pixel_locations_[i].x;
                y += pixel_locations_[i].y;
            } 
            assert(pixel_locations_.size()!=0);
            cluster_center_.x = (int)(x / pixel_locations_.size());
            cluster_center_.y = (int)(y / pixel_locations_.size());
            return cluster_center_;
        }

        void GetClusterSize(){
            return cluster_size;
        }
        void AddPixel(Point2i pixel){
            pixel_locations.push_back(pixel); 
            cluster_size++;
        } 
    private:
        int cluster_size_;
        Point2i cluster_center_;
        vector<Point2i> pixel_locations_; 
}


/*
 * Segment an image
 *
 * Given an image, return its cluster result.  
 * 
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
vector<ReflectanceCluster> GetReflectanceCluster(image<rgb> *im, float sigma, float c, int min_size,
        int *num_ccs) {
    int width = im->width();
    int height = im->height();

    image<float> *r = new image<float>(width, height);
    image<float> *g = new image<float>(width, height);
    image<float> *b = new image<float>(width, height);

    // smooth each color channel  
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(r, x, y) = imRef(im, x, y).r;
            imRef(g, x, y) = imRef(im, x, y).g;
            imRef(b, x, y) = imRef(im, x, y).b;
        }
    }
    image<float> *smooth_r = smooth(r, sigma);
    image<float> *smooth_g = smooth(g, sigma);
    image<float> *smooth_b = smooth(b, sigma);
    delete r;
    delete g;
    delete b;

    // build graph
    edge *edges = new edge[width*height*4];
    int num = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width-1) {
                edges[num].a = y * width + x;
                edges[num].b = y * width + (x+1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
                num++;
            }

            if (y < height-1) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + x;
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
                num++;
            }

            if ((x < width-1) && (y < height-1)) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + (x+1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
                num++;
            }

            if ((x < width-1) && (y > 0)) {
                edges[num].a = y * width + x;
                edges[num].b = (y-1) * width + (x+1);
                edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
                num++;
            }
        }
    }
    delete smooth_r;
    delete smooth_g;
    delete smooth_b;

    // segment
    universe *u = segment_graph(width*height, num, edges, c);

    // post process small components
    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }
    delete [] edges;
    *num_ccs = u->num_sets();
    

    // get the pixels in each cluster 
    map<int,int> index;
    vector<ReflectanceCluster> result; 
    int count = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int comp = u->find(y * width + x);
            if(index.count(comp) > 0){
                int temp = index[comp];
                result[temp].AddPixel(Point2i(x,y));           
            }
            else{
                index[comp] = count;
                count++;
                ReflectanceCluster new_cluster;
                new_cluster.AddPixel(Point2i(x,y)); 
                result.push_back(new_cluster);
            }
        }
    }  

    delete u;

    return result;
}


#endif


