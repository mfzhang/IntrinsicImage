#ifndef INTRINSIC_H
#define INTRINSIC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include "../superpixel/segment-image.h"
#include <map>

using namespace std;
using namespace cv;
typedef Mat_<Vec3b> CVImage;


/*
 * Given an image in Mat_<Vec3b>, turn it into image<rgb>
 */
image<rgb>* MatToImage(const CVImage& input){
    int height = input.rows;
    int width = input.cols;
    image<rgb>* output = new image<rgb>(width,height);

    for(int x = 0;x < height;x++){
        for(int y = 0;y < width;y++){
            rgb color;
            color.b = input(x,y)[0];
            color.g = input(x,y)[1];
            color.r = input(x,y)[2];  
            output->access[x][y] = color; 
        }
    } 
    return output;
}

/*
 * Given an image stored in image<rgb>, turn it into Mat_<Vec3b>.
 */
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


/*
 * Class for representing a cluster of pixels.
 */
class ReflectanceCluster{
    public:
        ReflectanceCluster(){
            cluster_size_ = 0;
            cluster_center_ = Point2i(-1,-1); 
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

        int GetClusterSize(){
            return cluster_size_;
        }
        void AddPixel(Point2i pixel){
            pixel_locations_.push_back(pixel); 
            cluster_size_++;
        } 
        vector<Point2i> GetPixelLocations(){
            return pixel_locations_;
        }
    private:
        int cluster_size_;
        Point2i cluster_center_;
        vector<Point2i> pixel_locations_; 
};


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
 * pixel_label: the label for each pixel, pixels in the same cluster have the same label
 */
vector<ReflectanceCluster> GetReflectanceCluster(image<rgb> *im, float sigma, float c, int min_size,
        int *num_ccs, CVImage& output, Mat_<int>& pixel_label) {
    int width = im->width();
    int height = im->height();

    image<float> *r = new image<float>(width, height);
    image<float> *g = new image<float>(width, height);
    image<float> *b = new image<float>(width, height);

    pixel_label = Mat_<int>(height,width);

    // smooth each color channel  
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
			r->access[y][x] = im->access[y][x].r;
			g->access[y][x] = im->access[y][x].g;
			b->access[y][x] = im->access[y][x].b;
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
                result[temp].AddPixel(Point2i(y,x));
                pixel_label(y,x) = temp;           
            }
            else{
                index[comp] = count;
				pixel_label(y,x) = count;
                count++;
                ReflectanceCluster new_cluster;
                new_cluster.AddPixel(Point2i(y,x)); 
                result.push_back(new_cluster);
            }
        }
    }  
    
    output = CVImage(height,width, Vec3b(0,0,0));  
    rgb *colors = new rgb[width*height];
    for (int i = 0; i < width*height; i++){
        colors[i] = random_rgb();
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int comp = u->find(y * width + x);
            // imRef(output, x, y) = colors[comp];
            output(y,x)[0] = colors[comp].b;
            output(y,x)[1] = colors[comp].g;
            output(y,x)[2] = colors[comp].r;
        }
    }  


    delete u;

    return result;
}


/*
 * Calculate the pairwise weight between ReflectanceCluster. 
 * For details, see "Intrinsic Images in the Wild", SIGGRAPH 2014
 */
Mat_<double> GetPairwiseWeight(vector<ReflectanceCluster>& clusters,
                               const CVImage& image){ 
    cout<<"Get pairwise weight..."<<endl;
    int cluster_num = clusters.size();
    int image_width = image.cols;
    int image_height = image.rows;
    Mat_<double> weight(cluster_num,cluster_num,0.0);
    // calculate the feature for each cluster 
    // feature: x, y, intensity, red-chromaticity, green-chromaticity 
    Mat_<double> feature(cluster_num,5,0.0);    
    double theta_p = 0.1;
    double theta_l = 0.1;
    double theta_c = 0.025;
    double image_diameter = sqrt(image_width * image_width + image_height * image_height); 

    for(int i = 0;i < cluster_num;i++){
        Point2i center = clusters[i].GetClusterCenter();
        feature(i,0) = center.x / (image_diameter * theta_p);
        feature(i,1) = center.y / (image_diameter * theta_p);
        double b = 0;
        double g = 0;
        double r = 0;
        vector<Point2i> pixel_locations = clusters[i].GetPixelLocations();
        for(int j = 0;j < pixel_locations.size();j++){
            int x = pixel_locations[j].x;
            int y = pixel_locations[j].y;
            b += image(x,y)[0];
            g += image(x,y)[1];
            r += image(x,y)[2];
        }
        b = b / pixel_locations.size();
        g = g / pixel_locations.size();
        r = r / pixel_locations.size();
        feature(i,2) = (b + g + r) / (3 * theta_l);
        feature(i,3) = r / (theta_c * (b + g + r));
        feature(i,4) = g / (theta_c * (b + g + r));
    }
    
    // calculate the weight
    for(int i = 0;i < cluster_num;i++){
        for(int j = i+1;j < cluster_num;j++){
            Mat_<double> row_1 = feature.row(i);
            Mat_<double> row_2 = feature.row(j);
            double w = exp( -0.5 * pow(norm(row_1 - row_2), 2.0));         
            weight(i,j) = w;
            weight(j,i) = w; 
        }
    }
    return weight;
} 



#endif


