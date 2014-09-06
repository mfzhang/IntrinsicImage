/**
 * @author Bi Sai 
 * @version 2014/08/27
 */

#include "../include/intrinsic/intrinsic.h"
#include "../include/intrinsic/optimize.h"
#include <algorithm>

int main(){
    string original_image_path = "C:\\Users\\BiSai\\Desktop\\baby.tif";    
    // parameters
    double sigma = 0; 
    int k = 100;
    int min_size = 20;
    
    // read image 
    CVImage original_image = imread(original_image_path);
    Mat_<uchar> original_image_gray = imread(original_image_path,0);  
    int image_width = original_image.cols;
    int image_height = original_image.rows; 
    image<rgb>* input = MatToImage(original_image); 
    // imshow("Original image", original_image); 
    // waitKey(0);

    // segment the image into superpixels and get clusters 
    int num_css; 
    cout<<"Segment image...\n"<<endl;
    CVImage segment_result;
    Mat_<int> pixel_label;
    vector<ReflectanceCluster> clusters = GetReflectanceCluster(input, sigma, k, min_size, &num_css, segment_result, pixel_label);
    cout<<"Number of clusters: "<<num_css<<endl;
    imshow("Segment result", segment_result);
    // waitKey(0);
    
    // Get pairwise weight of clusters
    Mat_<double> pairwise_weight = GetPairwiseWeight(clusters,original_image);

    cout<<"Initialize reflectance..."<<endl;
    // Initial reflectance to log(intensity) 
    Mat_<double> log_image(image_height,image_width,0.0);
    for(int i = 0;i < image_height;i++){
        for(int j = 0;j < image_width;j++){
            // prevent log(0)
            // log_image(i,j) = log(original_image_gray(i,j) + 1); 
			// use red channel
			log_image(i,j) = log(original_image(i,j)[0] + 1);
		}
    }

    
    // Solve reflectance
	double gamma = 1;
    double alpha = 0.01;
    double mu = 1;
    int iteration_num = 100;
    double lambda = 1;
    double beta = 1;
    double theta = 10000;
    Mat_<double> reflectance;
	Mat_<double> intensity(num_css,1);
	for(int i = 0;i < num_css;i++){
        double temp = 0;
        vector<Point2i> cluster_pixels = clusters[i].GetPixelLocations();
        for(int j = 0;j < cluster_pixels.size();j++){
            temp = temp + log_image(cluster_pixels[j].x, cluster_pixels[j].y);                
        }
        intensity(i,0) = temp / cluster_pixels.size();
    }

	// cout<<*max_element(intensity.begin(),intensity.end());
	

    reflectance = intensity.clone();
    reflectance = L1Regularization(log_image, original_image, intensity, pixel_label, clusters, gamma, alpha, mu, lambda, beta, theta, iteration_num);

	Mat_<Vec3b> reflectance_image(image_height, image_width, Vec3b(0,0,0));
	for(int i = 0; i < image_height; ++i){
		for(int j = 0;j < image_width; ++j){
			int label = pixel_label(i,j);
			int temp = exp(reflectance(label));
			if(temp > 255){
				temp = 255;
			}
			reflectance_image(i,j)[0] = (uchar)temp;
			reflectance_image(i,j)[1] = (uchar)(temp * original_image(i,j)[1] / (double)original_image(i,j)[0]);
			reflectance_image(i,j)[2] = (uchar)(temp * original_image(i,j)[2] / (double)original_image(i,j)[0]);
		}
	}
	imshow("Result", reflectance_image);
	waitKey(0);
    cout<<endl;
    return 0;
}




