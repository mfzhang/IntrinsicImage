/**
 * @author Bi Sai 
 * @version 2014/08/27
 */

#include "../include/intrinsic/intrinsic.h"
#include "../include/intrinsic/optimize.h"

int main(){
    string original_image_path = "C:\\Users\\BiSai\\Desktop\\baby.tif";    
    // parameters
    double sigma = 0; 
    int k = 300;
    int min_size = 20;
    
    // read image 
    CVImage original_image = imread(original_image_path);
    Mat_<uchar> original_image_gray = imread(original_image_path,0);  
    int image_width = original_image.cols;
    int image_height = original_image.rows; 
    image<rgb>* input = MatToImage(original_image); 
    imshow("Original image", original_image); 
    //waitKey(0);
    // segment the image into superpixels and get clusters 
    int num_css; 
    printf("Segment image...\n");
    CVImage segment_result;
    vector<ReflectanceCluster> clusters = GetReflectanceCluster(input, sigma, k, min_size, &num_css, segment_result);
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
            log_image(i,j) = log(original_image_gray(i,j) + 1); 
        }
    }
     
    // Solve reflectance
    int alpha = 1;
    int mu = 1;
    int iteration_num = 10;
    int lambda = 2 * mu;
    Mat_<double> reflectance = log_image.clone();
	Mat_<double> intensity(num_css,1);
	
	for(int i = 0;i < num_css;i++){
		Point2i center = clusters[i].GetClusterCenter();
		intensity(i,0) = log_image(center.x,center.y);
	}
	
    reflectance = L1Regularization(pairwise_weight, reflectance, intensity, alpha, mu, lambda, iteration_num);

    for(int i = 0;i < 100;i++){
        cout<<reflectance(0,i)<<" ";
    } 
	waitKey(0);
    cout<<endl;
    

    // Solve shading
    // Mat_<double> raio = ShadingSmooth(reflectance, log_image);
    return 0;
}




