/**
 * @author Bi Sai 
 * @version 2014/08/27
 */

#include "intrinsic.h"
#include "optimize.h"

int main(){
    string path = "./../res/MIT/MIT-Berkeley-Laboratory/";    
    string original_image_path = path + "raccoon/original.png";
    string mask_path = path + "raccoon/mask.png"; 
    // parameters
    double sigma = 0; 
    int k = 30;
    int min_size = 1;
    
    // read image 
    CVImage original_image = imread(original_image_path);
    Mat_<uchar> original_image_gray = imread(original_image_path,0);  
    int image_width = original_image.cols;
    int image_height = original_image.rows; 
    image<rgb> input = MatToImage(original_image); 
    imshow("Original image", original_image); 
    
    // segment the image into superpixels and get clusters 
    int num_css; 
    printf("Segment image...\n");
    vector<ReflectanceCluster> clusters = GetReflectanceCluster(&input, sigma, k, min_size, &num_css);
    waitKey(0); 
    
    // Get pairwise weight of clusters
    Mat_<double> pairwise_weight = GetPairwiseWeight(clusters,original_image);

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
    reflectance = L1Regularization(pairwise_weight, reflectance, log_image, alpha, mu, lambda, iteration_num);
    
    // Solve shading
    Mat_<double> ratio = ShadingSmooth(reflectance, log_image);
    return 0;
}




