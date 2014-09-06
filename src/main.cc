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
    int k = 300;
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

    Mat_<double> r_reflectance = GetReflectance(clusters, original_image, pixel_label, 2);
    Mat_<double> g_reflectance = GetReflectance(clusters, original_image, pixel_label, 1);
    Mat_<double> b_reflectance = GetReflectance(clusters, original_image, pixel_label, 0);
    cout<<"Initialize reflectance..."<<endl;

	
	Mat_<Vec3b> reflectance_image(image_height, image_width, Vec3b(0,0,0));

	set<int> r_different_value;
    set<int> g_different_value;
    set<int> b_different_value;

	for(int i = 0; i < image_height; ++i){
		for(int j = 0;j < image_width; ++j){
			int label = pixel_label(i,j);
			int temp = exp(r_reflectance(label));
			reflectance_image(i,j)[2] = (uchar)temp;
            r_different_value.insert(temp);

            temp = exp(g_reflectance(label));
            reflectance_image(i,j)[1] = (uchar)temp;
            g_different_value.insert(temp);
            
            temp = exp(b_reflectance(label));
            reflectance_image(i,j)[0] = (uchar)temp;
			b_different_value.insert(temp);
		}
	}

	cout<<"Number of different values of R: "<<r_different_value.size()<<endl;
    cout<<"Number of different values of G: "<<g_different_value.size()<<endl;
    cout<<"Number of different values of B: "<<b_different_value.size()<<endl;
	
	imshow("Result", reflectance_image);
	waitKey(0);
    cout<<endl;
    return 0;
}




