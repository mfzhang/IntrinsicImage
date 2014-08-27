/**
 * @author Bi Sai 
 * @version 2014/08/27
 */

#include "./intrinsic/intrinsic.h"

int main(){
    string path = "./../res/MIT/MIT-Berkeley-Laboratory/";    
    string original_image_path = path + "raccoon/original.png";
    string mask_path = path + "raccoon/mask.png"; 
    // parameters
    double sigma = 0; 
    int k = 30;
    int min_size = 1;
    /*
    CVImage original_image = imread(original_image_path);
    Mat_<uchar> mask_image = imread(mask_path,0);
    image<rgb> input = MatToImage(original_image); 
    imshow("Original image", original_image); 
    int num_css; 
    printf("Segment image...\n");
    image<rgb> *output = segment_image(&input, sigma, k, min_size, &num_css);
    CVImage segment_result = ImageToMat(*output);        
    imshow("Segment result", segment_result);
    printf("got %d components\n", num_css);
    printf("done! uff...thats hard work.\n");
    waitKey(0); 
    */
    return 0;
}




