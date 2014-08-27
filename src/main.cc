/**
 * @author 
 * @version 2014/08/27
 */

#include "intrinsic.h"

int main(){
    string path = "./../res/MIT/MIT-Berkeley-Laboratory/";    
    string original_image_path = path + "raccoon/original.png";
    string mask_path = path + "raccoon/mask.png"; 

    // parameters
    double sigma = 0.5; 
    int k = 3000;
    int min_size = 20;

    Mat_<Vec3b> original_image = imread(original_image_path);
    Mat_<uchar> mask_image = imread(mask_path,0);
    imshow("image", original_image);
    waitKey(0);
    
    /* 
    int num_css; 
    printf("Segment image...\n");
    image<rgb> *output = segment_image(input, sigma, k, min_size, &num_css);
    
    savePPM(output, "result.ppm");
    printf("got %d components\n", num_css);
    printf("done! uff...thats hard work.\n");
    */
    return 0;
}




