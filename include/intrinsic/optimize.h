#ifndef OPTIMIZE_H
#define OPTIMIZE_H
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstdio>
using namespace std;
using namespace cv;

/*
 * Shrinkage process
 */
Mat_<double> Shrink(const Mat_<double>& input, double lambda){
    Mat_<double> output(input.rows, input.cols, 0.0);
    for(int i = 0;i < input.rows;i++){
        for(int j = 0;j < input.cols;j++){
            double temp = input(i,j);
            output(i,j) = temp / abs(temp) * max(abs(temp) - lambda,0.0);
        }
    }
    return output;
}


/*
 * Solve the l1-regularization problem using Bregman Iteration.
 * Return the reflectance value.
 * For details, please see "The Split Bregman Method for L1 Regularized Problems", 
 *
 * pairwise_weight: weight between each pair of ReflectanceCluster
 * intensity: the intensity of center of each ReflectanceCluster
 * reflectance: initial reflectance. 
 * alpha: weight for global sparsity of reflectance
 * mu: weight for difference between reflectance and intensity
 * lambda: weight for l1-regularization
 * iteration_num: number of iterations for l1-regularization 
 */
Mat_<double> L1Regularization(const Mat_<double>& pairwise_weight,
                              const Mat_<double>& reflectance,
                              const Mat_<double>& intensity,
                              double alpha,
                              double mu,
                              double lambda,
                              int iteration_num){
    // construct the matrix for global entropy
    cout<<"Solve reflectance..."<<endl;
    int cluster_num = pairwise_weight.rows;
    cout<<cluster_num<<endl;
    Mat_<double> global_sparsity_matrix(cluster_num*(cluster_num+1)/2, cluster_num); 
    cout<<"Here OK 1"<<endl;
	int count = 0;
    for(int i = 0;i < cluster_num;i++){
        for(int j = i+1;j < cluster_num;j++){
            global_sparsity_matrix(count,i) = alpha;
            global_sparsity_matrix(count,j) = -alpha; 
            count++;
        }
    }
    Mat_<double> identity_matrix = Mat::eye(cluster_num,cluster_num, CV_64FC1); 
    Mat_<double> left_hand = mu * identity_matrix + lambda * pairwise_weight.t() * pairwise_weight
                                + lambda * global_sparsity_matrix.t() * global_sparsity_matrix;
    
    Mat_<double> curr_reflectance = intensity;
    Mat_<double> d_1(pairwise_weight.rows,1,0.0);
    Mat_<double> d_2(global_sparsity_matrix.rows,1,0.0);
    Mat_<double> b_1(pairwise_weight.rows,1,0.0);
    Mat_<double> b_2(global_sparsity_matrix.rows,1,0.0);
   
    for(int i = 0;i < iteration_num;i++){
        cout<<"Iter: "<<i<<endl;
        // solve for new reflectance
	
        Mat_<double> right_hand = mu * intensity + lambda * pairwise_weight.t() * (d_1 - b_1) + 
                                lambda * global_sparsity_matrix.t() * (d_2 - b_2); 
        solve(left_hand,right_hand,curr_reflectance);
        
        Mat_<double> temp_1 = pairwise_weight * curr_reflectance;
        Mat_<double> temp_2 = global_sparsity_matrix * curr_reflectance;
        // update d_1 and d_2
        d_1 = Shrink(temp_1 + b_1, 1.0 / lambda);
        d_2 = Shrink(temp_2 + b_2, 1.0 / lambda); 
        
        // update b_1 and b_2
        b_1 = b_1 + temp_1 - d_1;
        b_2 = b_2 + temp_2 - d_2;
    } 
    
    return curr_reflectance;
}

/*
 * Given an image, and its reflectance, enforce shading smooth.
 * This can be solved by "Iteratively Reweighted Least Square"
 */
Mat_<double> ShadingSmooth(const Mat_<double>& reflectance,
                           const Mat_<double>& log_image){
    int image_width = log_image.cols;
    int image_height = log_image.rows;
    Mat_<double> new_ratio(image_height,image_width,1.0);
    Mat_<double> old_ratio(image_height,image_width,0.0);
    int lambda = 10;
    double step_size = 0.01; // initial step size for gradient descent
    double precision = 1.0e-7;
    
    while(true){
        pow(new_ratio - old_ratio, 2.0, old_ratio);
        if(sum(old_ratio)[0] < precision){
            break;
        } 
        old_ratio = new_ratio.clone();
        // get laplacian result on current shading 
        Mat_<double> shading = log_image - reflectance.mul(new_ratio);  
        Laplacian(shading,shading,1);
        double penalty_part = sum(old_ratio)[0] - image_height * image_width;
        // center pixels
        for(int i = 1;i < image_height-1;i++){
            for(int j = 1;j < image_width - 1;j++){
                new_ratio(i,j) = new_ratio(i,j) - step_size * (8 * shading(i,j) * reflectance(i,j)
                    - 2 * reflectance(i,j) * shading(i,j-1) 
                    - 2 * reflectance(i,j) * shading(i,j+1)
                    - 2 * reflectance(i,j) * shading(i-1,j)
                    - 2 * reflectance(i,j) * shading(i+1,j)
                    + 2 * penalty_part); 
            }
        }
        // for first row pixels 
        for(int i = 0;i < 1;i++){
            for(int j = 1;j < image_width - 1;j++){
                new_ratio(i,j) = new_ratio(i,j) - step_size * (6 * shading(i,j) * reflectance(i,j)
                    - 2 * reflectance(i,j) * shading(i,j-1) 
                    - 2 * reflectance(i,j) * shading(i,j+1)
                    - 2 * reflectance(i,j) * shading(i+1,j)
                    + 2 * penalty_part); 
            }
        }
        // for last row pixels
        for(int i = image_width-1;i < image_width;i++){
            for(int j = 1;j < image_width - 1;j++){
                new_ratio(i,j) = new_ratio(i,j) - step_size * (6 * shading(i,j) * reflectance(i,j)
                    - 2 * reflectance(i,j) * shading(i,j-1) 
                    - 2 * reflectance(i,j) * shading(i,j+1)
                    - 2 * reflectance(i,j) * shading(i-1,j)
                    + 2 * penalty_part); 
            }
        }
        // for first column pixels
        for(int i = 1;i < image_height-1;i++){
            for(int j = 0;j < 1;j++){
                new_ratio(i,j) = new_ratio(i,j) - step_size * (6 * shading(i,j) * reflectance(i,j)
                    - 2 * reflectance(i,j) * shading(i,j+1)
                    - 2 * reflectance(i,j) * shading(i-1,j)
                    - 2 * reflectance(i,j) * shading(i+1,j)
                    + 2 * penalty_part); 
            }
        }
        // for last column pixels
        for(int i = 1;i < image_height-1;i++){
            for(int j = image_width - 1;j < image_width;j++){
                new_ratio(i,j) = new_ratio(i,j) - step_size * (6 * shading(i,j) * reflectance(i,j)
                        - 2 * reflectance(i,j) * shading(i,j-1)
                        - 2 * reflectance(i,j) * shading(i-1,j)
                        - 2 * reflectance(i,j) * shading(i+1,j)
                        + 2 * penalty_part); 
            }
        }
        // corner pixels
        int i = 0; int j = 0; 
        new_ratio(i,j) = new_ratio(i,j) - step_size * (4 * shading(i,j) * reflectance(i,j)
                - 2 * reflectance(i,j) * shading(i,j+1)
                - 2 * reflectance(i,j) * shading(i+1,j)
                + 2 * penalty_part); 
        i = image_height - 1; j = 0;
        new_ratio(i,j) = new_ratio(i,j) - step_size * (4 * shading(i,j) * reflectance(i,j)
                - 2 * reflectance(i,j) * shading(i-1,j)
                - 2 * reflectance(i,j) * shading(i,j+1)
                + 2 * penalty_part);
        i = image_height - 1; j = image_width-1;
        new_ratio(i,j) = new_ratio(i,j) - step_size * (4 * shading(i,j) * reflectance(i,j)
                - 2 * reflectance(i,j) * shading(i-1,j)
                - 2 * reflectance(i,j) * shading(i,j-1)
                + 2 * penalty_part);
        i = 0; j = image_width-1;
        new_ratio(i,j) = new_ratio(i,j) - step_size * (4 * shading(i,j) * reflectance(i,j)
                - 2 * reflectance(i,j) * shading(i,j-1)
                - 2 * reflectance(i,j) * shading(i+1,j)
                + 2 * penalty_part);
    
        // calculate objective function
        Mat_<double> new_shading = log_image - reflectance.mul(new_ratio);
        Laplacian(new_shading,new_shading,1);
        pow(new_shading,2.0,new_shading);
        double new_objective_value = sum(new_shading)[0] + lambda * 
                pow((sum(new_ratio)[0] - image_width * image_height),2.0);
        pow(shading,2.0,shading);
        double objective_value = sum(shading)[0] + lambda * pow(penalty_part,2.0);
        
        if(new_objective_value < objective_value){
            step_size = 2 * step_size; 
        } 
        else{
            new_ratio = old_ratio;
            step_size = step_size / 2.0;
        }
    }
    return new_ratio;
} 


#endif


