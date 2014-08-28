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
    Mat_<double> output(input.rows, input.cols, 0);
    for(int i = 0;i < input.rows;i++){
        for(int j = 0;j < input.cols;j++){
            double temp = input(i,j);
            output(i,j) = temp / abs(temp) * max(abs(temp) - lambda,0);
        }
    }
    return output;
}


/*
 * Solve the l1-regularization problem using Bregman Iteration.
 * For details, please see "The Split Bregman Method for L1 Regularized Problems" 
 */
Mat_<double> L1Regularization(const Mat_<double>& pairwise_weight,
                              const Mat_<double>& reflectance,
                              const Mat_<uchar>& intensity,
                              double alpha,
                              double mu,
                              double lambda,
                              int iteration_num){
    // construct the matrix for global entropy
    Mat_<double> global_sparsity_matrix(cluster_num*(cluster_num+1)/2, cluster_num); 
    int cluster_num = pairwise_weight.rows;
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
    Mat_<double> d_1(pairwise_weight.rows,1,0);
    Mat_<double> d_2(global_sparsity_matrix.rows,1,0);
    Mat_<double> b_1(pairwise_weight.rows,1,0);
    Mat_<double> b_2(global_sparsity_matrix.rows,1,0);
   
    for(int i = 0;i < iteration_num;i++){
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



