#ifndef OPTIMIZE_H
#define OPTIMIZE_H
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "./intrinsic.h"
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
            // output(i,j) = temp / abs(temp) * max(abs(temp) - lambda,0.0);
            if(temp > lambda){
                output(i,j) = temp - lambda;
            }
            else if(temp < -lambda){
                output(i,j) = temp + lambda;
            }
        }
    }
    return output;
}


/*
 * Solve the l1-regularization problem using Bregman Iteration.
 * Return the reflectance value.
 * For details, please see "The Split Bregman Method for L1 Regularized Problems"
 * The Engergy function is: |AR| + alpha/2 * |BR| + mu/2 * |I-R|^2 + beta/2 * (CR+D)^2 + theta/2 * (ER - 0.5)^2 
 * 
 * log_image: image in log domain
 * original_image: image in Mat_<Vec3b> format
 * I: average intensity of each clusters
 * pixel_label: the label of each pixel, pixels in the same cluster have the same label
 * clusters: cluster result for image
 */
Mat_<double> L1Regularization(const Mat_<double>& log_image,
                              const CVImage& original_image,
                              const Mat_<double>& I,
                              const Mat_<int>& pixel_label,
                              const vector<ReflectanceCluster>& clusters,
                              double alpha,
                              double mu,
                              double lambda,
                              double beta,
                              double theta,
                              int iteration_num){
    int image_width = log_image.cols;
    int image_height = log_image.rows;

    // calculate the weight between each pair of clusters
    Mat_<double> A = GetPairwiseWeight(clusters,original_image);
    
    // construct the matrix for global entropy
    cout<<"Solve reflectance..."<<endl;
    int cluster_num = pairwise_weight.rows;
    Mat_<double> B(cluster_num*(cluster_num+1)/2, cluster_num); 
    int count = 0;
    for(int i = 0;i < cluster_num;i++){
        for(int j = i+1;j < cluster_num;j++){
            B(count,i) = alpha;
            B(count,j) = -alpha; 
            count++;
        }
    }

    // shading smooth part
    vector<Point2i> pixel_pairs_1;
    vector<Point2i> pixel_pairs_2;
    for(int i = 0;i < image_height-1;i++){
        for(int j = 0;j < image_width-1;j++){
            if(pixel_label(i,j) != pixel_label(i,j+1)){
                pixel_pairs_1.push_back(Point2i(i,j));
                pixel_pairs_2.push_back(Point2i(i,j+1));    
            }    
            if(pixel_label(i,j) != pixel_label(i+1,j)){
                pixel_pairs_1.push_back(Point2i(i,j));
                pixel_pairs_2.push_back(Point2i(i+1,j));
            }
        }
    }

    int pair_num = pixel_pairs_1.size();    
    Mat_<double> D(pair_num,cluster_num,0.0);
    Mat_<double> C(pair_num,1,0.0);

    for(int i = 0;i < pair_num;i++){
        int x_1 = pixel_pairs_1[i].x;
        int y_1 = pixel_pairs_1[i].y;
        int x_2 = pixel_pairs_2[i].x;
        int y_2 = pixel_pairs_2[i].y;
        int label_1 = pixel_label(x_1,y_1);
        int label_2 = pixel_label(x_2,y_2);
        D(i,label_1) = -1;
        D(i,label_2) = 1;
        C(i,0) = log_image(x_1,y_1) - log_image(x_2,y_2); 
    }

    // average reflectance is set to 0.5
    double average_reflectance = 0.5;
    Mat_<double> E(1,cluster_num);
    double cluster_total_size = 0;
    for(int i = 0;i < clusters.size();i++){
        int temp = clusters[i].GetClusterSize();
        E(0,i) = temp; 
        cluster_total_size += temp;
    }
    E = 1.0 / cluster_total_size * E;





    Mat_<double> identity_matrix = Mat::eye(cluster_num,cluster_num, CV_64FC1); 
    Mat_<double> left_hand = mu * identity_matrix + lambda * (A.t() * A)
                                + lambda * (B.t() * B) + beta * D.t() * D + theta * (E.t() * E);

    Mat_<double> curr_reflectance = I.clone();
    Mat_<double> d_1(A.rows,1,0.0);
    Mat_<double> d_2(B.rows,1,0.0);
    Mat_<double> b_1(A.rows,1,0.0);
    Mat_<double> b_2(B.rows,1,0.0);
   
    for(int i = 0;i < iteration_num;i++){
        cout<<"Iter: "<<i<<endl;
        // solve for new reflectance
    
        Mat_<double> right_hand = mu * I + lambda * A.t() * (d_1 - b_1) + 
                                lambda * B.t() * (d_2 - b_2) - beta * D.t() * C + average_reflectance * theta * E.t(); 
        solve(left_hand,right_hand,curr_reflectance);

        // cout<<curr_reflectance<<endl;


        
        Mat_<double> temp_1 = pairwise_weight * curr_reflectance;
        Mat_<double> temp_2 = global_sparsity_matrix * curr_reflectance;
        // update d_1 and d_2
        d_1 = Shrink(temp_1 + b_1, 1.0 / lambda);
        d_2 = Shrink(temp_2 + b_2, 1.0 / lambda); 
        
        // update b_1 and b_2
        b_1 = b_1 + temp_1 - d_1;
        b_2 = b_2 + temp_2 - d_2;

        // calculate current objective function value and output
        double part_1 = sum(abs(temp_1))[0];
        double part_2 = sum(abs(temp_2))[0];
        double part_3 = lambda * pow(norm(curr_reflectance - intensity),2.0);
        double obj_value = part_1 + part_2 + part_3;
        cout<<obj_value<<" "<<part_1<<" "<<part_2<<" "<<part_3<<endl;
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


