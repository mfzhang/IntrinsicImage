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

Mat_<double> MatrixMulti(const vector<Mat_<double> >& matrix_columns, const Mat_<double>& A){
    Mat_<double> result(matrix_columns.size(), 1, 0.0);
    for(int i = 0; i < matrix_columns.size(); ++i){
        result(i) = A.dot(matrix_columns[i]);
    }
    return result;
}

Mat_<double> MatrixMultiRows(const vector<Mat_<double> >& matrix_columns, const Mat_<double>& A){
    int row_num = matrix_columns[0].rows;
    Mat_<double> result(row_num, 1, 0.0);
    for(int i = 0;i < row_num; ++i){
        double temp = 0;
        for(int j = 0;j < A.rows; ++j){
            temp = temp + matrix_columns[j](i) * A(j);
        }
        result(i) = temp;
    }
    return result;
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
                              vector<ReflectanceCluster>& clusters,
							  double gamma,
                              double alpha,
                              double mu,
                              double lambda,
                              double beta,
                              double theta,
                              int iteration_num){
    int image_width = log_image.cols;
    int image_height = log_image.rows;
	int cluster_num = clusters.size();
	int b_row_num = (cluster_num*(cluster_num+1))/2;

    // calculate the weight between each pair of clusters
    Mat_<double> pairwise_weight = GetPairwiseWeight(clusters,original_image);

    cout<<"Solve reflectance..."<<endl;	
	/*
	Mat_<double> B(b_row_num, cluster_num); 
    int count = 0;
    for(int i = 0;i < cluster_num;i++){
        for(int j = i+1;j < cluster_num;j++){
            B(count,i) = alpha / 2;
            B(count,j) = -alpha / 2; 
			A(count,i) = pairwise_weight(i,j) * gamma;
			A(count,j) = -pairwise_weight(i,j) * gamma;
			count++;
        }
    }

	vector<Mat_<double> > a(2, Mat_<double>(1,2));
	a[0](0,1) = 2;
    cout<<a[0]<<endl;
    cout<<a[1]<<endl;
    */

	vector<Mat_<double> > b_columns(cluster_num);
	vector<Mat_<double> > a_columns(cluster_num);
    
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

	cout<<"pixel pair: "<<pixel_pairs_1.size()<<endl;
    int pair_num = pixel_pairs_1.size();    
    Mat_<double> C(pair_num,1,0.0);
    vector<Mat_<double> > d_columns(cluster_num);

    for(int i = 0; i < cluster_num; ++i){
        a_columns[i] = Mat_<double>(b_row_num, 1, 0.0);
        b_columns[i] = Mat_<double>(b_row_num, 1, 0.0);
        d_columns[i] = Mat_<double>(pair_num, 1, 0.0);
    }

    int count = 0;
    for(int i = 0;i < cluster_num;i++){
        for(int j = i+1;j < cluster_num;j++){
            b_columns[i](count,0) = alpha / 2.0;
            b_columns[j](count,0) = - alpha / 2.0;
            a_columns[i](count,0) = pairwise_weight(i,j) * gamma / 2.0;
            a_columns[j](count,0) = - pairwise_weight(i,j) * gamma / 2.0;
            count++;
        }
    }

    for(int i = 0;i < pair_num;i++){
        int x_1 = pixel_pairs_1[i].x;
        int y_1 = pixel_pairs_1[i].y;
        int x_2 = pixel_pairs_2[i].x;
        int y_2 = pixel_pairs_2[i].y;
        int label_1 = pixel_label(x_1,y_1);
        int label_2 = pixel_label(x_2,y_2);
        d_columns[label_1](i) = -beta / 2.0;
        d_columns[label_2](i) = beta / 2.0;
        C(i,0) = beta / 2.0 * (log_image(x_1,y_1) - log_image(x_2,y_2)); 
    }

    Mat_<double> BTB(cluster_num, cluster_num, 0.0);
    Mat_<double> ATA(cluster_num, cluster_num, 0.0);
    Mat_<double> DTD(cluster_num, cluster_num, 0.0);
    for(int i = 0; i < cluster_num; ++i){
        for(int j = 0; j < cluster_num; ++j){
            BTB(i,j) = b_columns[i].dot(b_columns[j]);
            ATA(i,j) = a_columns[i].dot(a_columns[j]);
            DTD(i,j) = d_columns[i].dot(d_columns[j]);
        }
    }


    // average reflectance is set to 0.5
	// double average_reflectance = sum(I)[0] / (double)cluster_num;
    double average_reflectance = 7;
	Mat_<double> E(1,cluster_num);
    double cluster_total_size = 0;
	for(int i = 0;i < cluster_num;i++){
        int temp = clusters[i].GetClusterSize();
        E(0,i) = temp; 
        cluster_total_size += temp;
    }
    E = (1.0 / cluster_total_size) * E;

    Mat_<double> init = theta * (E.t() * E);
	// Mat_<double> left_hand = lambda * (A.t() * A) + lambda * (B.t() * B) + lambda *  (D.t() * D) + init;
    Mat_<double> left_hand = lambda * ATA + lambda * BTB + lambda * DTD + init;
    Mat_<double> init_2 = average_reflectance * theta * E.t();


    Mat_<double> curr_reflectance;
	solve(init, init_2, curr_reflectance);
    Mat_<double> d_1(b_row_num,1,0.0);
	Mat_<double> d_2(b_row_num,1,0.0);
    Mat_<double> b_1(b_row_num,1,0.0);
    Mat_<double> b_2(b_row_num,1,0.0);
    Mat_<double> d_3(pair_num, 1, 0.0);
    Mat_<double> b_3(pair_num, 1, 0.0);
   
    for(int i = 0;i < iteration_num;i++){
        cout<<"Iter: "<<i<<endl;

        // solve for new reflectance
		Mat_<double> temp_1;
		Mat_<double> temp_2;
        Mat_<double> temp_3;

        for(int j = 0;j < 1; j++){
            // Mat_<double> right_hand = lambda * A.t() * (d_1 - b_1) + lambda * B.t() * (d_2 - b_2) - beta * D.t() * C + average_reflectance * theta * E.t(); 
            Mat_<double> t_1 = MatrixMulti(a_columns, d_1 - b_1);
            Mat_<double> t_2 = MatrixMulti(b_columns, d_2 - b_2);
            Mat_<double> t_3 = MatrixMulti(d_columns, d_3 - b_3 - C);
            // Mat_<double> right_hand = lambda * A.t() * (d_1 - b_1) + lambda * B.t() * (d_2 - b_2) + average_reflectance * theta * E.t() + lambda * D.t() * (d_3 - b_3 - C); 
            Mat_<double> right_hand = lambda * (t_1 + t_2 + t_3) + average_reflectance * theta * E.t();
			
            solve(left_hand,right_hand,curr_reflectance);

            // temp_1 = A * curr_reflectance;
            // temp_2 = B * curr_reflectance;
            // temp_3 = D * curr_reflectance + C;
            temp_1 = MatrixMultiRows(a_columns, curr_reflectance);
            temp_2 = MatrixMultiRows(b_columns, curr_reflectance);
            temp_3 = MatrixMultiRows(d_columns, curr_reflectance) + C;

            d_1 = Shrink(temp_1 + b_1, 1.0 / lambda);
            d_2 = Shrink(temp_2 + b_2, 1.0 / lambda); 
            d_3 = Shrink(temp_3 + b_3, 1.0 / lambda);
        }

        // update b_1 and b_2
        b_1 = b_1 + temp_1 - d_1;
        b_2 = b_2 + temp_2 - d_2;
        b_3 = b_3 + temp_3 - d_3;

        // calculate current objective function value and output
        double part_1 = sum(abs(temp_1))[0];
        double part_2 = sum(abs(temp_2))[0];
		double part_4 = sum(abs(temp_3))[0];
        double part_5 = pow(E.t().dot(curr_reflectance) - average_reflectance,2.0);
        double obj_value = part_1 + part_2  + part_4 + theta * part_5;
		cout<<obj_value<<" "<<part_1<<" "<<part_2<<" "<<part_4<<" "<<part_5<<endl;		
    } 
    
    return curr_reflectance;
}


Mat_<double> GetReflectance(vector<ReflectanceCluster>& clusters, const CVImage& image, const Mat_<int>& pixel_label, int channel){
    cout<<"Initialize reflectance..."<<endl;
    // Initial reflectance to log(intensity) 
	int image_height = image.rows;
	int image_width = image.cols;
	int num_css = clusters.size();

    Mat_<double> log_image(image_height,image_width,0.0);
    for(int i = 0;i < image_height;i++){
        for(int j = 0;j < image_width;j++){
            // prevent log(0)
            log_image(i,j) = log(image(i,j)[channel] + 1); 
			// log_image(i,j) = log((image(i,j)[0] + image(i,j)[1] + image(i,j)[2]) / 3); 
        }
    }

    double gamma = 2000;
    double alpha = 50;
    double mu = 100;
    int iteration_num = 100;
    double lambda = 1;
    double beta = 400;
    double theta = 1000000;	
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

    reflectance = intensity.clone();
    reflectance = L1Regularization(log_image, image, intensity, pixel_label, clusters, gamma, alpha, mu, lambda, beta, theta, iteration_num);

    return reflectance;
}


#endif


