% Input: segment result
segment_result  = 'C:\\Users\\BiSai\\Desktop\\segment_result.txt';
file = fopen(segment_result);
segment = fscanf(file,'%d');
cluster_num = segment(1);
segment = segment(2:end);

image_path = 'C:\\Users\\BiSai\\Desktop\\baby.tif';
image = imread(image_path);

% get pairwise weight
image_width = size(image, 2);
image_height = size(image, 1);

segment = reshape(segment, image_width, image_height);
cluster_feature = zeros(cluster_num, 5);
image_diameter = (image_width^2 + image_height^2)^0.5;
cluster_count = zeros(cluster_num,1);
for i = 1 : image_height
	for j = 1 : image_width
		label = segment(i,j);
		cluster_count(label) = cluster_count(label) + 1;
		cluster_feature(label,1) = cluster_feature(label,1) + i / image_diameter;
		cluster_feature(label,2) = cluster_feature(label,2) + j / image_diameter;
		cluster_feature(label,3) = cluster_feature(label,3) + image(i,j,1);
		cluster_feature(label,4) = cluster_feature(label,4) + image(i,j,2);
		cluster_feature(lable,5) = cluster_feature(label,5) + image(i,j,3);		
	end
end

theta_p = 0.1;
theta_l = 0.1;
theta_c = 0.025;
for i = 1 :  cluster_num
	cluster_feature(i,1) = cluster_feature(i,1) / (theta_p * cluster_count(i));
	cluster_feature(i,2) = cluster_feature(i,2) / (theta_p * cluster_count(i));
	temp = cluster_feature(3) + cluster_feature(4) + cluster_feature(5);
	temp = temp / (double)cluster_num;
	cluster_feature(i,3) = temp / (3 * theta_l);
	cluster_feature(i,4) = cluster_feature(4) / (cluster_num * temp * theta_c);
	cluster_feature(i,5) = cluster_feature(5) / (cluster_num * temp * theta_c);
end

dist = 

