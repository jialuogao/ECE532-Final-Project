%% setting up
clear
close all
image = im2double(rgb2gray(imread('bucky.jpg')));
image = imresize(image, 0.3);
figure;
imshow(image)
title('Original Image')

%% generate different filters
% Gaussian filter
hsize = 5;
sigma = 5;
gas = fspecial('gaussian',hsize,sigma);
% Motion filter (Weiner)
len = 5;
theta = 5;
mtn = fspecial('motion', len, theta);
% Averaging filter
hsize = 5;
avg = fspecial('average',hsize);
% Edge filter
edg = [0 -1 0; -1 4 -1; 0 -1 0];
% Sharpen filter
shr = [0 -1 0; -1 5 -1; 0 -1 0];

%% blurring using Gaussian filter 
filter = gas;
blurred_img = conv2(image, filter, 'same');
figure, imshow(blurred_img);
title('Blurred Image')

%% deblurring by solving least square
T = convmtx2(filter, size(image));
A = (T'*T)\T';
deblurred_img = reshape(A' * blurred_img(:), size(filter)+size(image)-1);
% resize the blurred image
b_minRow = ceil(size(filter,1)/2);
b_maxRow = size(deblurred_img,1)-floor(size(filter,1)/2);
b_minCol = ceil(size(filter,2)/2);
b_maxCol = size(deblurred_img,2)-floor(size(filter,2)/2);
deblurred_img = deblurred_img(b_minRow:b_maxRow, b_minCol:b_maxCol);

figure,imshow(deblurred_img)
title('Restored Image')

%% add noise to the blurred image
noise = rand(size(image))*0.01;
blurred_img = blurred_img - noise;
figure;imshow(blurred_img);
title('Blurred Image With Noise')

%% deblurring by solving least square
A = (T'*T)\T';
deblurred_img = reshape(A' * blurred_img(:), size(filter)+size(image)-1);
% resize the blurred image
b_minRow = ceil(size(filter,1)/2);
b_maxRow = size(deblurred_img,1)-floor(size(filter,1)/2);
b_minCol = ceil(size(filter,2)/2);
b_maxCol = size(deblurred_img,2)-floor(size(filter,2)/2);
deblurred_img = deblurred_img(b_minRow:b_maxRow, b_minCol:b_maxCol);

figure,imshow(deblurred_img)
title('Restored Noisy Image with LS')

%% deblurring by solving least square
r = 25;
[U,S,V] = svd(blurred_img);
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
blurred_img_svd = U*S*V';

deblurred_img = reshape(A' * blurred_img_svd(:), size(filter)+size(image)-1);
% resize the blurred image
b_minRow = ceil(size(filter,1)/2);
b_maxRow = size(deblurred_img,1)-floor(size(filter,1)/2);
b_minCol = ceil(size(filter,2)/2);
b_maxCol = size(deblurred_img,2)-floor(size(filter,2)/2);
deblurred_img = deblurred_img(b_minRow:b_maxRow, b_minCol:b_maxCol);
figure,imshow(deblurred_img)
title('Restored Noisy Image with SVD Low Rank Approximation and LS')

%% deblurring by solving least square
A = (T'*T + 0.005*eye(size(T,2),size(T,2)))\T';
deblurred_img = reshape(A' * blurred_img(:), size(filter)+size(image)-1);
% resize the blurred image
b_minRow = ceil(size(filter,1)/2);
b_maxRow = size(deblurred_img,1)-floor(size(filter,1)/2);
b_minCol = ceil(size(filter,2)/2);
b_maxCol = size(deblurred_img,2)-floor(size(filter,2)/2);
deblurred_img = deblurred_img(b_minRow:b_maxRow, b_minCol:b_maxCol);

figure,imshow(deblurred_img)
title('Restored Noisy Image with LS with Tikhonov')