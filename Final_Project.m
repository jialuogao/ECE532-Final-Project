clear
close all
image = im2double(imread('image_1.jpg'));
image = imresize(image, 0.2);
figure;
imshow(image)
title('Original Image')



%%Weiner filter
LEN = 21;
THETA = 11;
PSF_W = fspecial('motion', LEN, THETA);


%%Gaussian filter
hsize = 10;
sigma = 5;
PSF_G = fspecial('gaussian',hsize,sigma);

filter = PSF_W
T = convmtx2(filter, [size(image)]);
Y1 = reshape(T*image(:), size(filter)+[size(image)]-1);
figure, imshow(Y1)
title('Blurred Image')

Y1_minRow = ceil(size(filter,1)/2)+1;
Y1_maxRow = size(Y1,1)-floor(size(filter,1)/2);
Y1_minCol = ceil(size(filter,2)/2)+1;
Y1_maxCol = size(Y1,2)-floor(size(filter,2)/2);
Y1 = Y1(Y1_minRow:Y1_maxRow, Y1_minCol:Y1_maxCol);

A = inv(T'*T)*T';
X = reshape(A' * Y1(:), size(filter)+[size(image)]-1);
figure,imshow(X)
title('Restored Image')



