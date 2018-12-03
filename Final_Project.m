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

T = convmtx2(PSF_W, [size(image)]);
Y1 = reshape(T*image(:), size(PSF_W)+[size(image)]-1);
figure, imshow(Y1)
title('Blurred Image')
Y1 = Y1(3:size(Y1,1)-2,11:size(Y1,2)-10);
A = inv(T'*T)*T';
X = reshape(A' * Y1(:), size(PSF_W)+[size(image)]-1);
figure,imshow(X)
title('Restored Image')



