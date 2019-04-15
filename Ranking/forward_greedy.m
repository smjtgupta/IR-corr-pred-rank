close all;
clear;
clc;

%%
% Sample Covariance matrix taken from the paper:
% Selecting band combinations from multispectral data, Sheffield
% They have computed the most relevant features from this matrix
% Optimal Ordering in decreasing Order is: 5, 4, 1, 7, 6, 3, 2

C = [53.32, 27.41, 35.74, 5.86, 36.04, 33.56, 7.77;
     27.41, 17.01, 21.35, 11.36, 29.35, 21.29, 4.13;
     35.74, 21.35, 31.66, 20.01, 46.46, 31.03, 6.69;
     5.86, 11.36, 20.01, 131.71, 131.64, 38.14, 8.26;
     36.04, 29.35, 46.56, 131.64, 210.83, 86.25, 19.10;
     33.56, 21.29, 31.03, 38.14, 86.25, 50.01, 11.51;
     7.77, 4.13, 6.69, 8.26, 19.10, 11.51, 9.80];

% convert the covariance matrix to correaltion matrix if needed
% while working with entities of uneven scale
% Use [ExpSigma,ExpCorrC] = cov2corr(ExpCovariance) for that 
 
figure;
imagesc(C);
title 'Covariance between Entries';
colorbar;

%% greedy_forward
 
k = 1; % counter
max_k = size(C,1); % number of elements
n = size(C,1); % count size
SsX = C; % temp variable for sub-determinant
val_det_for = zeros(1,max_k); % list of sub-determinant values after each exclusion
identity_det_for = zeros(1,max_k); % list of sub-determinant index chosen
list = 1:max_k;

while k<=n
    [maxval,argmax] = max(sum(SsX,2)); % compute principal index values
    val_det_for(1,k) = maxval;
    SsX(argmax,:) = []; % remove column entries for rejected index
    SsX(:,argmax) = []; % remove row entries for rejected index
    identity_det_for(1,k) = list(argmax);
    list(argmax) = [];
    k = k+1;
end


%%

disp('Greedy Forward Approach');
disp('Entries with decresing order of importance:');
disp(identity_det_for);
disp('Corresponding principal index value:');
disp(num2str(val_det_for(:).'))