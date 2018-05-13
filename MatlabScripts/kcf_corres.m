clear;
close all;
fclose all;

addpath('./KCF');

ref_edge = imread('../data/rgb/local_00.jpg');
local_edge = imread('../data/rgb/local_00_refBlk.jpg');

ref_edge = imresize(rgb2gray(ref_edge), 0.2);
local_edge = imresize(rgb2gray(local_edge), 0.2);

% kcf_mesh_match(local_edge, ref_edge);

patch_size = 128;
search_size = 128;
lu_roi_x = 100;
lu_roi_y = 100;
roi_x = lu_roi_x : (lu_roi_x + patch_size - 1);
roi_y = lu_roi_y : (lu_roi_y + patch_size - 1);
lu_search_x = 75;
lu_search_y = 60;
search_x = lu_search_x : (lu_search_x + search_size - 1);
search_y = lu_search_y : (lu_search_y + search_size - 1);
templ = local_edge(roi_x, roi_y, :);
background = ref_edge(search_x, search_y, :);

figure(1);
subplot(1, 2, 1);imshow(templ, []);
subplot(1, 2, 2);imshow(background, []);
[response, dpos] = kcf_match(templ, background);
disp(dpos);