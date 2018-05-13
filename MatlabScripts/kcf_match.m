function [response, dpos] = kcf_match(templ, background)
%% kcf matching function
% function used to find correspondence between templ patch and background
% patch using kernelized correlation filter
% 
%
%
    templ = double(templ);
    background = double(background);
    
    sigma = 0.2;
    lambda = 1e-4;
    output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
    cell_size = 1;
    target_sz = [size(templ, 1), size(templ, 2)];
    window_sz = [size(background, 1), size(background, 2)];
    %create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';
  
    % fast training
%     xf = fft2(templ .* cos_window);
    xf = fft2(templ);
    kf = gaussian_correlation(xf, xf, sigma);
    alphaf = yf ./ (kf + lambda);
    
    % fast tracking
%     zf2 = fft2(background .* cos_window);
    zf2 = fft2(background);
    kzf = gaussian_correlation(zf2, xf, sigma);
    response = real(ifft2(alphaf .* kzf));
    
    %target location is at the maximum response. we must take into
    %account the fact that, if the target doesn't move, the peak
    %will appear at the top-left corner, not at the center (this is
    %discussed in the paper). the responses wrap around cyclically.
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    if vert_delta > size(zf2,1) / 2,  %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - size(zf2,1);
    end
    if horiz_delta > size(zf2,2) / 2,  %same for horizontal axis
        horiz_delta = horiz_delta - size(zf2,2);
    end
    dpos = cell_size * [vert_delta - 1, horiz_delta - 1];
%     disp(dpos);
end