function [matches] = kcf_mesh_match(local, ref)
%% kcf mesh matching function
% find matching points located on the vertices of the mesh
%
    matches = [];
    width = size(local, 2);
    height = size(ref, 1);
    
    patch_size = 128;
%     search_size = 128;
    
    step_width_num = floor((width - patch_size) / 64 + 1);
    step_height_num = floor((height - patch_size) / 64 + 1);
    step_width = (width - patch_size) / double(step_width_num);
    step_height = (height - patch_size) / double(step_height_num);
    
    for row = 1:step_height_num
        for col = 1:step_width_num
            lu_roi_x = floor((col - 1) * step_height + 1);
            lu_roi_y = floor((row - 1) * step_width + 1);
            roi_x = lu_roi_x : (lu_roi_x + patch_size - 1);
            roi_y = lu_roi_y : (lu_roi_y + patch_size - 1);
            
            templ = local(roi_y, roi_x, :);
            background = ref(roi_y, roi_x, :);
            [~, dpos] = kcf_match(templ, background);
            
            match_pt = [lu_roi_x + patch_size / 2, lu_roi_y + patch_size / 2, ...
                lu_roi_x + patch_size / 2 + dpos(2), ...
                lu_roi_y + patch_size / 2 + dpos(1)];
            matches = [matches; match_pt];
        end
    end
    
    figure; ax = axes;
    showMatchedFeatures(local, ref, matches(:, 1:2), matches(:, 3:4), ...
        'montage', 'PlotOptions', {'ro', 'g+', ''});
    
end