#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from tqdm import tqdm
#pc:gaussians
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #radii:radio of gaussian in image
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled(not in visible range) or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,##gaossian 
            "visibility_filter" : radii > 0,
            "radii": radii}

#     return points_2d

# these method should refer to forward.cu,like float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
def project_points(points_3d, full_proj_transform):
    """
    Project 3D points to 2D using a full projection transformation.

    Args:
        points_3d (np.ndarray): Array of shape (N, 3) representing 3D points.
        full_proj_transform (np.ndarray): Full projection transformation matrix.

    Returns:
        np.ndarray: Array of shape (N, 2) containing the (u, v) coordinates on the film.
    """
    # Convert points_3d to homogeneous coordinates
    points_3d_homogeneous = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)), axis=1)

    # Apply the transformation
    # TODO full_proj_transform may need transverse
    # full_proj_transform
    points_2d_homogeneous = np.matmul(full_proj_transform.T, points_3d_homogeneous.T).T

    # # Keep only the x and y coordinates
    # points_2d = points_2d_homogeneous[:, :2]

    # Normalize by the fourth coordinate to get 3D coordinates
    epsilon = 0.0001
    points_2d = points_2d_homogeneous[:, :2] / (points_2d_homogeneous[:, 3][:, np.newaxis] + epsilon)

    return points_2d
def generate_depth_map(points_3d, camera_center, projection_matrix, image_size,radii):
    with torch.no_grad():
        # Move tensors to the device of points_3d
        camera_center = camera_center.to(points_3d.device)
        projection_matrix = projection_matrix.to(points_3d.device)

        # Convert PyTorch tensors to NumPy arrays
        points_3d_np = points_3d.cpu().numpy()

        # Calculate camera to points vector
        camera_to_points = points_3d_np - camera_center.cpu().numpy()
        # Get image size
        width, height = image_size
        # Project points to 2D using NumPy
        points_2d = project_points(points_3d_np, projection_matrix.cpu().numpy())
        # Transform x from the range [-1, 1] to [0, width]

        # __forceinline__ __device__ float ndc2Pix(float v, int S)
        # {
        #     return ((v + 1.0) * S - 1.0) * 0.5;
        # }
        points_2d[:, 0] = ((points_2d[:, 0] + 1) * width - 1)* 0.5
        points_2d[:, 1] = ((points_2d[:, 1] + 1) * height - 1)* 0.5

        # Initialize depth map using NumPy
        depth_map_np = np.full((height, width), float('inf'), dtype=np.float32)

        # Iterate over projected points and compute depth map
        for i in tqdm(range(points_2d.shape[0]), desc="Processing points_2d", unit="point"):
            depth_value = np.linalg.norm(camera_to_points[i])

            # Store depth value in depth map
            x, y = points_2d[i]
            x = int(x)
            y = int(y)
            if 0 <= x < width and 0 <= y < height:
                depth_map_np[y, x] = min(depth_value,depth_map_np[y, x])
            # #can't use points_2d, should points_2d[i]
            # x = int(points_2d[i, 0])
            # y = int(points_2d[i, 1])
            # radius = int(min(1/2*radii[i],2))

            # # 确保不越界，处理边界情况
            # x_min, x_max = max(x - radius, 0), min(x + radius, width-1)
            # y_min, y_max = max(y - radius, 0), min(y + radius, height - 1)


            # # 更新 depth_map_np
            # depth_map_np[y_min:y_max + 1, x_min:x_max + 1] = np.minimum(depth_map_np[y_min:y_max + 1, x_min:x_max + 1], depth_value)
        # Convert the NumPy depth map back to a PyTorch tensor
        depth_map = torch.from_numpy(depth_map_np).to(points_3d.device)

        return depth_map
def render_point(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #radii:radio of gaussian in image
    __, radii= rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # First filtering step based on radii
    with torch.no_grad():
        radii_np = radii.cpu().numpy()
        valid_mask_radii = radii_np > 0
        valid_indices_radii = torch.from_numpy(valid_mask_radii.astype(np.bool_))

    # Use the boolean mask to filter means3D and opacity
    points_3d = means3D[valid_indices_radii]
    opacity = opacity[valid_indices_radii]
    radii = radii[valid_indices_radii]
    # Second filtering step based on opacity
    with torch.no_grad():
        opacity_np = opacity.cpu().numpy()
        valid_mask_opacity = opacity_np > 0.1
        valid_indices_opacity = torch.from_numpy(valid_mask_opacity.astype(np.bool_))
        valid_indices_opacity = valid_indices_opacity.squeeze()

    # # Use the boolean mask to filter means3D and opacity again
    # points_3d = points_3d[valid_indices_opacity]
    # opacity = opacity[valid_indices_opacity]
    # radii = radii[valid_indices_opacity]
    camera_center = viewpoint_camera.camera_center
    projection_matrix = viewpoint_camera.full_proj_transform
    # image_size = [int(viewpoint_camera.image_height),int(viewpoint_camera.image_width)]
    image_size = [int(viewpoint_camera.image_width),int(viewpoint_camera.image_height)]
    depth_map = generate_depth_map(points_3d, camera_center, projection_matrix, image_size,radii)
    return depth_map