import os
import sys
import os.path as op
sys.path.append(op.abspath(op.join(os.getcwd(), '..')))
sys.path.append(op.abspath(op.join(os.getcwd(), '..','..','python')))

# mesh path collection
from constants import MESHES_SHAPENET_DIR
fp_abs_ShapeNet = MESHES_SHAPENET_DIR
ids_ShapeNet = sorted(os.listdir(fp_abs_ShapeNet))
fn_abs_meshes_ShapeNet = [op.join(fp_abs_ShapeNet, id, "models/model_normalized.obj") \
                          for id in ids_ShapeNet]

from diff_mesh_utils import *
import time

force = False
num_samples = 16
imgs_num = 18


scale_lr = 0.008
euler_angles_lr = 0.000001
translation_lr  = 0.000001

num_iters_stage1 = 300
verbose_stage1 = ['plot_the_lo ss','save_opt_imgs_ togo_gif_display']

num_iters_stage2 = 800
verbose_stage2 = ['plot_the_lo ss','save_opt_imgs_ togo_gif_display']

num_iters_stage3 = 500
verbose_stage3 = ['plot_the_lo ss','save_opt_imgs_ togo_gif_display']

num_iters_stage4 = 300
verbose_stage4 = []

render_upsample_iter = [220]
sdf_n_iter = 512
sdf_write_opt_images = False

args = sys.argv
input_values = args[1:]
start_shapenet_id = int(input_values[0])
end_shapenet_id = int(input_values[1])
fp_save_head = input_values[2]
fp_rst_head = input_values[3]
# fp_save_head = 'Groceries3'

for shapenet_id in range(start_shapenet_id, end_shapenet_id+1):
    print("----------[",shapenet_id,"]------------")
    print(">>",ids_ShapeNet[shapenet_id],"<<")
    _start_time = time.time()
    # gen GT
    from ref_img_utils import load_ref_images, gen_ref_imgs_from_mesh
    resx = 256;resy = 256
    fn_model = fn_abs_meshes_ShapeNet[shapenet_id]
    fp_output = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'ref')
    ref_image_paths, ref_mask_paths = gen_ref_imgs_from_mesh(fn_model,fp_output,move_cam=False,
                                             resx=resx,resy=resy,num_samples=num_samples,
                                             imgs_num = imgs_num,format = 'exr',
                                             force=force)
    
    import torch
    import pyredner
    import matplotlib.pyplot as plt
    pyredner.set_use_gpu(torch.cuda.is_available())
    pyredner.set_print_timing(False)
    GT_with_texture = []
    GT_with_mask = []
    texture_color = []
    for i in range(len(ref_image_paths)):
        GT_with_texture.append(pyredner.imread(ref_image_paths[i]).to(pyredner.get_device()))
        GT_with_mask.append(pyredner.imread(ref_mask_paths[i]).to(pyredner.get_device()))
        mask_area = (GT_with_mask[-1] == 0).all(dim=-1)
        texture_color.append(GT_with_texture[-1][mask_area])
    most_frequent_color = most_common_color_histogram(texture_color).to(pyredner.get_device())
    """---------------------------------------------------------------------------------
    set
        """
    cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device())
    cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device() )
    cam_up = torch.tensor([0., 1., 0.], device = pyredner.get_device())
    white_background = torch.ones((resx,resy, 3)).to(pyredner.get_device())
    BLACK_COLOR = torch.zeros(3)
    DEFUALT_COLOR, V_COLOR, USE_UV_MAP = 0, 1, 2

    """---------------------------------------------------------------------------------
    gen sphere
        """
    theta_steps = 16
    phi_steps = 16
    
    sphere_num = 1
    radius = 0.05*torch.norm(cam_pos - cam_look_at).item()
    init_t = torch.zeros(sphere_num,3).to(pyredner.get_device())
    xyz_cam = CameraModel(resx,resy)
    img, new_XYZ_ = xyz_cam.init_sphere(pos_center = init_t,
                                        cam_pos = cam_pos,
                                        cam_look_at = cam_look_at,
                                        sphere_num = sphere_num, 
                                        theta_steps = theta_steps, 
                                        phi_steps = phi_steps,
                                        radius = radius,
                                        )
    xyz_cam.save_obj_in_setted_scene(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'sphere.obj'))
  
    
    
    while True:
        try:
            center = find_load_position(GT_with_mask[0], cam_pos, cam_look_at, CHECK = False)
            """---------------------------------------------------------------------------------
            stage 1
                """
            fn_obj = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'sphere.obj')
            cam = CameraModel(resx=resx,resy=resy)
            cam.load_obj(fn_obj)
            slice_mode = False
            standard_scale_mode = True
            
            translation = center.clone()
            euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            
            init_radius =  0.08*torch.norm(cam_pos - cam_look_at).item()
            scale = torch.eye(3).to(pyredner.get_device())
            scale.diagonal().fill_(init_radius)
            
            init_color = BLACK_COLOR
            obj_color = init_color.repeat(cam.get_v(obj_idx=-1).shape[0],1).cuda()
            color = None; color_mode = DEFUALT_COLOR; update_uv_coordinate = False
            obj_idx = 0; use_backup = False; return_img = True
            
            GT = GT_with_mask[0]
            background = white_background
            penalize_overlap = True
            opt_params_and_lr = [
                translation.detach_().requires_grad_(), 0.005,
                euler_angles.detach_().requires_grad_(), 0.005,
                scale.detach_().requires_grad_(), 0.005,
            ]
            best_params = diff_mesh_opt(cam = cam, GT = GT, penalize_overlap = penalize_overlap,
                                        color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate,
                                        slice_mode = slice_mode, 
                                        scale = scale, standard_scale_mode = standard_scale_mode,
                                        translation = translation,
                                        euler_angles = euler_angles,
                                        cam_pos = cam_pos, cam_look_at = cam_look_at,
                                        obj_idx = obj_idx, use_backup = use_backup, return_img = return_img,
                                        background = background,
                                        verbose = verbose_stage1,
                                        opt_params_and_lr = opt_params_and_lr,
                                        num_iters = num_iters_stage1,
            )
            """---------------------------------------------------------------------------------
            stage 2
                """
            stage2_add_sphere_imgs = []
            _, translation, euler_angles, scale, stage1_best_loss = best_params
            img_no_color = cam.set_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                       slice_mode = slice_mode, 
                                       scale = scale, standard_scale_mode = standard_scale_mode,
                                       translation = translation,
                                       euler_angles = euler_angles,
                                       cam_pos = cam_pos, cam_look_at = cam_look_at,
                                       obj_idx = obj_idx, use_backup = use_backup, return_img = return_img,
                                       )
            with_bg_img = add_background(img_no_color, white_background)
            stage2_add_sphere_imgs.append(torch.pow(with_bg_img.data, 1.0/2.2).detach().cpu())
            os.makedirs(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage1'), exist_ok=True)
            cam.save_obj_in_setted_scene(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage1','stage1.obj'), obj_idx = -1)
            
            print(">> start stage2 : add spheres")
            start_time = time.time()
            fn_obj = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage1','stage1.obj')
            cam = CameraModel(resx=resx,resy=resy)
            cam.load_obj(fn_obj)
            samples = cam.get_v().tolist()
            samples.extend(cam.gen_samples(number_of_points = 2000))
            
            translation = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            scale = torch.eye(3).to(pyredner.get_device())
            color = None; color_mode = DEFUALT_COLOR; update_uv_coordinate = False
            standard_scale_mode = True
            slice_mode = False
            obj_idx = -1; use_backup = False; return_img = True
            
            theta_steps = 16
            phi_steps = 16
            radius = 0.008*torch.norm(cam_pos - cam_look_at).item() # 0.015
            
            sample_num_iters = 0
            stage2_best_loss = stage1_best_loss
            fine_loss = Decimal('0.004999')
            GT = GT_with_mask[0]
            background = white_background
            penalize_overlap = True
            mask_weight = 2.0
            while(len(samples)>0):
                if len(samples)%200 == 0:
                    print(">>> ",len(samples))
                    
                sample_num_iters+=1
                pos = samples.pop(0)
                cam.transform_with_sphere(pos = pos, theta_steps = theta_steps, phi_steps = phi_steps, radius = radius)
            
                img_no_color = cam.set_obj(cam_pos = cam_pos, cam_look_at = cam_look_at, no_set = True)
                with_bg_img = add_background(img_no_color, white_background)
                loss = (with_bg_img - GT).pow(2).mean()
                with_gt_bg_img = add_background(img_no_color, GT)
                loss += (with_gt_bg_img - GT).pow(2).mean()*mask_weight # 2
                
                curr_loss = Decimal(f'{loss:.6f}')
            
                with localcontext() as ctx:
                    ctx.prec = 6
                    diff = stage2_best_loss - curr_loss
                    if diff > Decimal('0.000050'):
                        print(stage2_best_loss,'->', curr_loss)
                        opt_params_and_lr = [
                            euler_angles.detach_().requires_grad_(), 0.01,
                            scale.detach_().requires_grad_(), 0.01,
                        ]
                        best_params = diff_mesh_opt(cam = cam, GT = GT, penalize_overlap = penalize_overlap,
                                                    color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate,
                                                    slice_mode = slice_mode, 
                                                    scale = scale, standard_scale_mode = standard_scale_mode,
                                                    translation = translation,
                                                    euler_angles = euler_angles,
                                                    cam_pos = cam_pos, cam_look_at = cam_look_at,
                                                    obj_idx = obj_idx, use_backup = use_backup, return_img = return_img,
                                                    background = background,
                                                    verbose = verbose_stage2,
                                                    opt_params_and_lr = opt_params_and_lr,
                                                    num_iters = num_iters_stage2, out_iters =15,mask_weight = mask_weight,
                        )
                        _, euler_angles, scale, stage2_best_loss = best_params
                        cam.update_obj_list_with_scene_set(obj_idx = obj_idx)
                        imm = cam.set_obj(cam_pos = cam_pos, cam_look_at = cam_look_at, no_set = True)
                        with_bg_img = add_background(imm, white_background)
                        stage2_add_sphere_imgs.append(torch.pow(with_bg_img.data, 1.0/2.2).detach().cpu())
            
                        if stage2_best_loss < fine_loss:
                            del samples
                            break
                        
                        new_pos = cam.get_v_from_scene(obj_idx = obj_idx)
                        samples += new_pos.tolist()
                        euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
                        scale = torch.eye(3).to(pyredner.get_device())
                    else:
                        cam.transform_backward()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time: {:.2f} seconds".format(elapsed_time))
        
            obj_num = len(cam.get_object_list())
            cam.align_all_obj_in_list()
            os.makedirs(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage2'), exist_ok=True)
            cam.save_obj_in_list(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage2','stage2_add_sphere.obj'), obj_idx = -1)
            from ref_img_utils import rdImgs2Gif
            rdImgs2Gif(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage2','stage2.gif'),torch.stack(stage2_add_sphere_imgs))
            break
        except Exception as e:
            print("error and re-opt:", e)
            continue
    
    """---------------------------------------------------------------------------------
    stage 3
        """
    fn_obj = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage2','stage2_add_sphere.obj')
    cam = CameraModel(resx=resx,resy=resy)
    cam.load_obj(fn_obj)
    cam.test_update_v_num(obj_num)
    normals_backup, uvs_backup, uv_indices_backup = cam.cal_uv_coordinate()
    latitude_slice_num = 5
    longitude_slice_num = 2
    longi_and_lati = False
    if longi_and_lati:
        slice_num = latitude_slice_num*longitude_slice_num
    else:
        slice_num = latitude_slice_num
    cam.cal_sphere_slice_region(longi_and_lati = longi_and_lati, latitude_slice_num = latitude_slice_num, longitude_slice_num = longitude_slice_num)
    background = white_background
    uv_map = torch.zeros(1024, 1024, 3).to(pyredner.get_device()) + most_frequent_color
    after_pose_estimation_imgs = []
    XYZ_info = []
    EA_info = []
    XYZ_info.append(torch.tensor([0., 0., 0.], device=pyredner.get_device()))
    EA_info.append(torch.tensor([0., 0., 0.], device=pyredner.get_device()))

    print(">> start stage3 : init pose")
    start_time = time.time()
    
    for img_idx in range(len(GT_with_mask)):
        print("> ",img_idx,"...")
        """---------------------------------------------------------------------------------
        phase1 多段opt+uv map
            """
        verbose = ['plot_the_lo ss','save_opt_imgs_ togo_gif_display']
        slice_mode = True; easy_slice = False  ; scale_slice_num = 2 #; slice_num = scale_slice_num
        standard_scale_mode = True
    
        translation_ = [torch.tensor([0., 0., 0.], device=pyredner.get_device()) for _ in range(obj_num)]
        translation = torch.stack(translation_)
        euler_angles_ = [torch.tensor([0., 0., 0.], device=pyredner.get_device()) for _ in range(obj_num)]
        euler_angles = torch.stack(euler_angles_)
        
        scale_ = [torch.eye(3).to(pyredner.get_device()) for _ in range(slice_num)]
        scale = torch.stack(scale_)
        scale_ = [scale.clone() for _ in range(obj_num)]
        scale = torch.stack(scale_)
        
        color = uv_map; color_mode = USE_UV_MAP; update_uv_coordinate = False
        
        GT = [GT_with_texture[img_idx],GT_with_mask[img_idx]]
        penalize_overlap = True
        opt_params_and_lr = [
            translation.detach_().requires_grad_(), translation_lr,
            euler_angles.detach_().requires_grad_(), euler_angles_lr,
            scale.detach_().requires_grad_(), scale_lr,
            uv_map.detach_().requires_grad_(), 0.08,
        ]
        best_params = diff_mesh_opt(cam = cam, GT = GT, penalize_overlap = penalize_overlap,
                                    color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate,
                                    slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                    scale = scale, standard_scale_mode = standard_scale_mode,
                                    translation = translation,
                                    euler_angles = euler_angles,
                                    cam_pos = cam_pos, cam_look_at = cam_look_at,
                                    background = background,
                                    verbose = verbose_stage3,
                                    opt_params_and_lr = opt_params_and_lr,
                                    num_iters = num_iters_stage3,out_iters = 15,
                                    opt_aligned_obj_mode = True,
                                    pose_estimation_mode = True, mask_weight = 5.0, color_weight = 1.0,
        )
        _, translation, euler_angles, scale, uv_map, _ = best_params
        # _, euler_angles, scale, uv_map, _ = best_params
        uv_map = torch.clamp(uv_map, 0, 1)
        color = uv_map # important or can't use deepcopy
        # record
        img_with_texture = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = False, 
                                                slice_mode = slice_mode,  standard_scale_mode = standard_scale_mode,
                                                scale = scale,
                                                translation = translation,
                                                euler_angles = euler_angles,
                                                cam_pos = cam_pos, cam_look_at = cam_look_at,
                                               )
        with_bg_img_with_texture = add_background(img_with_texture, background)
        with_bg_img_with_texture = torch.clamp(with_bg_img_with_texture, 0, 1)
        after_pose_estimation_imgs.append(torch.pow(with_bg_img_with_texture, 1.0/2.2).cpu().detach())
        
        if img_idx == 0:
            os.makedirs(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','one_frame'), exist_ok=True)
            cam.save_obj_in_setted_scene(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','one_frame','stage3_one_frame_opts.obj'), obj_idx = -1)
    
        cam.update_obj_list_with_scene_set(obj_idx = -1)
        """---------------------------------------------------------------------------------
        phase2 pos esti
            """
        if img_idx == len(GT_with_mask) -1 :
            break
        slice_mode = False
        standard_scale_mode = True
        translation = torch.tensor([0., 0., 0.], device = pyredner.get_device())
        euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
        scale = torch.eye(3).to(pyredner.get_device())
        color = uv_map; color_mode = USE_UV_MAP; update_uv_coordinate = False

        GT = [GT_with_texture[img_idx+1],GT_with_mask[img_idx+1]]
        opt_params_and_lr = [
            translation.detach_().requires_grad_(), 0.001,
            euler_angles.detach_().requires_grad_(), 0.005,
        ]
        best_params = diff_mesh_opt(cam = cam, GT = GT, penalize_overlap = penalize_overlap,
                                    color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate,
                                    slice_mode = slice_mode, 
                                    scale = scale, standard_scale_mode = standard_scale_mode,
                                    translation = translation,
                                    euler_angles = euler_angles,
                                    cam_pos = cam_pos, cam_look_at = cam_look_at,
                                    background = background,
                                    verbose = verbose_stage3,
                                    opt_params_and_lr = opt_params_and_lr,
                                    num_iters = num_iters_stage3,out_iters = 15,
                                    opt_aligned_obj_mode = True,
                                    pose_estimation_mode = True, mask_weight = 3.,
        )
        _, translation, euler_angles, _ = best_params
        XYZ_info.append(translation)
        EA_info.append(euler_angles)
        img_with_texture = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                                slice_mode = slice_mode, 
                                                scale = scale,
                                                translation = translation,
                                                euler_angles = euler_angles,
                                                cam_pos = cam_pos, cam_look_at = cam_look_at,
                                               )
        cam.update_obj_list_with_scene_set(obj_idx = -1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    os.makedirs(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3'), exist_ok=True)
    rdImgs2Gif(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3.gif'),torch.stack(after_pose_estimation_imgs))
    cam.save_obj_in_setted_scene(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3.obj'), obj_idx = -1)
    translation_estimated_info = []
    euler_angles_estimated_info = []
    XYZ = 0
    EA = 0
    for i in range(len(XYZ_info)):
        XYZ += XYZ_info[i]
        EA += EA_info[i]
        translation_estimated_info.append(XYZ.clone()) # 注意一定要clone
        euler_angles_estimated_info.append(EA.clone())
    fn_obj_pose_info = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3_object_pose_info.pt')
    torch.save((translation_estimated_info, euler_angles_estimated_info), fn_obj_pose_info)
    """---------------------------------------------------------------------------------
    stage 4
        """
    print(">> start stage4 : all")
    cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device())
    cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device() )
    cam_up = torch.tensor([0., 1., 0.], device = pyredner.get_device())
    fn_obj = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage2','stage2_add_sphere.obj')
    cam = CameraModel(resx=resx,resy=resy)
    cam.load_obj(fn_obj)
    cam.test_update_v_num(obj_num)
    
    # translation_estimated_info, euler_angles_estimated_info = torch.load(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3_object_pose_info.pt'))
    # normals_backup, uvs_backup, uv_indices_backup = cam.cal_uv_coordinate()

    
    cam.normals_backup[str(-1)], cam.uvs_backup[str(-1)], cam.uv_indices_backup[str(-1)] = normals_backup, uvs_backup, uv_indices_backup
    latitude_slice_num = 5
    longitude_slice_num = 2
    longi_and_lati = False
    if longi_and_lati:
        slice_num = latitude_slice_num*longitude_slice_num
    else:
        slice_num = latitude_slice_num
    cam.cal_sphere_slice_region(longi_and_lati = longi_and_lati, latitude_slice_num = latitude_slice_num, longitude_slice_num = longitude_slice_num)
    background = white_background
    uv_map = torch.zeros(1024, 1024, 3).to(pyredner.get_device()) + most_frequent_color

    penalize_overlap = True
    color = uv_map
    color_mode = USE_UV_MAP
    update_uv_coordinate = False
    slice_mode = True; easy_slice = False; scale_slice_num = 2
    
    standard_scale_mode = True
    translation_ = [torch.tensor([0., 0., 0.], device=pyredner.get_device()) for _ in range(obj_num)]
    translation = torch.stack(translation_)
    euler_angles_ = [torch.tensor([0., 0., 0.], device=pyredner.get_device()) for _ in range(obj_num)]
    euler_angles = torch.stack(euler_angles_)
    
    scale_ = [torch.eye(3).to(pyredner.get_device()) for _ in range(slice_num)]
    scale = torch.stack(scale_)
    scale_ = [scale.clone() for _ in range(obj_num)]
    scale = torch.stack(scale_)
    
    opt_params_and_lr = [
            translation.detach_().requires_grad_(), translation_lr,
            euler_angles.detach_().requires_grad_(), euler_angles_lr,
            scale.detach_().requires_grad_(), scale_lr,
            uv_map.detach_().requires_grad_(), 0.08,
        ]
    best_params =  diff_mesh_opt_all(cam = cam, GT_with_texture = GT_with_texture, GT_with_mask = GT_with_mask, penalize_overlap = penalize_overlap,
                                      color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                      slice_mode = slice_mode, easy_slice = easy_slice,
                                      scale = scale, standard_scale_mode = standard_scale_mode,
                                      translation = translation,
                                      euler_angles = euler_angles,
                                      cam_pos = cam_pos,
                                      cam_look_at = cam_look_at,
                                      translation_for_all_list = translation_estimated_info,
                                      euler_angles_for_all_list = euler_angles_estimated_info,
                                      background = background,
                                      verbose = verbose_stage4,
                                      opt_params_and_lr = opt_params_and_lr,
                                      num_iters = num_iters_stage4, out_iters = 15,
                                      pose_estimation_mode = True, mask_weight = 5., 
                                      write_opt_images = False, 
                                      fp_write_opt_images = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4','opt_imgs'),
                                     )
    _, translation, euler_angles, scale, uv_map, _ = best_params
    uv_map = torch.clamp(uv_map, 0, 1)
    os.makedirs(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4'), exist_ok=True)
    torch.save((translation, euler_angles, scale, uv_map), op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4', 'stage4_object_opt_info.pt'))
    color = uv_map
    img_with_texture = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                              slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                              scale = scale, standard_scale_mode = standard_scale_mode,
                                              translation = translation,
                                              euler_angles = euler_angles,
                                              cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                             )
    with_bg_img_with_texture = add_background(img_with_texture, background)
    with_bg_img_with_texture = torch.clamp(with_bg_img_with_texture, 0, 1)
    cam.save_obj_in_setted_scene(op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4','stage4_after_all_opts.obj'), obj_idx = -1)
    cam.update_obj_list_with_scene_set(obj_idx = -1)
    # """---------------------------------------------------------------------------------
    # stage 5 sdf
    #     """
    # print(">> start stage5 : sdf")
    # n_sensors = 9
    # radius = torch.norm(cam_look_at - cam_pos).item()
    # cam_pos_candidates = sample_spherical(n_sensors,cam_look_at,radius)

    # syn_cam_poses = []
    # adjusted_cam_pos = cam_pos - cam.center[-1]
    # syn_cam_poses.append(adjusted_cam_pos.tolist())
    # ref_img_idx = 0
    # sdf_ref_imgs_path = []
    # sdf_ref_imgs_path.append(ref_image_paths[0])
    # weight_box = []
    # weight_box.append(5.0)
    # __fp_ref = op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage_sdf','ref_img')
    # os.makedirs(__fp_ref, exist_ok=True)
    # from ref_img_utils import img2exr
    # color = uv_map; color_mode = USE_UV_MAP; update_uv_coordinate = False
    # slice_mode = False
    # scale = torch.eye(3).to(pyredner.get_device())
    # translation = torch.tensor([0., 0., 0.], device = pyredner.get_device())
    # euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
    # for cam_pos_candidate in cam_pos_candidates:
    #     ref_img_idx += 1
    #     syn_cam_poses.append(cam_pos_candidate.tolist())
    #     img_with_texture = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
    #                                           slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
    #                                           scale = scale, standard_scale_mode = standard_scale_mode,
    #                                           translation = translation - cam.center[-1],
    #                                           euler_angles = euler_angles,
    #                                           cam_pos = cam_pos_candidate, cam_look_at = cam_look_at, cam_up = cam_up,
    #                                          )
    #     with_bg_img_with_texture = add_background(img_with_texture, background)
    #     with_bg_img_with_texture = torch.clamp(with_bg_img_with_texture, 0, 1)
    #     fn_save_exr = op.abspath(op.join(__fp_ref, f'{ref_img_idx:03d}.exr'))
    #     img2exr(with_bg_img_with_texture,fn_save_exr)
    #     sdf_ref_imgs_path.append(fn_save_exr)
    #     weight_box.append(1.0)

    # from constants import  RENDER_DIR
    # fp_syn_imgs = op.join(RENDER_DIR,'syn_imgs',ids_ShapeNet[shapenet_id])
    # new_fp_syn_imgs = op.join(RENDER_DIR,'syn_imgs_exr',ids_ShapeNet[shapenet_id])
    # os.makedirs(new_fp_syn_imgs, exist_ok=True)
    # from ref_img_utils import convert_to_exr
    # convert_to_exr(fp_syn_imgs,new_fp_syn_imgs)

    # need = [
    #     # '0_0+0_0.exr',
    #     '0_0+22_5.exr',
    #     '0_0+-22_5.exr',
    #     # '90+0.exr',
    #     # '-90+0.exr',
    # ]
    # zore_123_ref_imgs_path, zore_123_cam_pos_data = get_images(new_fp_syn_imgs, need)
    # sdf_ref_imgs_path += zore_123_ref_imgs_path
    # for i in range(len(zore_123_ref_imgs_path)):
    #     weight_box.append(3.0)
    # pyredner_cam_look_at = cam_look_at.tolist()
    # pyredner_cam_pos = adjusted_cam_pos.tolist()
    # zore_123_cam_poses = calculate_camera_positions(zore_123_cam_pos_data,pyredner_cam_look_at,pyredner_cam_pos)
    # syn_cam_poses += zore_123_cam_poses

    # import mitsuba as mi
    # mi.set_variant('cuda_ad_rgb')
    # my_sensors = []
    # sdf_target_pos = [0.5,0.5,0.5]
    # for i in range(len(syn_cam_poses)):
    #     ref_cam_pos = syn_cam_poses[i][:]
    #     for idx in range(3):
    #         ref_cam_pos[idx] += sdf_target_pos[idx]
    #     my_sensors.append(mi.load_dict({
    #                 'type': 'perspective',
    #                 'fov': 45.0,
    #                 'to_world': mi.ScalarTransform4f.look_at(target=sdf_target_pos, origin=ref_cam_pos, up=[0, 1, 0]),
    #                 'sampler': {'type': 'independent'},
    #                 'film': {
    #                     'type': 'hdrfilm', 
    #                     'width': resx, 'height': resy,
    #                     'filter': {'type': 'gaussian'},
    #                     'sample_border': True,
    #                 }
    #             }))

    # scene_name = 'LfD_base'
    # resx = 256;resy = 256
    # dryrun = False # 不创建文件夹
    # opt_config_name = 'principled-lfd-hq'
    # render_spp = 256 # 越低图像噪点越多
    # sdf_res = 128
    # sdf_init_radius = 0.2 # 128[0.1] 64[0.15] 32[0.35]
    # render_upsample_iter = [120] # 前220 iter用的是scene_config.init_res，低resolution以加快初始变形速度
    # checkpoint_frequency = 0 # 每隔128 iters保存一次optimized shape
    # import importlib
    # import diff_sdf_utils
    # importlib.reload(diff_sdf_utils)
    # from diff_sdf_utils import sdfScene
    # from constants import SCENE_DIR
    # opt_scene_fn = op.join(SCENE_DIR, scene_name, f'{scene_name}.xml')

    # weight_important = True
    # from ref_img_utils import load_ref_images
    # sdf_opt_ref_imgs = load_ref_images(sdf_ref_imgs_path, multiscale = True)
    # # get new_sdf_state
    # verbose = []
    # diff_sdf_scene =  sdfScene(scene_name, verbose, opt_config_name, dryrun = dryrun,
    #                        render_upsample_iter = render_upsample_iter,
    #                        checkpoint_frequency = checkpoint_frequency,
    #                        resx = resx,resy = resy,sdf_res=sdf_res,sdf_init_radius=sdf_init_radius,
    #                        render_spp = render_spp, my_sensors = my_sensors,
    #                        dir_endswith = ids_ShapeNet[shapenet_id])
    # diff_sdf_scene.load_sdf_scene(opt_scene_fn)
    # diff_sdf_scene.sdf_shape_opt(sdf_opt_ref_imgs,0,
    #                              n_iter = sdf_n_iter,out_iters = 20,
    #                              update_sensors = True, my_sensors = my_sensors,
    #                              weight_important=weight_important, weight_box = weight_box,
    #                             save_load_img = False,write_opt_images=sdf_write_opt_images)

    # sdf_filename = diff_sdf_scene.get_vol_fn(0)[0]
    # opt_mesh_dir = diff_sdf_scene.get_mesh_dir()
    # from marching_cube import MarchingCube
    # mc_scale = 0.01
    # mc_eu = [0.0, 0.0, 0.0]
    # fn_save_obj = op.join(opt_mesh_dir,f'mc_{0:03d}.obj')
    # MarchingCube(sdf_filename, mc_scale, mc_eu).save_obj(fn_save_obj)

    # diff_sdf_scene.gen_render_turntable()
    # string = opt_mesh_dir
    # input_folder = string.replace('mesh', 'turntable')
    # fn_save_gif = string.replace('mesh', 'vid.gif')
    # gen_gif(input_folder,fn_save_gif)

    """---------------------------------------------------------------------------------
    turntable
        """
    fn_obj_list = [
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','one_frame','stage3_one_frame_opts.obj'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3.obj'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4','stage4_after_all_opts.obj'),
    ]
    fn_gif_list = [
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','one_frame','obj_stage3_one_frame_opts.gif'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','obj_stage3.gif'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage4','obj_stage4_after_all_opts.gif'),
    ]
    fn_copy_gif_list = []
    for fn_obj, fn_gif in zip(fn_obj_list, fn_gif_list):
        if op.exists(fn_obj):
            fn_copy_gif_list.append(fn_gif)
            cam = CameraModel(resx=resx,resy=resy)
            cam.load_obj(fn_obj)
            color_mode = DEFUALT_COLOR
            slice_mode = False
            cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device())
            cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            cam_up = torch.tensor([0., 1., 0.], device = pyredner.get_device())
            translation = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device())
            scale = torch.eye(3).to(pyredner.get_device())
            num_turn_imgs = 30
            azimuth_min = torch.tensor(torch.pi*0).cuda()
            azimuth_max = torch.tensor(torch.pi*2).cuda()
            azimuth = torch.linspace(azimuth_min, azimuth_max, num_turn_imgs+1, device = pyredner.get_device())
            turntable_imgs = []
            background = black_background = torch.zeros((resx,resy, 3)).to(pyredner.get_device())
            for i in range(num_turn_imgs):
                neo_euler_angles = euler_angles.clone()
                neo_euler_angles[1] += azimuth[i]
                img_with_texture = cam.set_aligned_obj(color_mode = color_mode,
                                                          slice_mode = slice_mode, 
                                                          scale = scale,
                                                          translation = translation,
                                                          euler_angles = neo_euler_angles,
                                                          cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                                         )
                with_bg_img_with_texture = add_background(img_with_texture, background)
                with_bg_img_with_texture = torch.clamp(with_bg_img_with_texture, 0, 1)
                turntable_imgs.append(torch.pow(with_bg_img_with_texture, 1.0/2.2).cpu().detach())
            ex_gif_imgs = stage2_add_sphere_imgs.copy()
            ex_gif_imgs.extend(turntable_imgs)
            rdImgs2Gif(fn_gif,torch.stack(ex_gif_imgs))

    import shutil
    fp_rst = op.join(fp_rst_head,ids_ShapeNet[shapenet_id])
    os.makedirs(fp_rst, exist_ok=True)
    source_files = [
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'ref','video.gif'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'ref','pose_info.pt'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3.gif'),
        op.join(fp_save_head,ids_ShapeNet[shapenet_id],'stage3','stage3_object_pose_info.pt'),
    ]
    source_files += fn_copy_gif_list
    for file in source_files:
        if os.path.isdir(file):
            shutil.copytree(file, op.join(fp_rst,op.basename(file)),  dirs_exist_ok =True)
        else:
            shutil.copy(file, fp_rst)
    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    print(">>Total elapsed time: {:.2f} seconds".format(_elapsed_time))