import os
import sys
import glob
import os.path as op
sys.path.append(op.abspath(op.join(os.getcwd(), '..')))
sys.path.append(op.abspath(op.join(os.getcwd(), '..','..','python')))

# mesh path collection
from constants import MESHES_SHAPENET_DIR
from ref_img_utils import load_ref_images, gen_ref_imgs_from_mesh
fp_abs_ShapeNet = MESHES_SHAPENET_DIR
ids_ShapeNet = sorted(os.listdir(fp_abs_ShapeNet))
fn_abs_meshes_ShapeNet = [op.join(fp_abs_ShapeNet, id, "models/model_normalized.obj") \
                          for id in ids_ShapeNet]

resx = 256;resy = 256
from diff_mesh_utils import *
num_turn_imgs = 15
background = white_background = torch.ones((resx,resy, 3)).to(pyredner.get_device())
scale = torch.eye(3).to(pyredner.get_device())
azimuth_min = torch.tensor(torch.pi*0).cuda()
azimuth_max = torch.tensor(torch.pi*2).cuda()
azimuth = torch.linspace(azimuth_min, azimuth_max, num_turn_imgs+1, device = pyredner.get_device())
color_mode = DEFUALT_COLOR
slice_mode = False
from constants import  RENDER_DIR
from ref_img_utils import img2exr, img2png

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
sdf_target_pos = [0.5,0.5,0.5]
import copy

scene_name = 'LfD_base'
dryrun = False # 不创建文件夹
opt_config_name = 'principled-lfd-hq'
# render_spp = 500 # 越低图像噪点越多
# sdf_res = 256
render_spp = 768 # 越低图像噪点越多
sdf_res = 384
sdf_init_radius = 0.1 # 128[0.1] 64[0.15] 32[0.35]
render_upsample_iter = [400] # 前220 iter用的是scene_config.init_res，低resolution以加快初始变形速度
checkpoint_frequency = 0 # 每隔128 iters保存一次optimized shape
import diff_sdf_utils
# importlib.reload(diff_sdf_utils)
from diff_sdf_utils import sdfScene
from constants import SCENE_DIR
opt_scene_fn = op.join(SCENE_DIR, scene_name, f'{scene_name}.xml')
sdf_n_iter = 280
sdf_write_opt_images = False
weight_important = False
from ref_img_utils import load_ref_images

from marching_cube import MarchingCube
mc_scale = 0.01
mc_eu = [0.0, 0.0, 0.0]

args = sys.argv
input_values = args[1:]
fp_diff_mesh_head = input_values[0]
fp_sdf_out = input_values[1]
start_shapenet_id = int(input_values[2])
end_shapenet_id = int(input_values[3])
import time

haha = [
    # '15e651b8b7a0c880ac13edc49a7166b9',
 # '1941c37c6db30e481ef53acb6e05e27a',
 # '19e66890128da2d629647c2672d3167a',
 # '1dc0db78671ac363f99976ddcaae4d11',
 # '1f1e747bfce16fe0589ac2c115a083f1',
 # '247ca61022a4f47e8a94168388287ad5',
 '27f4207dce674969c3bd24f986301745',
 # '2d856dae6ae9373add5cf959690bb73f',
 # '2d8bb293b226dcb678c34bb504731cb9',
 # '3139001aa8148559bc3e724ff85e2d1b',
 # '38aa6c6632f64ad5fdedf0d8aa5213c',
 # '3a5351666689a7b2b788559e93c74a0f',
 # '3ae3a9b74f96fef28fe15648f042f0d9',
 # '3d81cebaa1226e9329fdbaf17f41d872',
 # '3f6d549265532efa2faef6cbf71ebacc',
 # '41391118c374e765ff3ef8f363b2cc43',
 # '49429e1d1e90c1ca202be79d8b285c1e',
 # '49943fa341bd0fc86bc258c0f0234be0',
 # '4a1f62dbe8b091eabc49cae1a831a9e',
 '4ad7d03773d3767c2bc52a80abcabb17',
 # '4d3308cc92ffdab432b72c6a3d82ffd6',
 # '4de2413a28db0137100e9bd5c4b0af54',
 # '514024b4a66a57d592ecd319dfd8c5d',
 # '563f955f46e4368d8ae69567bc5a3efb',
 # '58bc4edaefdf3b0e1db42cf774e49bf',
]

# for shapenet_id in range(start_shapenet_id, end_shapenet_id+1):
for str_id in haha:
    shapenet_id = ids_ShapeNet.index(str_id)
    print("----------[",shapenet_id,"]------------")
    print(">>",ids_ShapeNet[shapenet_id],"<<")
    _start_time = time.time()
    
    fn_s4_obj = op.join(fp_diff_mesh_head, ids_ShapeNet[shapenet_id], 'stage4', 'stage4_after_all_opts.obj')
    cam = CameraModel(resx=resx,resy=resy)
    cam.load_obj(fn_s4_obj)
    cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device())
    cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device())
    cam_up = torch.tensor([0., 1., 0.], device = pyredner.get_device())

    # diff-mesh
    cam_list = []
    ref_img = []
    weight_box = []
    zeros = torch.tensor([0., 0., 0.], device = pyredner.get_device())
    obj_pos = zeros - cam.center[-1]
    init_cam_pos = cam_pos - cam.center[-1]
    for i in range(num_turn_imgs):
        goal_euler_angles = zeros.clone()
        goal_euler_angles[1] += azimuth[i]
        neo_cam_pos, neo_cam_look_at, neo_cam_up = obj_to_cam(zeros, 
                                                               zeros, 
                                                               zeros, 
                                                               goal_euler_angles, 
                                                               init_cam_pos, cam_look_at, cam_up)
        img_with_texture = cam.set_aligned_obj(color_mode = color_mode,
                                                  slice_mode = slice_mode, 
                                                  scale = scale,
                                                  translation = obj_pos,
                                                  euler_angles = zeros,
                                                  cam_pos = neo_cam_pos, cam_look_at = neo_cam_look_at, cam_up = neo_cam_up,
                                                 )
        with_bg_img_with_texture = add_background(img_with_texture, background)
        with_bg_img_with_texture = torch.clamp(with_bg_img_with_texture, 0, 1)
        ref_img.append(with_bg_img_with_texture.cpu().detach())
        # plt.imshow(torch.pow(with_bg_img_with_texture, 1.0/2.2).cpu().detach());plt.show()
        cam_list.append((neo_cam_pos.cpu(), neo_cam_look_at.cpu(), neo_cam_up.cpu()))
        weight_box.append(1.0)

    # # zero 1 to 3
    # fp_syn_imgs = op.join(RENDER_DIR,'syn_imgs',ids_ShapeNet[shapenet_id])
    
    # need = [
    #     '0_0+0_0.png',
    #     '0_0+22_5.png',
    #     '0_0+-22_5.png',
    #     # '90+0.png',
    #     # '-90+0.png',
    # ]
    # zore_123_ref_imgs_path, zore_123_cam_pos_data = get_images(fp_syn_imgs, need, endwith='.png')
    # # cal cam
    # pyredner_cam_look_at = cam_look_at.tolist()
    # pyredner_cam_pos = init_cam_pos.tolist()
    # zore_123_cam_poses = calculate_camera_positions(zore_123_cam_pos_data,pyredner_cam_look_at,pyredner_cam_pos)
    # for cam_info in zore_123_cam_poses:
    #     neo_cam_pos = torch.tensor(cam_info)
    #     cam_list.append((neo_cam_pos.cpu(), cam_look_at.cpu(), cam_up.cpu()))

    # set an output file
    fp_output = op.join(fp_sdf_out, ids_ShapeNet[shapenet_id])
    fp_ref_output = op.join(fp_output,'ref')
    os.makedirs(fp_ref_output, exist_ok=True)
    # save ref_img .exr and .png
    # for image_file in zore_123_ref_imgs_path:
    #     img = pyredner.imread(image_file)
    #     ref_img.append(img.cpu())
    #     weight_box.append(5.0)
    ref_image_paths = []
    for idx in range(len(ref_img)):
        fn_img_out = os.path.join(fp_ref_output,f"ref-{idx:02d}.exr")
        ref_image_paths.append(copy.deepcopy(fn_img_out))
        img2exr(ref_img[idx], fn_img_out)
        fn_img_out = os.path.join(fp_ref_output,f"ref-{idx:02d}.png")
        img2png(ref_img[idx], fn_img_out)
    # save cam info
    torch.save((cam_list),op.join(fp_ref_output,'cam_info.pt'))

    # set suba sensor
    my_sensors = []
    ref_cam_list = copy.deepcopy(cam_list)
    for i in range(len(cam_list)):
        ref_cam_pos, ref_cam_look_at, ref_cam_up = ref_cam_list[i]
        for idx in range(3):
            ref_cam_pos[idx] += sdf_target_pos[idx]
            ref_cam_look_at[idx] += sdf_target_pos[idx]
        my_sensors.append(mi.load_dict({
                    'type': 'perspective',
                    'fov': 45.0,
                    'to_world': mi.ScalarTransform4f.look_at(target=ref_cam_look_at.tolist(), origin=ref_cam_pos.tolist(), up=ref_cam_up.tolist()),
                    'sampler': {'type': 'independent'},
                    'film': {
                        'type': 'hdrfilm', 
                        'width': resx, 'height': resy,
                        'filter': {'type': 'gaussian'},
                        'sample_border': True,
                    }
                }))
    
    # diff sdf
    sdf_opt_ref_imgs = load_ref_images(ref_image_paths, multiscale = True)
    verbose = []
    diff_sdf_scene =  sdfScene(scene_name, verbose, opt_config_name, dryrun = dryrun,
                           render_upsample_iter = render_upsample_iter,
                           checkpoint_frequency = checkpoint_frequency,
                           resx = resx,resy = resy,sdf_res=sdf_res,sdf_init_radius=sdf_init_radius,
                           render_spp = render_spp, my_sensors = my_sensors,
                           dir_endswith = ids_ShapeNet[shapenet_id])
    diff_sdf_scene.load_sdf_scene(opt_scene_fn)
    diff_sdf_scene.current_output_dir = op.join(fp_output,'sdf_output')
    diff_sdf_scene.sdf_shape_opt(sdf_opt_ref_imgs,0,
                                 n_iter = sdf_n_iter,out_iters = 20,
                                 update_sensors = True, my_sensors = my_sensors,
                                 weight_important=weight_important, weight_box = weight_box,
                                save_load_img = False,write_opt_images=sdf_write_opt_images)
    
    sdf_filename = diff_sdf_scene.get_vol_fn(0)[0]
    opt_mesh_dir = diff_sdf_scene.get_mesh_dir()
    try:
        fn_save_obj = op.join(opt_mesh_dir,f'mc_{0:03d}.obj')
        MarchingCube(sdf_filename, mc_scale, mc_eu).save_obj(fn_save_obj)
    except:
        pass
    try:
        diff_sdf_scene.gen_render_turntable()
        string = opt_mesh_dir
        input_folder = string.replace('mesh', 'turntable')
        fn_save_gif = string.replace('mesh', 'vid.gif')
        gen_gif(input_folder,fn_save_gif)
    except:
        pass