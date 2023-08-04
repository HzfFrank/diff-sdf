"""utils for reference image"""

"""---------------------------------------------------------------------------------
[mitsuba] compute scale pyramid for GT 
    """
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
# Initialize the image filters used to resample images
GAUSSIAN_RFILTER = mi.scalar_rgb.load_dict({'type': 'gaussian'})
BOX_RFILTER = mi.scalar_rgb.load_dict({'type': 'box'})
def resize_img(img, target_res, smooth=False):
    """Resizes a Mitsuba Bitmap using either a box filter (smooth=False)
       or a gaussian filter (smooth=True)"""
    assert isinstance(img, mi.Bitmap)
    source_res = img.size()
    if target_res[0] == source_res[0] and target_res[1] == source_res[1]:
        return img
    rfilter = GAUSSIAN_RFILTER if smooth else BOX_RFILTER
    return img.resample([target_res[1], target_res[0]], rfilter)

def show_pyramid(image_list):
    import matplotlib.pyplot as plt
    n = len(image_list)
    fig, axes = plt.subplots(1, n, figsize=(n*4, 4))
    for i, image in enumerate(image_list):
        if isinstance(image, str):
            image = plt.imread(image)
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def cal_scale_pyramid(bmp, SHOW = False, SAVE = False, fp_output = None):
    d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
    new_res = bmp.size()
    show = []
    while np.min(new_res) > 4:
        if SHOW:
            show.append(d[int(new_res[0])].torch().cpu().numpy())
        if SAVE:
            fn_ = os.path.join(fp_output, 'ref-' + fn[-6:-4]+"-"+str(new_res)+'.png')
            mi.util.write_bitmap(fn_, d[int(new_res[0])][..., :3], write_async=False)
        new_res = new_res // 2
        d[int(new_res[0])] = mi.TensorXf(resize_img(bmp, new_res, smooth=True))
    if SHOW:
        show_pyramid(show)
    return d

def load_ref_images(paths, multiscale=False):
    """Load the reference images and compute scale pyramid for multiscale loss"""
    if not multiscale:
        return [mi.TensorXf(mi.Bitmap(fn)) for fn in paths]
    result = []
    for fn in paths:
        bmp = mi.Bitmap(fn)
        result.append(cal_scale_pyramid(bmp))
    return result

"""---------------------------------------------------------------------------------
[pyredner] gen GT 
    """
import pyredner
import torch
import numpy as np
import OpenEXR
import Imath
import imageio
import os
import copy
from PIL import Image
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_print_timing(False)

def convert_to_exr(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]
    
    for image_file in image_files:
        fn_out = os.path.join(output_folder, os.path.basename(image_file).replace('.png', '.exr'))
        img = pyredner.imread(image_file)
        img2exr(img, fn_out)
        
def gen_gif(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]
    torch.stack(with_bg_rgb_imgs)
    for image_file in image_files:
        fn_out = os.path.join(output_folder, os.path.basename(image_file).replace('.png', '.exr'))
        img = pyredner.imread(image_file)
        img2exr(img, fn_out)

def gen_ref_imgs_from_mesh(fn_model,fp_output, move_cam = True,white_bg=True,
                           imgs_num = 10, resx = 256, resy = 256 , num_samples = 1024,
                           format = 'exr', gif_name = 'video.gif', force=False):
    # gen vid and ref-img(default: exr)
    if move_cam:
        if os.path.isfile(os.path.join(fp_output,gif_name)) and not force:
            print('ref-files exist')
            ref_image_paths = sorted([os.path.abspath(os.path.join(fp_output,file)) \
                                  for file in os.listdir(fp_output) if file.endswith('.'+format)])
        else:
            os.makedirs(fp_output, exist_ok=True)
            ref_image_paths = move_cam_gen_vid(fn_model, fp_output, 
                                                 resx = resx, resy = resy, 
                                                 imgs_num = imgs_num,
                                                 num_samples = num_samples,
                                                 white_bg = white_bg,
                                                 DATA = True,
                                                 EXR = (format=='exr'),
                                                 PNG = (format=='png'),
                                                 VID = True)
            print("gen ref-img <DONE>")
    else:
        if os.path.isfile(os.path.join(fp_output,gif_name)) and not force:
            print('ref-files exist')
            ref_image_paths = sorted([os.path.abspath(os.path.join(fp_output,file)) \
                                  for file in os.listdir(fp_output) if file.endswith('.'+format)])
            ref_mask_paths = sorted([os.path.abspath(os.path.join(fp_output,file)) \
                                  for file in os.listdir(fp_output) if file.endswith('.png') and file.startswith('mask')])
        else:
            os.makedirs(fp_output, exist_ok=True)
            ref_image_paths, ref_mask_paths = gen_vid(fn_model, fp_output, 
                                                     resx = resx, resy = resy, 
                                                     imgs_num = imgs_num,
                                                     num_samples = num_samples,
                                                     white_bg = white_bg,
                                                     DATA = True,
                                                     EXR = (format=='exr'),
                                                     PNG = (format=='png'),
                                                     VID = True)
            print("gen ref-img <DONE>")
    return ref_image_paths, ref_mask_paths
    

def img2exr(img, fn_out):
    image_np = img.cpu().numpy()
    
    # Normalize the image values between 0 and 1
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

    # Convert the numpy array to an EXR format
    image_exr = np.float16(image_np).transpose(2, 0, 1)
    height, width = image_exr.shape[1], image_exr.shape[2]
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    }

    # Save the image in EXR format
    out_file = OpenEXR.OutputFile(fn_out, header)
    out_file.writePixels({'R': image_exr[0].tobytes(), 'G': image_exr[1].tobytes(), 'B': image_exr[2].tobytes()})
    out_file.close()

def img2png(img, fn_out):
    image_np = img.cpu().numpy()

    # Normalize the image values between 0 and 1
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

    # Convert the numpy array to the format expected by imageio (uint8 and range 0-255)
    image_uint8 = (image_np * 255).astype(np.uint8)

    # Save the image in PNG format
    imageio.imwrite(fn_out, image_uint8)

def rdImgs2Gif(fn_out, rgb_imgs, img_is_numpy = False):
    if img_is_numpy:
        rgb_imgs_uint8 = (rgb_imgs[:, :, :, :3] * 255).astype(np.uint8)
    else:
        rgb_imgs_uint8 = (torch.pow(rgb_imgs[:, :, :, :3], 1.0/2.2).cpu().numpy() * 255).astype(np.uint8)
        # check_render_img(rgb_imgs_uint8, img_is_numpy = True)
    imageio.mimsave(fn_out, rgb_imgs_uint8, loop=0)
    return
def check_render_img(rgb_imgs, img_is_numpy = False):
    if img_is_numpy:
        for i in range(len(rgb_imgs)):
            plt.imshow(rgb_imgs[i]); plt.show()
    else:
        for i in range(len(rgb_imgs)):
            plt.imshow(torch.pow(rgb_imgs[i], 1.0/2.2).detach().cpu()); plt.show()
def gen_3D_data(dx_num_start, dy_num_start, dz_num_start, 
                beta_x=1.0, beta_y=1.0, beta_z=1.0, pos = False,
                imgs_num = 15):
    random_x = np.random.uniform(1., 2.)*beta_x
    random_sign_x = np.random.choice([-1, 1])
    random_x = random_x*random_sign_x
    random_y = np.random.uniform(1., 2.)*beta_y
    random_sign_y = np.random.choice([-1, 1])
    random_y = random_y*random_sign_y
    random_z = np.random.uniform(1., 2.)*beta_z
    random_sign_z = np.random.choice([-1, 1])
    random_z = random_z*random_sign_z
    dx_num_end = dx_num_start + random_x
    dy_num_end = dy_num_start + random_y
    dz_num_end = dz_num_start + random_z
    if pos:
        def sample_spherical(N, center, radius, device="cuda"):
                torch.manual_seed(0)
                polar = torch.rand(N, device=device) * torch.pi
                azimuth = torch.rand(N, device=device) * 2 * torch.pi
            
                x = radius * torch.sin(polar) * torch.cos(azimuth) + center[0]
                y = radius * torch.sin(polar) * torch.sin(azimuth) + center[1]
                z = radius * torch.cos(polar) + center[2]
            
                return torch.stack((x, y, z), dim=1)
        tensor1 = torch.tensor([dx_num_start,dy_num_start,dz_num_start]).cuda()
        tensor2 = torch.tensor([0.5,0.5,0.5]).cuda()
        radius = torch.norm(tensor2 - tensor1)
        cam_pos_candidates = sample_spherical(1, tensor2, radius)
        dx_num_end = cam_pos_candidates[0][0].item()
        dy_num_end = cam_pos_candidates[0][1].item()
        dz_num_end = cam_pos_candidates[0][2].item()
    dx = torch.linspace(dx_num_start, dx_num_end, imgs_num)
    dy = torch.linspace(dy_num_start, dy_num_end, imgs_num)
    dz = torch.linspace(dz_num_start, dz_num_end, imgs_num)
    return dx, dy, dz

def gen_pos_data(dx_num_start, dy_num_start, dz_num_start, 
                radius=1.0, imgs_num=15, device="cuda"):
    torch.manual_seed(0)
    # Convert the start point to spherical coordinates
    center = torch.tensor([dx_num_start, dy_num_start, dz_num_start], device=device)
    r = torch.norm(center)
    polar_start = torch.acos(center[2]/r)
    azimuth_start = torch.atan2(center[1], center[0])

    # Generate a series of continuous polar and azimuth angles
    d_polar = torch.linspace(0, torch.pi, imgs_num, device=device)
    d_azimuth = torch.linspace(0, 2*torch.pi, imgs_num, device=device)

    # To avoid all trajectories converging to the same point at the poles,
    # add a small random offset near the poles
    near_poles = ((d_polar < 0.1) | (d_polar > torch.pi - 0.1))
    d_azimuth[near_poles] += torch.rand_like(d_azimuth[near_poles]) * 0.2 - 0.1

    # Add the start angles to the differentials
    polar = (polar_start + d_polar) % (2*torch.pi)
    azimuth = (azimuth_start + d_azimuth) % (2*torch.pi)

    # Convert back to Cartesian coordinates
    x = radius * torch.sin(polar) * torch.cos(azimuth)
    y = radius * torch.sin(polar) * torch.sin(azimuth)
    z = radius * torch.cos(polar)
    
    return x, y, z

def gen_vid(fn_model, fp_output, 
             resx = 256, resy = 256, 
             imgs_num = 10,
             num_samples = 256,
             white_bg = True,
             DATA = True,
             EXR = True,
             PNG = False,
             VID = True,fn_env_map = None,Only3D = False):
    # load objects and cal center
    objects = pyredner.load_obj(fn_model, return_objects=True)
    vertices = []
    for obj in objects:
        vertices.append(obj.vertices.clone())
    center = torch.mean(torch.cat(vertices), 0)
    
    # set camera
    camera = pyredner.automatic_camera_placement(objects, resolution=(resx, resy))
    camera.position += torch.tensor([0,0,-1.])
    pos_info = {}
    pos_info['sensor'] = {}
    for key in camera.__dict__.keys():
        info = camera.__dict__[key]
        if isinstance(info, torch.Tensor):
            pos_info['sensor'][key] = info.cpu()
        else:
            pos_info['sensor'][key] = info
        # print(key,info)
    
    # gen different scene from different angles
    scenes = []
    # scenes.append(pyredner.Scene(camera = camera, objects = objects))
    
    random_x = np.random.uniform(60, 90)
    random_sign = np.random.choice([-1, 1])
    random_x = random_x*random_sign
    random_y = np.random.uniform(180, 359)
    random_sign = np.random.choice([-1, 1])
    random_y = random_y*random_sign
    random_z = np.random.uniform(60, 90)
    random_sign = np.random.choice([-1, 1])
    random_z = random_z*random_sign
    dx = torch.linspace(torch.deg2rad(torch.tensor(30)).item(), 
                        torch.deg2rad(torch.tensor(30 + random_x)).item(),
                        imgs_num)
    dy = torch.linspace(torch.deg2rad(torch.tensor(45)).item(), 
                        torch.deg2rad(torch.tensor(45 + random_y)).item(),
                        imgs_num)
    dz = torch.linspace(torch.deg2rad(torch.tensor(30)).item(), 
                        torch.deg2rad(torch.tensor(30 + random_z)).item(),
                        imgs_num)
    
    if Only3D:
        beta = 2
    else:
        beta = 1
    random_x = np.random.uniform(0.1, 0.3)*beta
    random_sign = np.random.choice([-1, 1])
    random_x = random_x*random_sign
    random_y = np.random.uniform(0.2, 0.4)*beta
    random_sign = np.random.choice([-1, 1])
    random_y = random_y*random_sign
    random_z = np.random.uniform(0.15, 0.35)*beta
    random_sign = np.random.choice([-1, 1])
    random_z = random_z*random_sign
    Dx = torch.linspace(0, random_x, imgs_num)
    Dy = torch.linspace(0, random_y, imgs_num)
    Dz = torch.linspace(0, random_z, imgs_num)
    
    # set 6D pos
    obj_pos = []
    
    # set env-map
    if fn_env_map is not None:
        envmap_img = pyredner.imread(fn_env_map).cuda()
        # Downsample the environment map
        target_res = [int(envmap_img.shape[0] / 4), int(envmap_img.shape[1] / 4)]
        img = envmap_img.permute(2, 0, 1) # HWC -> CHW
        img = img.unsqueeze(0) # CHW -> NCHW
        img = torch.nn.functional.interpolate(img, size = target_res, mode = 'area')
        img = img.squeeze(dim = 0) # NCHW -> CHW
        img = img.permute(1, 2, 0)
        envmap_img_lowres = img
        envmap = pyredner.EnvironmentMap(envmap_img_lowres)
    
    def scene_model(translation, euler_angles):
        rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
        for obj, v in zip(objects, vertices):
            obj.vertices = (v - center) @ torch.t(rotation_matrix) + center + translation
        if fn_env_map is not None:
            return pyredner.Scene(camera = camera, objects = objects, envmap=envmap)
        return pyredner.Scene(camera = camera, objects = objects)
        
    for i in range(imgs_num):
        XYZ = torch.tensor([Dx[i], Dy[i], Dz[i]]).cuda()
        if Only3D:
            EA = torch.tensor([0., 0., 0.]).cuda()
        else:
            EA = torch.tensor([dx[i], dy[i], dz[i]]).cuda()
        scenes.append(scene_model(XYZ,EA))
        obj_pos.append({'xyz':XYZ.cpu(),'euler_angle':EA.cpu()})
    pos_info['obj'] = obj_pos
    if DATA:
        torch.save(pos_info, os.path.join(fp_output,"pose_info.pt"))
        
    if fn_env_map is not None:
        rgb_imgs = pyredner.render_pathtracing(scenes, num_samples = (resx, 6))
    else:
        rgb_imgs = pyredner.render_albedo(scenes, alpha=white_bg, num_samples = (num_samples, 0))
        # get mask
        for sc in scenes:
            for i in range(len(sc.materials)):
                sc.materials[i].use_vertex_color = True # that only set to ture and don't assign a value will lead to black
        mask_imgs = pyredner.render_albedo(scenes, alpha=white_bg, num_samples = (num_samples, 0))
        if white_bg:
            with_bg_rgb_imgs = []
            with_bg_mask_imgs = []
            def add_background(_img,background_img):
                alpha = _img[:, :, 3:4]
                with_bg_img = _img[:, :, :3] * alpha + background_img * (1 - alpha)
                return with_bg_img
            white_background = torch.ones((resx,resy, 3)).to(pyredner.get_device())
            for idx in range(imgs_num):
                with_bg_rgb_imgs.append(add_background(rgb_imgs[idx], white_background))
                with_bg_mask_imgs.append(add_background(mask_imgs[idx], white_background))
            rgb_imgs = torch.stack(with_bg_rgb_imgs)
            mask_imgs = torch.stack(with_bg_mask_imgs)
    
    ref_image_paths = []
    ref_mask_paths = []
    
    for idx in range(imgs_num):
        if EXR:
            fn_img_out = os.path.join(fp_output,f"ref-{idx:02d}.exr")
            img2exr(rgb_imgs[idx], fn_img_out)
        elif PNG:
            fn_img_out = os.path.join(fp_output,f"ref-{idx:02d}.png")
            img2png(rgb_imgs[idx], fn_img_out)
        else: break
        ref_image_paths.append(fn_img_out)
        
        fn_mask_img_out = os.path.join(fp_output,f"mask-{idx:02d}.png")
        img2png(mask_imgs[idx], fn_mask_img_out)
        ref_mask_paths.append(fn_mask_img_out)
        
    if VID:
        fn_gif_out = os.path.join(fp_output,'video.gif')
        rdImgs2Gif(fn_gif_out,rgb_imgs)

    return ref_image_paths, ref_mask_paths

def move_cam_gen_vid(fn_model, fp_output, 
                     resx = 256, resy = 256, 
                     imgs_num = 10,
                     num_samples = 1024,
                     white_bg = True,
                     DATA = True,
                     EXR = False,
                     PNG = True,
                     VID = True):
    objects = pyredner.load_obj(fn_model, return_objects=True)
    vertices = []
    for obj in objects:
        vertices.append(obj.vertices.clone())
    center = torch.mean(torch.cat(vertices), 0)
    def set_obj_6D(translation, euler_angles):
        rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
        for obj, v in zip(objects, vertices):
            obj.vertices = (v - center) @ torch.t(rotation_matrix) + center + translation
        return objects
    EA = torch.tensor([0,0,0.]).cuda()
    XYZ = torch.tensor([0.5,0.5,0.5]).cuda()
    objects = set_obj_6D( XYZ,EA)
    camera = pyredner.automatic_camera_placement(objects, resolution=(resx, resy))
    camera.position += torch.tensor([ -2.0,  0.5, -0.])
    init_cam_pos = camera.position
    init_cam_look_at = camera.look_at
    init_cam_up = camera.up 

    tensor1 = init_cam_pos.cuda()
    tensor2 = torch.tensor([0.5,0.5,0.5]).cuda()
    radius = torch.norm(tensor2 - tensor1).item()
                         
    pos_dx, pos_dy, pos_dz = gen_pos_data(init_cam_pos[0].item(),
                                     init_cam_pos[1].item(),
                                     init_cam_pos[2].item(),
                                     radius = radius,
                                     imgs_num = imgs_num + 2) # +2 是为了避免最后一帧太突兀
    look_at_dx, look_at_dy, look_at_dz = gen_3D_data(init_cam_look_at[0].item(),
                                                     init_cam_look_at[1].item(),
                                                     init_cam_look_at[2].item(),
                                                     beta_x=0.3, beta_y=0.3, beta_z=0.3,
                                                     imgs_num = imgs_num+2)
    up_dx, up_dy, up_dz = gen_3D_data(init_cam_up[0].item(),
                                      init_cam_up[1].item(),
                                      init_cam_up[2].item(),
                                      beta_x=0.2, beta_y=0.2, beta_z=0.2,
                                      imgs_num = imgs_num+2)                        
    scenes = []
    cameras_info = {}
    for i in range(imgs_num):
        camera.position = torch.tensor([pos_dx[i], pos_dy[i], pos_dz[i]]).cuda()
        camera.look_at = torch.tensor([look_at_dx[i], look_at_dy[i], look_at_dz[i]]).cuda()
        camera.up = torch.tensor([up_dx[i], up_dy[i], up_dz[i]]).cuda()
        scenes.append(pyredner.Scene(camera = copy.deepcopy(camera), objects = objects))# copy:important for batch rendering
        cameras_info[str(i)] = {
            'cam_pos' : camera.position.cpu().tolist(),
            'cam_look_at' : camera.look_at.cpu().tolist(),
            'cam_up' : camera.up.cpu().tolist(),
            'cam_fov' : camera._fov.cpu().tolist(),
            'cam_resx' : camera.resolution[0],
            'cam_resy' : camera.resolution[1],
        }
    
    # batch rendering
    rgb_imgs = pyredner.render_albedo(scenes,alpha=white_bg,num_samples = (num_samples, 0))
    if DATA:
        import json
        file_path = os.path.join(fp_output,"cameras_info.json")
        with open(file_path, "w") as json_file:
            json.dump(cameras_info, json_file)
            
    ref_image_paths = []
    for idx in range(imgs_num):
        if EXR:
            fn_img_out = os.path.join(fp_output,f"ref-{idx:02d}.exr")
            img2exr(rgb_imgs[idx], fn_img_out)
        elif PNG:
            fn_img_out = os.path.join(fp_output,f"ref-{idx:02d}.png")
            img2png(rgb_imgs[idx], fn_img_out)
        else: break
        ref_image_paths.append(fn_img_out)
    if VID:    
        # save video.gif
        fn_gif_out = os.path.join(fp_output,'video.gif')
        rdImgs2Gif(fn_gif_out,rgb_imgs)
    return ref_image_paths
                
    