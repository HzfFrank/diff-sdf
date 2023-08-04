import os
import sys
import os.path as op
sys.path.append(op.abspath(op.join(os.getcwd(), '..')))
sys.path.append(op.abspath(op.join(os.getcwd(), '..','..','python')))

import torch
import pyredner
import matplotlib.pyplot as plt
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_print_timing(False)

import numpy as np
import math
"""---------------------------------------------------------------------------------
gen sphere, Adapted from pyredner/utils.py
    """
def gen_sphere(theta_steps: int,
               phi_steps: int,
               radius:float,
               need_uv = True):
    """
        Generate a triangle mesh representing a UV sphere,
        center at (0, 0, 0) with radius 1.
        Args
        ====
        theta_steps: int
            zenith subdivision
        phi_steps: int
            azimuth subdivision
    """
    d_theta = math.pi / (theta_steps - 1) #Angle difference in longitudinal direction between each two layers
    d_phi = (2 * math.pi) / (phi_steps - 1)

    num_vertices = theta_steps * phi_steps - 2 * (phi_steps - 1)
    vertices = np.zeros([num_vertices, 3], dtype = np.float32)
    if need_uv:
        uvs = np.zeros([num_vertices, 2], dtype = np.float32)
    vertices_index = 0
    for theta_index in range(theta_steps):
        sin_theta = radius*math.sin(theta_index * d_theta)
        cos_theta = radius*math.cos(theta_index * d_theta)
        if theta_index == 0:
            # For the two polars of the sphere, only generate one vertex
            vertices[vertices_index, :] = \
                np.array([0.0, radius, 0.0], dtype = np.float32)
            if need_uv:
                uvs[vertices_index, 0] = 0.0
                uvs[vertices_index, 1] = 0.0
            vertices_index += 1
        elif theta_index == theta_steps - 1:
            # For the two polars of the sphere, only generate one vertex
            vertices[vertices_index, :] = \
                np.array([0.0, (-1) * radius, 0.0], dtype = np.float32)
            if need_uv:
                uvs[vertices_index, 0] = 0.0
                uvs[vertices_index, 1] = 1.0
            vertices_index += 1
        else:
            for phi_index in range(phi_steps):
                sin_phi = math.sin(phi_index * d_phi)
                cos_phi = math.cos(phi_index * d_phi)
                vertices[vertices_index, :] = \
                    np.array([sin_theta * cos_phi, cos_theta, sin_theta * sin_phi])
                if need_uv:
                    uvs[vertices_index, 0] = phi_index * d_phi / (2 * math.pi)
                    uvs[vertices_index, 1] = theta_index * d_theta / math.pi
                vertices_index += 1

    indices = []
    for theta_index in range(1, theta_steps):
        for phi_index in range(phi_steps - 1):
            if theta_index < theta_steps - 1:
                id0 = phi_steps * theta_index + phi_index - (phi_steps - 1)
                id1 = phi_steps * theta_index + phi_index + 1 - (phi_steps - 1)
            else:
                # There is only one vertex at the pole
                assert(theta_index == theta_steps - 1)
                id0 = num_vertices - 1
                id1 = num_vertices - 1
            if theta_index > 1:
                id2 = phi_steps * (theta_index - 1) + phi_index - (phi_steps - 1)
                id3 = phi_steps * (theta_index - 1) + phi_index + 1 - (phi_steps - 1)
            else:
                # There is only one vertex at the pole
                assert(theta_index == 1)
                id2 = 0
                id3 = 0

            if (theta_index < theta_steps - 1):
                indices.append([id0, id2, id1])
            if (theta_index > 1):
                indices.append([id1, id2, id3])

    indices = torch.tensor(indices, dtype=torch.int32)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    if need_uv:
        uvs = torch.tensor(uvs, dtype=torch.float32)
    normals = torch.clone(vertices)
    if need_uv:
        return (vertices, indices, uvs, normals)
    else:
        return (vertices, indices, normals)

import torch.nn as nn
import copy
import open3d as o3d
class CameraModel(nn.Module):
    def __init__(self, resx=256, resy=256):
        super().__init__()
        # set color mode
        self.DEFUALT_COLOR, self.V_COLOR, self.USE_UV_MAP = 0, 1, 2

        self.resx = resx
        self.resy = resy
        self.up = torch.tensor([0.,1.,0.], device = pyredner.get_device())
        self.fov = torch.tensor([45.0], device = pyredner.get_device())
        self.clip_near = 1e-10
        
        self.object_list = []
        self.obj_load_path = []
        self.center = []
        self.init_material = pyredner.Material(diffuse_reflectance = torch.tensor((0., 0., 0.), device = pyredner.get_device()))
        
        self.scene_backup = {}
        self.normals_backup = {}
        self.uvs_backup = {}
        self.uv_indices_backup = {}
    """---------------------------------------------------------------------------------
    load .obj file
        1. align_obj
        2. load_obj
        3. save_obj_in_setted_scene
    """
    def align_obj(self,obj_list):
        #-# --------------------Vertices
        save_vertices = obj_list[0].vertices
        for i in range(1,len(obj_list)):
            save_vertices = torch.cat((save_vertices, obj_list[i].vertices),dim=0)
        #-# --------------------Indices
        save_indices =  obj_list[0].indices
        v_num = 0
        for i in range(1,len(obj_list)):
            v_num += (obj_list[i-1].vertices.shape[0])
            save_indices = torch.cat((save_indices, (obj_list[i].indices + v_num)),dim=0)
        obj = pyredner.Object(vertices = save_vertices,
                                  indices = save_indices, 
                                  uvs = None, 
                                  normals = None,
                                  material = self.init_material)
        return obj
        
    def load_obj(self,load_file):
        self.obj_load_path.append(load_file)
        new_obj = pyredner.load_obj(load_file, return_objects=True)
        if len(new_obj) > 1:
            new_obj = self.align_obj(new_obj)
        if type(new_obj) is list:
            new_obj = new_obj[0]
        self.object_list.append(new_obj)
        self.center.append(torch.mean(self.object_list[-1].vertices.clone().cuda(), 0))
        return
        
    def save_obj_in_setted_scene(self, fn_save, obj_idx = 0):
        obj = pyredner.Object(vertices = self.scene.shapes[obj_idx].vertices.detach(),
                              indices = self.scene.shapes[obj_idx].indices.detach(), 
                              
                              uvs = self.scene.shapes[obj_idx].uvs.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].uvs, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uvs, 
                              
                              uv_indices = self.scene.shapes[obj_idx].uv_indices.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].uv_indices, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uv_indices, 
                              
                              normals = self.scene.shapes[obj_idx].normals.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].normals, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uv_indices, 
                              
                              material = self.scene.materials[obj_idx].detach() \
                              
                                      if isinstance(self.scene.materials[obj_idx], torch.Tensor) \
                              
                                      else self.scene.materials[obj_idx], 
                             )
        pyredner.save_obj(obj, fn_save)
        return

    def save_obj_in_list(self, fn_save, obj_idx = -1):
        pyredner.save_obj(self.object_list[obj_idx], fn_save)
        return
    """---------------------------------------------------------------------------------
    set obj
        1. cart2sph
        2. cal_sphere_slice_region
        3. get_v
        4. get_v_from_scene 
        5. set_obj
        6. set_aligned_obj
        7. get_object_list
    """  
    def cal_sphere_slice_region(self, longi_and_lati=False, latitude_slice_num=0, longitude_slice_num=0):
        self.cart2sph()
        segments_latitude = torch.linspace(0, np.pi, latitude_slice_num + 1).cuda()
        self.sphere_region_mask = []
        
        if longi_and_lati:  # The slice mode is latitude and longitude segmentation
            segments_longitude = torch.linspace(-np.pi, np.pi, longitude_slice_num + 1).cuda()
            for m in range(longitude_slice_num):
                for n in range(latitude_slice_num):
                    idx = m * latitude_slice_num + n
                    start_longitude = segments_longitude[m]
                    end_longitude = segments_longitude[m + 1]
                    start_latitude = segments_latitude[n]
                    end_latitude = segments_latitude[n + 1]
                    mask_longitude = (self.sphere_phi >= start_longitude) & (self.sphere_phi < end_longitude)
                    mask_latitude = (self.sphere_theta >= start_latitude) & (self.sphere_theta < end_latitude)
                    mask = mask_longitude & mask_latitude
                    # Remove overlapping regions from the current mask
                    for existing_mask in self.sphere_region_mask:
                        mask = mask & ~existing_mask
                    self.sphere_region_mask.append(mask)
            self.sphere_region_num = latitude_slice_num * longitude_slice_num
        else:  # slice mode is latitude slice
            for n in range(latitude_slice_num):
                start_latitude = segments_latitude[n]
                end_latitude = segments_latitude[n + 1]
                mask_latitude = (self.sphere_theta >= start_latitude) & (self.sphere_theta < end_latitude)
                mask = mask_latitude
                # Remove overlapping regions from the current mask
                for existing_mask in self.sphere_region_mask:
                    mask = mask & ~existing_mask
                self.sphere_region_mask.append(mask)
            self.sphere_region_num = latitude_slice_num
    
    def cart2sph(self): # 标准球体生成的
        xyz, _, _ = gen_sphere(theta_steps = 16, phi_steps = 16 , radius = 1., need_uv =False)
        xyz = xyz.cuda()
        self.xyz = xyz
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        self.sphere_theta = theta
        self.sphere_phi = phi
        
    def set_obj(self,
                color = None, color_mode = 0, update_uv_coordinate = True, 
                slice_mode = False, 
                scale = torch.eye(3).to(pyredner.get_device()), standard_scale_mode = True,
                translation = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device()),
                cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                cam_up = torch.tensor([0., 1., 0.], device = pyredner.get_device()),
                obj_idx = 0, use_backup = False, return_img = True, no_set = False,
               ):   
        """
        only set one obj
        """
        # 注意deepcopy的使用： objects = copy.deepcopy(self.object_list)
        # 不要重复定义含有grad运算的单个变量
        if obj_idx == -1: 
            obj_idx = len(self.object_list) - 1 
        
        # set camera
        self.camera = pyredner.Camera(position = cam_pos,
                                   look_at = cam_look_at,
                                   up = cam_up,
                                   fov = self.fov,
                                   clip_near = self.clip_near,
                                   resolution = (self.resx, self.resy))
                   
        # set scene
        self.scene = pyredner.Scene(camera = self.camera, objects = copy.deepcopy(self.object_list)) 
        # self.scene = pyredner.Scene(camera = self.cam, objects = self.object_list)
        # self.init_object_list = copy.deepcopy(self.object_list)

        if no_set:
            img = pyredner.render_albedo(scene = self.scene, alpha = True)
            return img

        # 6-D and scale
        if slice_mode is False:
            rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
            if standard_scale_mode is True:
                scale = torch.diag(scale.diag())
            self.scene.shapes[obj_idx].vertices = ((self.scene.shapes[obj_idx].vertices  - self.center[obj_idx]) @ scale @ torch.t(rotation_matrix) \
                                             + self.center[obj_idx]) + translation
        else:
            if standard_scale_mode is True:
                for idx in range(self.sphere_region_num):
                    mask = self.sphere_region_mask[idx]
                    self.scene.shapes[obj_idx].vertices[mask] = (self.scene.shapes[obj_idx].vertices[mask] - self.center[obj_idx]) \
                                                            @ torch.diag(scale[idx].clone().diag()) 
                self.scene.shapes[obj_idx].vertices = self.scene.shapes[obj_idx].vertices @ torch.t(pyredner.gen_rotate_matrix(euler_angles)) \
                                                + self.center[obj_idx] + translation
            else:
                for idx in range(self.sphere_region_num):
                    mask = self.sphere_region_mask[idx]
                    self.scene.shapes[obj_idx].vertices[mask] = (self.scene.shapes[obj_idx].vertices[mask]  - self.center[obj_idx]) \
                                                            @ scale[idx] \
                                                            @ torch.t(pyredner.gen_rotate_matrix(euler_angles[idx])) \
                                                            + self.center[obj_idx] + translation[idx]
            
        # normals & color & uvs
        if color_mode is self.DEFUALT_COLOR:
            self.scene.materials[obj_idx].use_vertex_color = False
        else:
            self.scene.materials[obj_idx].use_vertex_color = True
            if color_mode is self.V_COLOR:
                self.scene.shapes[obj_idx].colors = color
            elif color_mode is self.USE_UV_MAP:
                if update_uv_coordinate:
                    self.scene.shapes[obj_idx].normals = pyredner.compute_vertex_normal(self.scene.shapes[obj_idx].vertices,
                                                                  self.scene.shapes[obj_idx].indices)
                    self.scene.shapes[obj_idx].uvs, self.scene.shapes[obj_idx].uv_indices = pyredner.compute_uvs(self.scene.shapes[obj_idx].vertices,
                                                                                                     self.scene.shapes[obj_idx].indices,
                                                                                                     print_progress=False)
                    self.normals_backup[str(obj_idx)] = self.scene.shapes[obj_idx].normals
                    self.uvs_backup[str(obj_idx)] = self.scene.shapes[obj_idx].uvs
                    self.uv_indices_backup[str(obj_idx)] = self.scene.shapes[obj_idx].uv_indices
                else:
                    self.scene.shapes[obj_idx].normals = self.normals_backup[str(obj_idx)]
                    self.scene.shapes[obj_idx].uvs = self.uvs_backup[str(obj_idx)]
                    self.scene.shapes[obj_idx].uv_indices = self.uv_indices_backup[str(obj_idx)]
                self.scene.materials[obj_idx] = pyredner.Material(
                    diffuse_reflectance = pyredner.Texture(texels=color, 
                                                           uv_scale=torch.tensor((1.0, 1.0)))
                )
                
        self.scene_backup[str(obj_idx)] = (self.scene.shapes[obj_idx],self.scene.materials[obj_idx])

        if return_img:
            if use_backup:
                for i in range(len(self.object_list)):
                    if i == obj_idx: continue
                    else:
                        self.scene.shapes[i],self.scene.materials[i] = self.scene_backup[str(i)]
                        
            img = pyredner.render_albedo(scene = self.scene, alpha = True)
            return img
    
    def get_v(self, obj_idx = 0):
        return self.object_list[obj_idx].vertices.clone()
    
    def get_v_from_scene(self, obj_idx = 0):
        return self.scene.shapes[obj_idx].vertices.clone()

    def set_aligned_obj(self,
                    color = None, color_mode = 0, update_uv_coordinate = True, 
                    slice_mode = False, easy_slice = False, scale_slice_num = 0, only_scale = False,
                    scale = torch.eye(3).to(pyredner.get_device()), standard_scale_mode = True,
                    translation = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device()),
                    cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    cam_up = None,
                    translation_for_all = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    euler_angles_for_all = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    opt_all = False,
                   ):   
          
        # set camera
        self.camera = pyredner.Camera(position = cam_pos,
                                   look_at = cam_look_at,
                                   up = self.up if cam_up is None else cam_up,
                                   fov = self.fov,
                                   clip_near = self.clip_near,
                                   resolution = (self.resx, self.resy))
                   
        # set scene
        self.scene = pyredner.Scene(camera = self.camera, objects = copy.deepcopy([self.object_list[-1]])) # only aligned object

        # 6-D and scale
        if slice_mode is False:
            rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
            if standard_scale_mode is True:
                scale = torch.diag(scale.diag())
            self.scene.shapes[0].vertices = (self.scene.shapes[0].vertices  - self.center[-1]) @ scale @ torch.t(rotation_matrix) \
                                             + self.center[-1] + translation
        else:
            if easy_slice:
                end = 0
                for obj_idx in range(len(self.object_v_num)): # the last one is aligned object
                    v_num = self.object_v_num[obj_idx]
                    v_num_slice = int(v_num/scale_slice_num)
                    start = end
                    end += v_num
                    curr_center = torch.mean(self.scene.shapes[0].vertices[start:end].clone().cuda(), 0)
                    for mask_idx in range(scale_slice_num):
                        start_idx = mask_idx * v_num_slice
                        if mask_idx == scale_slice_num - 1:
                            self.scene.shapes[0].vertices[start:end][start_idx:] = (self.scene.shapes[0].vertices[start:end][start_idx:]  - curr_center) \
                                                                                    @ torch.diag(scale[obj_idx][mask_idx].diag()) 
                        else:
                            end_idx = (mask_idx+1) * v_num_slice
                            self.scene.shapes[0].vertices[start:end][start_idx:end_idx] = (self.scene.shapes[0].vertices[start:end][start_idx:end_idx]  - curr_center) \
                                                                                           @ torch.diag(scale[obj_idx][mask_idx].diag()) 
                    self.scene.shapes[0].vertices[start:end] = self.scene.shapes[0].vertices[start:end].clone() @ torch.t(pyredner.gen_rotate_matrix(euler_angles[obj_idx])) \
                                                        + curr_center + translation[obj_idx]
            else:
                end = 0
                for obj_idx in range(len(self.object_v_num)): # the last one is aligned object
                    start = end
                    end += self.object_v_num[obj_idx]
                    curr_center = torch.mean(self.scene.shapes[0].vertices[start:end].clone().cuda(), 0)
                    if standard_scale_mode is True:
                        for mask_idx in range(self.sphere_region_num): # scale.size() = n_obj*sphere_region_num,3,3
                            mask = self.sphere_region_mask[mask_idx]
                            self.scene.shapes[0].vertices[start:end][mask] = (self.scene.shapes[0].vertices[start:end][mask].clone()  - curr_center) \
                                                                                        @ torch.diag(scale[obj_idx][mask_idx].diag()) 
                        if not only_scale:
                            self.scene.shapes[0].vertices[start:end] = self.scene.shapes[0].vertices[start:end].clone() @ torch.t(pyredner.gen_rotate_matrix(euler_angles[obj_idx])) \
                                                            + curr_center + translation[obj_idx]
                    else:
                        for mask_idx in range(self.sphere_region_num):
                            mask = self.sphere_region_mask[mask_idx]
                            self.scene.shapes[0].vertices[start:end][mask] = (self.scene.shapes[0].vertices[start:end][mask]  - curr_center) \
                                                                    @ scale[obj_idx][mask_idx] \
                                                                    @ torch.t(pyredner.gen_rotate_matrix(euler_angles[mask_idx])) \
                                                                    + curr_center + translation[mask_idx]
                if only_scale:
                    self.scene.shapes[0].vertices = self.scene.shapes[0].vertices.clone() @ torch.t(rotation_matrix) \
                                                         + self.center[-1] + translation
        if opt_all: # 
            rotation_matrix = pyredner.gen_rotate_matrix(euler_angles_for_all)
            self.scene.shapes[0].vertices = (self.scene.shapes[0].vertices.clone()  - self.center[-1]) @ torch.t(rotation_matrix) \
                                             + self.center[-1] + translation_for_all
            
        # normals & color & uvs
        obj_idx = -1
        if color_mode is self.DEFUALT_COLOR:
            self.scene.materials[obj_idx].use_vertex_color = False
        else:
            self.scene.materials[obj_idx].use_vertex_color = True
            if color_mode is self.V_COLOR:
                self.scene.shapes[obj_idx].colors = color
            elif color_mode is self.USE_UV_MAP:
                if update_uv_coordinate:
                    self.scene.shapes[obj_idx].normals = pyredner.compute_vertex_normal(self.scene.shapes[obj_idx].vertices,
                                                                  self.scene.shapes[obj_idx].indices)
                    self.scene.shapes[obj_idx].uvs, self.scene.shapes[obj_idx].uv_indices = pyredner.compute_uvs(self.scene.shapes[obj_idx].vertices,
                                                                                                     self.scene.shapes[obj_idx].indices,
                                                                                                     print_progress=False)
                    self.normals_backup[str(obj_idx)] = self.scene.shapes[obj_idx].normals
                    self.uvs_backup[str(obj_idx)] = self.scene.shapes[obj_idx].uvs
                    self.uv_indices_backup[str(obj_idx)] = self.scene.shapes[obj_idx].uv_indices
                else:
                    self.scene.shapes[obj_idx].normals = self.normals_backup[str(obj_idx)]
                    self.scene.shapes[obj_idx].uvs = self.uvs_backup[str(obj_idx)]
                    self.scene.shapes[obj_idx].uv_indices = self.uv_indices_backup[str(obj_idx)]
                self.scene.materials[obj_idx] = pyredner.Material(
                    diffuse_reflectance = pyredner.Texture(texels=color, 
                                                           uv_scale=torch.tensor((1.0, 1.0)))
                )
                       
        img = pyredner.render_albedo(scene = self.scene, alpha = True)
        return img
            
    def get_object_list(self):
        return self.object_list
    """---------------------------------------------------------------------------------
    find_load_position
        1. init_sphere
    """  
    def init_sphere(self,
                    pos_center : list,
                    cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device()),
                    cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                    sphere_num = 0, 
                    theta_steps = 0, 
                    phi_steps = 0,
                    radius = 0,
                    projection_matrix = None, use_projection_matrix_mode = False,
                    ):
        # set camera
        self.camera = pyredner.Camera(position = cam_pos,
                                      look_at = cam_look_at,
                                      up = self.up,
                                      fov = self.fov,
                                      clip_near = self.clip_near,
                                      resolution = (self.resx, self.resy))
                     
        # generate the sphere that meet the set requirements
        vertices, indices, normals = gen_sphere(theta_steps = theta_steps, phi_steps = phi_steps , radius = radius, need_uv =False)
        vertices = vertices.cuda(); indices = indices.cuda()
        m = self.init_material
        tiny_sphere_object = pyredner.Object(vertices = vertices, indices = indices, uvs = None, normals = normals, material = m)
        
        # load sphere objects
        self.sphere_num = sphere_num
        obj = []
        for i in range(self.sphere_num):
            obj.append(tiny_sphere_object)
        self.object_list.extend(obj)
        self.init_object_list = copy.deepcopy(self.object_list)
                        
        # set scene
        self.scene = pyredner.Scene(camera = self.camera, objects = self.object_list)

        """
        there is two method to set sphere color to black 
        1. self.scene.materials[0].use_vertex_color = True (and No need to set color)
        2. diffuse_reflectance = torch.tensor((0., 0., 0.))
        """                
        # set pos
        new_XYZ = pos_center.clone()
        if use_projection_matrix_mode is False:
            vec_cam = cam_pos - cam_look_at
            pos_tmp = pos_center.clone()
            for i in range(self.sphere_num):
                vec_sphere = pos_tmp[i] - cam_look_at
                dot = torch.dot(vec_sphere, vec_cam)
                projection_vector = (dot / torch.norm(vec_cam, 2) ** 2) * vec_cam
                pos_tmp_ = pos_tmp[i] - projection_vector
                new_XYZ[i] = pos_tmp_
                self.scene.shapes[i].vertices =  self.init_object_list[i].vertices.cuda() + pos_tmp_
        else:
            for i in range(self.sphere_num):
                displacement = torch.matmul(projection_matrix,pos_center[i].clone())
                new_XYZ[i] = displacement
                self.scene.shapes[i].vertices =  self.init_object_list[i].vertices.cuda() + displacement
                 
        img = pyredner.render_albedo(scene = self.scene, alpha = True)
        return img, new_XYZ
    """---------------------------------------------------------------------------------
    add sphere
        1. gen_samples
        2. transform_with_sphere
        3. transform_backward
        4. update_obj_list_with_scene_set
        5. align_all_obj_in_list
    """  
    def gen_samples(self, obj_idx = 0, number_of_points = 500):
        mesh = o3d.io.read_triangle_mesh(self.obj_load_path[obj_idx])
        if mesh.has_vertex_normals() == False:
                print("this mesh doesn't have vertex normals")
                mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points = number_of_points)
        pcd_points_np = np.array(pcd.points)
        samples = torch.tensor(pcd_points_np).tolist()
        return samples
    def transform_with_sphere(self,
                              pos: list,
                              theta_steps: int, 
                              phi_steps: int, 
                              radius: float):
        #-# --------------------gen one sphere
        vertices, indices, normals = gen_sphere(theta_steps = theta_steps, phi_steps = phi_steps , radius = radius, need_uv =False)
        vertices = vertices.cuda(); indices = indices.cuda()
        vertices += torch.tensor(pos).cuda()
        m = self.init_material
        tiny_sphere_object = pyredner.Object(vertices = vertices, indices = indices, uvs = None, normals = normals, material = m)
        self.object_list.append(tiny_sphere_object)
        self.center.append(torch.mean(vertices.clone().cuda(), 0))
        
    def transform_backward(self,obj_idx = None):
        if obj_idx is None:
            obj_idx = len(self.object_list) - 1
      
        del self.object_list[obj_idx]
        del self.center[obj_idx]
        if len(self.scene_backup) > obj_idx:
            del self.scene_backup[obj_idx]
            

    def update_obj_list_with_scene_set(self, obj_idx = -1): # for deepcopy, use detach()
        obj = pyredner.Object(vertices = self.scene.shapes[obj_idx].vertices.detach(),
                              indices = self.scene.shapes[obj_idx].indices.detach(), 
                              
                              uvs = self.scene.shapes[obj_idx].uvs.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].uvs, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uvs, 
                              
                              uv_indices = self.scene.shapes[obj_idx].uv_indices.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].uv_indices, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uv_indices, 
                              
                              normals = self.scene.shapes[obj_idx].normals.detach() \
                              
                                      if isinstance(self.scene.shapes[obj_idx].normals, torch.Tensor) \
                              
                                      else self.scene.shapes[obj_idx].uv_indices, 
                              
                              material = self.scene.materials[obj_idx].detach() \
                              
                                      if isinstance(self.scene.materials[obj_idx], torch.Tensor) \
                              
                                      else self.scene.materials[obj_idx], 
                             )
        # self.center[obj_idx] = torch.mean(obj.vertices.clone().cuda(), 0) 为了eu变化对上，不能改
        # for key in obj.__dict__.keys():
        #     info = obj.__dict__[key]
        #     print(key,info)
        # for key in obj.material.__dict__.keys():
        #     info = obj.material.__dict__[key]
        #     print(key,info)
        self.object_list[obj_idx] = copy.deepcopy(obj)
        
    def align_all_obj_in_list(self):
        save_vertices = self.object_list[0].vertices
        save_indices =  self.object_list[0].indices
        self.object_v_num = []
        self.object_v_num.append(self.object_list[0].vertices.shape[0])
        obj_idx = 1
        v_num = self.object_v_num[0]
        while(obj_idx<len(self.object_list)):
            save_vertices = torch.cat((save_vertices, self.object_list[obj_idx].vertices),dim=0)
            save_indices = torch.cat((save_indices, (self.object_list[obj_idx].indices + v_num)),dim=0)
            curr_v_num = self.object_list[obj_idx].vertices.shape[0]
            self.object_v_num.append(curr_v_num)
            v_num += curr_v_num
            obj_idx += 1
        obj = pyredner.Object(vertices = save_vertices,
                              indices = save_indices, 
                              uvs = None, 
                              normals = None,
                              material = self.init_material)
        self.object_list.append(obj)
        self.center.append(torch.mean(obj.vertices.clone().cuda(), 0))
            
    def cal_uv_coordinate(self, obj_inx = -1):
        self.normals_backup[str(obj_inx)] = pyredner.compute_vertex_normal(self.object_list[obj_inx].vertices,
                                                          self.object_list[obj_inx].indices)
        self.uvs_backup[str(obj_inx)], self.uv_indices_backup[str(obj_inx)] = pyredner.compute_uvs(self.object_list[obj_inx].vertices,
                                                                                         self.object_list[obj_inx].indices,
                                                                                         print_progress=False)
        return self.normals_backup[str(obj_inx)], self.uvs_backup[str(obj_inx)], self.uv_indices_backup[str(obj_inx)]
    def test_update_v_num(self, num):
        self.object_v_num = []
        vertices, indices, normals = gen_sphere(theta_steps = 16, phi_steps = 16 , radius = 1., need_uv =False)
        for i in range(num):
            self.object_v_num.append(vertices.shape[0])

from decimal import Decimal, localcontext
"""---------------------------------------------------------------------------------
find load position
    """
def find_load_position(shape_target, cam_pos, cam_look_at, CHECK = False):
    if CHECK:
        import matplotlib.pyplot as plt
        check_imgs = []
        check_imgs.append(shape_target.clone())
    resx = shape_target.shape[0]
    resy = shape_target.shape[1]
    white_background = torch.ones((resx,resy, 3)).to(pyredner.get_device())
    theta_steps = 3
    phi_steps = 3
    C = torch.zeros(1,3).cuda() #important: C.shape = (1,3)
    
    #------<1> big plane
    GET_CENTER_ = False
    BEST_DIFF = Decimal(f'{0:.6f}')
    num_find = 10
    while(GET_CENTER_ is False):
        sphere_num = 15
        radius = 0.05*torch.norm(cam_pos - cam_look_at).item()
        init_t = (torch.randn(sphere_num,3)*(radius*4.)).to(pyredner.get_device())
        xyz_cam = CameraModel(resx,resy)
        img, new_XYZ_ = xyz_cam.init_sphere(pos_center = init_t,
                                            cam_pos = cam_pos,
                                            cam_look_at = cam_look_at,
                                            sphere_num = sphere_num, 
                                            theta_steps = theta_steps, 
                                            phi_steps = phi_steps,
                                            radius = radius,
                                            )
        if CHECK: check_imgs.append(img.clone())
        for i in range(new_XYZ_.shape[0]):
            center_ = new_XYZ_[i]
            C[0] = center_
            center_cam_ = CameraModel(resx,resy)
            img, _ = center_cam_.init_sphere(pos_center = C,
                                             cam_pos = cam_pos,
                                             cam_look_at = cam_look_at,
                                             sphere_num = 1, 
                                             theta_steps = theta_steps, 
                                             phi_steps = phi_steps,
                                             radius = radius,
                                             )
            blend_img_white = add_background(img, white_background)
            loss_0 = (blend_img_white - white_background).pow(2).mean()
            if loss_0 < 1e-6: #空白 跳过
                continue
            else:
                blend_shape_target = add_background(img, shape_target)
                loss_1 = (blend_shape_target - shape_target).pow(2).mean()
                a = Decimal(f'{loss_0:.6f}')
                b = Decimal(f'{loss_1:.6f}')
                diff = a - b 
                with localcontext() as ctx:
                    ctx.prec = 6
                    if diff > 0: 
                        num_find -= 1
                        if CHECK: 
                            print(diff)
                            plt.imshow(torch.pow(img, 1.0/2.2).cpu());plt.show()
                        if diff > BEST_DIFF:
                            BEST_DIFF = diff
                            BEST_CENTER = center_
                            BEST_DIFF_IMG = img.clone()
                        if num_find == 0:
                            GET_CENTER_ = True
                            if CHECK: 
                                print("BEST_DIFF:",BEST_DIFF)
                                plt.imshow(torch.pow(BEST_DIFF_IMG, 1.0/2.2).cpu());plt.show()
                                check_imgs.append(BEST_DIFF_IMG.clone())
                            break
    #------<2> sprinkle salt
    while(True):
        radius = radius*0.15
        sphere_num = 300
        init_t = (torch.randn(sphere_num,3)*(radius*5.)).to(pyredner.get_device())+ BEST_CENTER
        xyz_cam = CameraModel(resx,resy)
        img, new_XYZ = xyz_cam.init_sphere(pos_center = init_t,
                                           cam_pos = cam_pos,
                                           cam_look_at = cam_look_at,
                                           sphere_num = sphere_num, 
                                           theta_steps = theta_steps, 
                                           phi_steps = phi_steps,
                                           radius = radius,
                                           )
        if CHECK: check_imgs.append(img.clone())
        for i in range(new_XYZ.shape[0]):
            center = new_XYZ[i]
            C[0] = center
            center_cam = CameraModel(resx,resy)
            img, _ = center_cam.init_sphere(pos_center = C,
                                             cam_pos = cam_pos,
                                             cam_look_at = cam_look_at,
                                             sphere_num = 1, 
                                             theta_steps = theta_steps, 
                                             phi_steps = phi_steps,
                                             radius = radius,
                                             )
            blend_img_white = add_background(img, white_background)
            loss_0 = (blend_img_white - white_background).pow(2).mean()
            if loss_0 < 1e-6:
                continue
            else:
                blend_shape_target = add_background(img, shape_target)
                loss_1 = (blend_shape_target - shape_target).pow(2).mean()
                a = Decimal(f'{loss_0:.6f}')
                b = Decimal(f'{loss_1:.6f}')
                with localcontext() as ctx:
                    ctx.prec = 6
                    if b < a:
                        print(">find center")
                        if CHECK: 
                            check_imgs.append(img.clone())
                            plt.figure(figsize=(12, 6))
                            for i in range(len(check_imgs)):
                                plt.subplot(1, len(check_imgs), i+1) # row, column, id
                                plt.imshow(torch.pow(check_imgs[i], 1.0/2.2).cpu())
                                # plt.axis('off')
                                plt.xticks([])
                                plt.yticks([])
                            # plt.savefig(op.join('Groceries',ids_ShapeNet[shapenet_id],'find_load_position.png'))
                            plt.show()
                        return center

# 用直方图的方式来寻找出现次数最多的颜色
def most_common_color_histogram(texture_color, bins=256):
    # Concatenate all color tensors and transfer to GPU
    all_colors = torch.cat(texture_color, dim=0).cuda()

    # Normalize colors to [0, 1] and quantize them to the nearest bin
    all_colors_quantized = (all_colors * bins).long()

    # Create a histogram for each color channel
    hist_r = torch.histc(all_colors_quantized[:, 0].float(), bins=bins, min=0, max=bins)
    hist_g = torch.histc(all_colors_quantized[:, 1].float(), bins=bins, min=0, max=bins)
    hist_b = torch.histc(all_colors_quantized[:, 2].float(), bins=bins, min=0, max=bins)

    # Find the most common bin for each color channel
    most_common_bin_r = torch.argmax(hist_r)
    most_common_bin_g = torch.argmax(hist_g)
    most_common_bin_b = torch.argmax(hist_b)

    # Convert bins back to color values
    most_common_color = torch.tensor([most_common_bin_r, most_common_bin_g, most_common_bin_b]).float() / bins

    return most_common_color
def add_background(_img,background_img):
    alpha = _img[:, :, 3:4]
    with_bg_img = _img[:, :, :3] * alpha + background_img * (1 - alpha)
    return with_bg_img
from decimal import Decimal, localcontext
def update_best_state(_loss, _best_loss, _best_params, force_update = False,
                      precision = 6, save_params_list = None):
    def deal_params_list(params_list):
        rst = []
        for data in params_list:
            if isinstance(data, torch.Tensor):
                rst.append(data.clone().detach())
            else:
                rst.append(data)
        return rst
    curr_loss = Decimal("{:.{}f}".format(_loss, precision))
    if force_update:
        _best_loss = curr_loss
        _best_params = deal_params_list(save_params_list)
        _best_params.append(_best_loss)
    else:
       with localcontext() as ctx:
            ctx.prec = precision
            if curr_loss < _best_loss:
                _best_loss = curr_loss
                _best_params = deal_params_list(save_params_list)
                _best_params.append(_best_loss)
    return _best_loss, _best_params
def plot_loss_and_img(losses, img, GT):
    # Plot the loss
    f, (ax_loss, ax_img) = plt.subplots(1, 2)
    ax_loss.plot(range(len(losses)), losses, label='loss')
    ax_loss.legend()
    ax_img.imshow((img - GT).pow(2).sum(axis=2).data.detach().cpu())
    plt.show()
lr_lambda = lambda x: max(0.0, 10**(-x*0.0002))
DEFUALT_COLOR, V_COLOR, USE_UV_MAP = 0, 1, 2
BLACK_COLOR = torch.zeros(3)
def diff_mesh_opt(cam = None, GT = None, penalize_overlap = False,
                  color = None, color_mode = 0, update_uv_coordinate = True, 
                  slice_mode = False, easy_slice = False, scale_slice_num = 0,
                  scale = torch.eye(3).to(pyredner.get_device()), standard_scale_mode = True,
                  translation = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                  euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                  cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device()),
                  cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                  cam_up  = None,
                  obj_idx = -1, use_backup = False, return_img = True,
                  background = None,
                  verbose = [],
                  opt_params_and_lr = [],
                  num_iters = 200, out_iters = 20,
                  opt_aligned_obj_mode = False,
                  pose_estimation_mode = False, mask_weight = 1., color_weight = 1.0,
                 ):
    if cam is None or GT is None:
        print("need camera model or GT")
        return
    if cam_up is None:
        cam_up = cam.up
    BLACK_COLOR = torch.zeros(3)
    obj_mask_color = BLACK_COLOR.repeat(cam.get_v(obj_idx=obj_idx).shape[0],1).cuda()
    from IPython.display import clear_output
    import tqdm
    pbar = tqdm.tqdm(range(num_iters))
    Decimal_losses = []
    if 'plot_the_loss' in verbose: plt.figure(); losses = []
    if 'save_opt_imgs_togo_gif_display' in verbose: opt_imgs = []
    best_loss = 0; best_params = []
    optimizers = [torch.optim.Adam([opt_params_and_lr[i]], lr=opt_params_and_lr[i+1]) for i in range(0, len(opt_params_and_lr), 2)]
    opt_params = [opt_params_and_lr[i] for i in range(0, len(opt_params_and_lr), 2)]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda) for opt in optimizers]
    for t in pbar:
        __opt = [opt.zero_grad() for opt in optimizers]
        if opt_aligned_obj_mode:
            img = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                      slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                      scale = scale, standard_scale_mode = standard_scale_mode,
                                      translation = translation,
                                      euler_angles = euler_angles,
                                      cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                     )
        else:
            img = cam.set_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                               slice_mode = slice_mode, 
                               scale = scale, standard_scale_mode = standard_scale_mode,
                               translation = translation,
                               euler_angles = euler_angles,
                               cam_pos = cam_pos, cam_look_at = cam_look_at,
                               obj_idx = obj_idx, use_backup = use_backup, return_img = return_img,
                               )
        with_bg_img = add_background(img, background)
        if pose_estimation_mode:
            # loss = 0
            # if t%2 ==0:
            loss = (with_bg_img - GT[0]).pow(2).mean()*color_weight
            if opt_aligned_obj_mode:
                img_with_no_texture = cam.set_aligned_obj(color = obj_mask_color, color_mode = V_COLOR, update_uv_coordinate = update_uv_coordinate, 
                                                          slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                                          scale = scale, standard_scale_mode = standard_scale_mode,
                                                          translation = translation,
                                                          euler_angles = euler_angles,
                                                          cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                                         )
            else:
                img_with_no_texture = cam.set_obj(color = obj_mask_color, color_mode = V_COLOR, update_uv_coordinate = update_uv_coordinate, 
                                                   slice_mode = slice_mode, 
                                                   scale = scale, standard_scale_mode = standard_scale_mode,
                                                   translation = translation,
                                                   euler_angles = euler_angles,
                                                   cam_pos = cam_pos, cam_look_at = cam_look_at,
                                                   obj_idx = obj_idx, use_backup = use_backup, return_img = return_img,
                                                   )
            with_bg_img_with_no_texture = add_background(img_with_no_texture, background)
            loss += (with_bg_img_with_no_texture - GT[1]).pow(2).mean()*mask_weight
            if penalize_overlap is True:
                with_gt_bg_img_with_no_texture = add_background(img_with_no_texture, GT[1])
                loss += (with_gt_bg_img_with_no_texture - GT[1]).pow(2).mean()
        else:
            loss = (with_bg_img - GT).pow(2).mean()
            if penalize_overlap is True and color_mode is DEFUALT_COLOR: # 有颜色的不设重叠惩罚，杂乱
                with_gt_bg_img = add_background(img, GT)
                loss += (with_gt_bg_img - GT).pow(2).mean()*mask_weight
        loss.backward()
        
        save_params_list = [t]
        save_params_list.extend(opt_params)
        best_loss, best_params = update_best_state(loss, best_loss, best_params, force_update=(t==0),
                                                save_params_list = save_params_list)
        __opt = [opt.step() for opt in optimizers]
        __opt = [sched.step() for sched in schedulers]
    
        
        if 'plot_the_loss' in verbose: 
            losses.append(loss.data.item())
            clear_output(wait=True)
            if pose_estimation_mode:
                plot_loss_and_img(losses, with_bg_img_with_no_texture, GT[1])
            else:
                plot_loss_and_img(losses, with_bg_img, GT)
        if 'save_opt_imgs_togo_gif_display' in verbose:
            opt_imgs.append(torch.pow(with_bg_img.clone().data, 1.0/2.2).detach().cpu())
        
        Decimal_losses.append(best_loss)
        if t > out_iters and best_loss == Decimal_losses[int(-1*out_iters)]:
            pbar.set_description(f'Loss: {best_loss:.6f}, '+'Converged, terminating early.')
            pbar.close()
            break
        pbar.set_description(f'Loss: {best_loss:.6f}')
    if 'save_opt_imgs_togo_gif_display' in verbose:
        return best_params, opt_imgs
    else:
        return best_params
#--------------------------------------------------------------------------------------------------------------------------------------
from ref_img_utils import img2png
def diff_mesh_opt_all(cam = None, GT_with_texture = None, GT_with_mask = None, penalize_overlap = False,
                      color = None, color_mode = 0, update_uv_coordinate = True, 
                      slice_mode = False, easy_slice = False, scale_slice_num = 0,
                      scale = torch.eye(3).to(pyredner.get_device()), standard_scale_mode = True,
                      translation = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                      euler_angles = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                      cam_pos = torch.tensor([0., 0., -1.], device = pyredner.get_device()),
                      cam_look_at = torch.tensor([0., 0., 0.], device = pyredner.get_device()),
                      cam_up  = None,
                      translation_for_all_list = [],
                      euler_angles_for_all_list = [],
                      obj_idx = -1, use_backup = False, return_img = True,
                      background = None,
                      verbose = [],
                      opt_params_and_lr = [],
                      num_iters = 200, out_iters = 20,
                      pose_estimation_mode = False, mask_weight = 1., 
                      write_opt_images = False, fp_write_opt_images = None,
                     ):
    if write_opt_images:
        os.makedirs(fp_write_opt_images, exist_ok=True)
    if cam is None:
        print("need camera model")
        return
    if cam_up is None:
        cam_up = cam.up
    BLACK_COLOR = torch.zeros(3)
    obj_mask_color = BLACK_COLOR.repeat(cam.get_v(obj_idx=obj_idx).shape[0],1).cuda()
    from IPython.display import clear_output
    import tqdm
    pbar = tqdm.tqdm(range(num_iters))
    Decimal_losses = []
    if 'plot_the_loss' in verbose: plt.figure(); losses = []
    if 'save_opt_imgs_togo_gif_display' in verbose: opt_imgs = []
    best_loss = 0; best_params = []
    optimizers = [torch.optim.Adam([opt_params_and_lr[i]], lr=opt_params_and_lr[i+1]) for i in range(0, len(opt_params_and_lr), 2)]
    opt_params = [opt_params_and_lr[i] for i in range(0, len(opt_params_and_lr), 2)]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda) for opt in optimizers]
    for t in pbar:
        __opt = [opt.zero_grad() for opt in optimizers]
        loss = 0
        for img_idx in range(len(GT_with_texture)):
            if img_idx == 0:
                img_weight = 8.0
            else:
                img_weight = 1.0
            translation_for_all = translation_for_all_list[img_idx]
            euler_angles_for_all = euler_angles_for_all_list[img_idx]
            img = cam.set_aligned_obj(color = color, color_mode = color_mode, update_uv_coordinate = update_uv_coordinate, 
                                      slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                      scale = scale, standard_scale_mode = standard_scale_mode,
                                      translation = translation,
                                      euler_angles = euler_angles,
                                      cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                      translation_for_all = translation_for_all,
                                      euler_angles_for_all = euler_angles_for_all,
                                      opt_all = True,
                                     )
            with_bg_img = add_background(img, background)
            loss += (with_bg_img - GT_with_texture[img_idx]).pow(2).mean()*img_weight
            if pose_estimation_mode:
                img_with_no_texture = cam.set_aligned_obj(color = obj_mask_color, color_mode = V_COLOR, update_uv_coordinate = update_uv_coordinate, 
                                                          slice_mode = slice_mode, easy_slice = easy_slice, scale_slice_num = scale_slice_num,
                                                          scale = scale, standard_scale_mode = standard_scale_mode,
                                                          translation = translation,
                                                          euler_angles = euler_angles,
                                                          cam_pos = cam_pos, cam_look_at = cam_look_at, cam_up = cam_up,
                                                          translation_for_all = translation_for_all,
                                                          euler_angles_for_all = euler_angles_for_all,
                                                          opt_all = True,
                                                         )
                with_bg_img_with_no_texture = add_background(img_with_no_texture, background)
                loss += (with_bg_img_with_no_texture - GT_with_mask[img_idx]).pow(2).mean()*mask_weight*img_weight
                if penalize_overlap is True:
                    with_gt_bg_img_with_no_texture = add_background(img_with_no_texture, GT_with_mask[img_idx])
                    loss += (with_gt_bg_img_with_no_texture - GT_with_mask[img_idx]).pow(2).mean()*img_weight
            if write_opt_images and img_idx == 0:
                img2png(with_bg_img.detach(),op.join(fp_write_opt_images,f'opt-color-{t:03d}.png'))
                if pose_estimation_mode:
                    img2png(with_bg_img_with_no_texture.detach(),op.join(fp_write_opt_images,f'opt-mask-{t:03d}.png'))
                
        loss.backward()
        
        save_params_list = [t]
        save_params_list.extend(opt_params)
        best_loss, best_params = update_best_state(loss, best_loss, best_params, force_update=(t==0),
                                                save_params_list = save_params_list)
        __opt = [opt.step() for opt in optimizers]
        __opt = [sched.step() for sched in schedulers]
    
        
        if 'plot_the_loss' in verbose: 
            losses.append(loss.data.item())
            clear_output(wait=True)
            if pose_estimation_mode:
                plot_loss_and_img(losses, with_bg_img_with_no_texture, GT[1])
            else:
                plot_loss_and_img(losses, with_bg_img, GT)
        if 'save_opt_imgs_togo_gif_display' in verbose:
            opt_imgs.append(torch.pow(with_bg_img.clone().data, 1.0/2.2).detach().cpu())
        
        Decimal_losses.append(best_loss)
        if t > out_iters and best_loss == Decimal_losses[int(-1*out_iters)]:
            pbar.set_description(f'Loss: {best_loss:.6f}, '+'Converged, terminating early.')
            pbar.close()
            break
        pbar.set_description(f'Loss: {best_loss:.6f}')
    if 'save_opt_imgs_togo_gif_display' in verbose:
        return best_params, opt_imgs
    else:
        return best_params

#----------------------------------------for-diff-sdf
def sample_spherical(N, center, radius, device="cuda"):
    polar_min = torch.deg2rad(torch.tensor(0.)).cuda()
    polar_max = torch.deg2rad(torch.tensor(180.)).cuda()
    polar = torch.linspace(polar_min, polar_max, N, device=device)

    azimuth_min = torch.deg2rad(torch.tensor(0.)).cuda()
    azimuth_max = torch.deg2rad(torch.tensor(360.)).cuda()
    azimuth = torch.linspace(azimuth_min, azimuth_max, N, device=device)

    x = radius * torch.sin(polar) * torch.cos(azimuth) + center[0]
    y = radius * torch.sin(polar) * torch.sin(azimuth) + center[1]
    z = radius * torch.cos(polar) + center[2]

    return torch.stack((x, y, z), dim=1)
import re
def get_images(directory, need, endwith='.exr'):
    formate = endwith
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(formate)]
    files.sort() 

    pattern = r"(-?\d+_\d+|-?\d+)\+(-?\d+_\d+|-?\d+)"+endwith+"$"
    regex = re.compile(pattern)

    data = []
    need_files = []
    for file in files:
        filename = os.path.basename(file)
        if filename not in need:
            continue
        need_files.append(file)
        match = regex.search(filename)
        if match is not None:
            a = float(match.group(1).replace('_', '.')) if '_' in match.group(1) else float(match.group(1))
            b = -1*float(match.group(2).replace('_', '.')) if '_' in match.group(2) else float(match.group(2))
            data.append((a, b))
        else:
            data.append(None)

    return need_files, data

def calculate_camera_positions(data, center, init_cam_pos, device="cuda"):
    center = torch.tensor(center).cuda()
    init_cam_pos = torch.tensor(init_cam_pos).cuda()
    radius = torch.norm(center - init_cam_pos).item()
    
    positions = []
    for polar, azimuth in data:
        # Convert degrees to radians
        polar = -1*np.deg2rad(polar)
        azimuth = np.deg2rad(azimuth)

        # Calculate the new position
        x = init_cam_pos[0] + radius*np.sin(azimuth)
        y = init_cam_pos[1] - radius*np.sin(polar)
        z = init_cam_pos[2] + (radius - radius*np.cos(azimuth)) + (radius - radius*np.cos(polar))
        positions.append([x.item(), y.item(), z.item()])
    
    return positions

def gen_gif(input_folder,fn_save_gif):
    import pyredner
    image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')])
    imgs= []
    for image_file in image_files:
        imgs.append(pyredner.imread(image_file))
    imgs = torch.stack(imgs)
    from ref_img_utils import rdImgs2Gif
    rdImgs2Gif(fn_save_gif, imgs, img_is_numpy = False)

def obj_to_cam(init_translation, init_euler_angles, goal_translation, goal_euler_angles, cam_pos, cam_look_at, cam_up):
    rotation_matrix = torch.inverse(pyredner.gen_rotate_matrix(goal_euler_angles))
    cam_v = torch.stack([
        cam_pos,
        cam_look_at,
        cam_up,
        goal_translation]).cuda()
    neo_cam_v = cam_v @ torch.t(rotation_matrix)
    neo_cam_pos, neo_cam_look_at, neo_cam_up, obj_pos_rotate = neo_cam_v[:]
    # 就xyz而言，cam_pos和cam_look_at要负向相对移动
    obj_xyz_vector = obj_pos_rotate - init_translation
    cam_xyz_vector = -1*obj_xyz_vector
    neo_cam_pos += cam_xyz_vector
    neo_cam_look_at += cam_xyz_vector
    return neo_cam_pos, neo_cam_look_at, neo_cam_up