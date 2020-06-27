# This program generate multi view random face dataset under 3DMM model

import torch
import pyredner
import os
import numpy as np

os.chdir('..')

name = '30_5'

os.system('rm -rf generated/env_dataset_' + name)

# Load the Basel face model
#shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
#indices = triangle_list.permute(1, 0).contiguous()

envmap_img = pyredner.imread('env2.png') * 3
pyredner.imwrite(envmap_img, 'generated/env_dataset_' + name + '/env_map.png')
envmap = pyredner.EnvironmentMap(envmap_img)

print("finish loading")


def model(cam_pos, cam_look_at, shape_coeffs, color_coeffs, resolution,
          center, all_euler_angles, all_translations):
    # First rotate around center, then translation

    imgs = []

    #obj = pyredner.load_obj('p_ones30/final.obj', return_objects=True)[0]
    vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
    vertices *= 80
    m = pyredner.Material(diffuse_reflectance=torch.ones(2, 2, 3, dtype=torch.float32))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, uvs=uvs, material=m)
    v = obj.vertices.clone()

    for i in range(len(all_translations)):
        rotation_matrix = pyredner.gen_rotate_matrix(all_euler_angles[i]).to(pyredner.get_device())
        center = center.to(pyredner.get_device())
        # vertices = ((shape_mean + shape_basis @ shape_coeffs).view(-1, 3) - center) @ torch.t(rotation_matrix) + center + all_translations[i].to(pyredner.get_device())
        obj.vertices = (v - center) @ torch.t(rotation_matrix) + center
        obj.normals = pyredner.compute_vertex_normal(obj.vertices, indices)
        # colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)
        # m = pyredner.Material(diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5]))
        m = pyredner.Material(use_vertex_color=True)
        # obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)

        if i == 0:
            pyredner.save_obj(obj, "generated/env_dataset_" + name + '/tgt_obj.obj')

        cam = pyredner.Camera(position=cam_pos,
                              look_at=cam_look_at,  # Center of the vertices
                              up=torch.tensor([0.0, 1.0, 0.0]),
                              fov=torch.tensor([45.0]),
                              resolution=resolution)
        scene = pyredner.Scene(camera=cam, objects=[obj], envmap=envmap)

        img = pyredner.render_pathtracing(scene=scene, num_samples=(128, 4))
        imgs.append(img)
    return imgs


cam_pos = torch.tensor([-0.2697, -0.7891, 360.9277])

cam_look_at = torch.tensor([-0.2697, -0.7891, 54.7918])

center = torch.zeros(3, dtype=torch.float32)  # torch.tensor([-0.2697, -0.7891, 4.7918])

all_euler_angles = [torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([0.2, 0.0, 0.0]),
                    torch.tensor([0.4, 0.0, 0.0]),
                    torch.tensor([0.2, 0.2, 0.01]),
                    torch.tensor([0.0, 0.4, 0.0]),
                    torch.tensor([-0.2, 0.2, -0.01]),
                    torch.tensor([-0.4, 0.0, 0.0]),
                    torch.tensor([-0.2, -0.2, 0.01]),
                    torch.tensor([0.0, -0.4, 0.0]),
                    torch.tensor([0.2, -0.2, -0.01])]

all_euler_angles = [torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([-0.4, 0.6, 0.0]),
                    torch.tensor([0.4, 0.6, 0.0]),
                    torch.tensor([-0.4, -0.6, 0.0]),
                    torch.tensor([0.4, -0.6, 0.0])]
all_euler_angles = [torch.tensor([0.4, 0.0, 0.0]),
                    torch.tensor([0.34, 0.2, 0.0]),
                    torch.tensor([0.2, 0.34, 0.0]),
                    torch.tensor([0.0, 0.4, 0.0]),
                    torch.tensor([-0.2, 0.34, 0.0]),
                    torch.tensor([-0.34, 0.2, 0.0]),
                    torch.tensor([-0.4, 0.0, 0.0]),
                    torch.tensor([-0.34, -0.2, 0.0]),
                    torch.tensor([-0.2, -0.34, 0.0]),
                    torch.tensor([0.0, -0.4, -0.0]),
                    torch.tensor([0.2, -0.34, 0.0]),
                    torch.tensor([0.34, -0.2, 0.0])]
all_euler_angles = [torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([0.0, 0.1, 0.0]),
                    torch.tensor([0.0, 0.2, 0.0]),
                    torch.tensor([0.0, 0.3, 0.0]),
                    torch.tensor([0.0, 0.4, 0.0]),
                    torch.tensor([0.0, 0.5, 0.0]),
                    torch.tensor([0.0, 0.6, 0.0]),
                    torch.tensor([1.0, 0.0, 0.0]),
                    torch.tensor([0.0, 1.0, 0.0])]
# head up, head turn and head lean
all_translations = [torch.zeros(3)] * len(all_euler_angles)

resolution = (400, 400)
env_data = np.array((cam_pos, cam_look_at, len(all_euler_angles), all_euler_angles, center, resolution))

# img = model(cam_pos, cam_look_at, torch.ones(199, device=pyredner.get_device()), torch.ones(199, device=pyredner.get_device()), torch.ones(3), torch.zeros(3))
# pyredner.imwrite(img.cpu(), 'img.png')

shape_coe = 0 * 20 * torch.ones(199, device=pyredner.get_device())
# torch.randn(199, device=pyredner.get_device(), dtype=torch.float32)
color_coe = 0 * 2 * torch.ones(199, device=pyredner.get_device())
# torch.tensor(3 * nprd.randn(199), device=pyredner.get_device(), dtype=torch.float32)

imgs = model(cam_pos, cam_look_at, shape_coe, color_coe, resolution, center, all_euler_angles, all_translations)
for i in range(len(imgs)):
    pyredner.imwrite(imgs[i].cpu(), 'generated/env_dataset_' + name + '/tgt_img{:0>2d}.png'.format(i))

# obj.material = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
# pyredner.save_obj(obj, 'generated/dataset_' + name + '/' + name + '.obj')
np.save('generated/env_dataset_' + name + '/env_data.npy', env_data)


