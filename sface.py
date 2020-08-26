import torch
import pyredner
import os
import numpy as np
import h5py

os.chdir('..')

name = '5views4'

os.system('rm -rf generated/senv_dataset_' + name)

# Load the Basel face model
shape_mean, shape_basis, triangle_list, color_mean, color_basis, _ = np.load("3dmm.npy", allow_pickle=True)
# with h5py.File(r'model2017-1_bfm_nomouth.h5', 'r') as hf:
#     shape_mean = torch.tensor(hf['shape/model/mean'], device=pyredner.get_device())
#     shape_basis = torch.tensor(hf['shape/model/pcaBasis'], device=pyredner.get_device())
#     triangle_list = torch.tensor(hf['shape/representer/cells'], device=pyredner.get_device())
#     color_mean = torch.tensor(hf['color/model/mean'], device=pyredner.get_device())
#     color_basis = torch.tensor(hf['color/model/pcaBasis'], device=pyredner.get_device())

indices = triangle_list.permute(1, 0).contiguous()

envmap_img = pyredner.imread('env3.png')
pyredner.imwrite(envmap_img, 'generated/senv_dataset_' + name + '/env_map.png')
envmap = pyredner.EnvironmentMap(envmap_img)

print("finish loading")


def model(cam_poses, cam_look_ats, shape_coeffs, color_coeffs, resolution):
    # First rotate around center, then translation

    imgs = []

    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)
    normals = pyredner.compute_vertex_normal(vertices, indices)
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)
    m = pyredner.Material(use_vertex_color=False,
                          specular_reflectance=torch.tensor([1., 1., 1.], device=pyredner.get_device()),
                          roughness=torch.tensor([0.02]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)
    obj = pyredner.load_obj('generated/env_dataset_oness_n/tgt_obj.obj', return_objects=True)[0]
    obj.material.specular_reflectance=pyredner.Texture(torch.tensor([0.05, 0.05, 0.05], device=pyredner.get_device()))
    obj.material.roughness=pyredner.Texture(torch.tensor([0.02]))
    pyredner.save_obj(obj, "generated/senv_dataset_" + name + '/tgt_obj.obj')

    for i in range(len(cam_poses)):
        cam = pyredner.Camera(position=cam_poses[i],
                              look_at=cam_look_ats[i % len(cam_look_ats)],  # Center of the vertices
                              up=torch.tensor([0.0, 1.0, 0.0]),
                              fov=torch.tensor([45.0]),
                              resolution=resolution)
        scene = pyredner.Scene(camera=cam, objects=[obj], envmap=envmap)

        img = pyredner.render_pathtracing(scene=scene, num_samples=(128, 4))
        imgs.append(img)
    return imgs


cam_poses = torch.tensor([[-0.2697, -5.7891, 350.9277],
                          [-240.2697, -5.7891, 240.9277],
                          [240.2697, -5.7891, 240.9277],
                          [-100.2697, -95.7891, 320.9277],
                          [-100.2697, 85.7891, 320.9277],
                          [100.2697, -95.7891, 320.9277],
                          [100.2697, 85.7891, 320.9277]])

cam_poses = torch.tensor([[-0.2697, -5.7891, 350.9277],
                          [-150.2697, -115.7891, 300.9277],
                          [-150.2697, 105.7891, 300.9277],
                          [150.2697, -115.7891, 300.9277],
                          [150.2697, 105.7891, 300.9277]])
cam_poses = torch.tensor([[-0.2697, -5.7891, 350.9277],
                          [-300.2697, -5.7891, 70.9277],
                          [-180.2697, -5.7891, 280.9277],
                          [300.2697, -5.7891, 70.9277],
                          [180.2697, -5.7891, 280.9277]])
cam_look_ats = torch.tensor([-0.2697, -0.7891, 54.7918]).reshape(-1, 3)

center = torch.zeros(3, dtype=torch.float32)  # torch.tensor([-0.2697, -0.7891, 4.7918])



resolution = (400, 400)
resolution = (200, 200)
env_data = np.array((cam_poses, cam_look_ats, resolution))

# img = model(cam_pos, cam_look_at, torch.ones(199, device=pyredner.get_device()), torch.ones(199, device=pyredner.get_device()), torch.ones(3), torch.zeros(3))
# pyredner.imwrite(img.cpu(), 'img.png')

shape_coe = 1 * 30 * torch.ones(199, device=pyredner.get_device())
# torch.randn(199, device=pyredner.get_device(), dtype=torch.float32)
color_coe = 1 * 3 * torch.ones(199, device=pyredner.get_device())
# torch.tensor(3 * nprd.randn(199), device=pyredner.get_device(), dtype=torch.float32)

imgs = model(cam_poses, cam_look_ats, shape_coe, color_coe, resolution)
for i in range(len(imgs)):
    pyredner.imwrite(imgs[i].cpu(), 'generated/senv_dataset_' + name + '/tgt_img{:0>2d}.png'.format(i))

np.save('generated/senv_dataset_' + name + '/env_data.npy', env_data)

print('finish generating!')
