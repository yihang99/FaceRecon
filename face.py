# This program generate multi view random face dataset under 3DMM model

import torch
import pyredner
import h5py
import os
import numpy as np

os.system("rm -rf generated")

# Load the Basel face model
shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
indices = triangle_list.permute(1, 0).contiguous()
print("finish loading")

def model(cam_pos, cam_look_at, shape_coeffs, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction):
    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)
    normals = pyredner.compute_vertex_normal(vertices, indices)
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)
    m = pyredner.Material(diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)

    cam = pyredner.Camera(position=cam_pos,
                          look_at=cam_look_at,  # Center of the vertices
                          up=torch.tensor([0.0, 1.0, 0.0]),
                          fov=torch.tensor([45.0]),
                          resolution=(512, 512))
    scene = pyredner.Scene(camera=cam, objects=[obj])
    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(dir_light_direction, dir_light_intensity)
    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light])
    return (img, obj)

dir_light_direction = torch.tensor([-1.0, -1.0, -1.0])
dir_light_intensity = torch.ones(3)
cam_poses = [[-0.2697, -5.7891, 373.9277],
             [-80.2697, -55.7891, 373.9277],
             [-80.2697, 45.7891, 373.9277],
             [80.2697, -55.7891, 373.9277],
             [80.2697, 45.7891, 373.9277]]
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918])
env_data = np.array((cam_poses, cam_look_at, dir_light_intensity, dir_light_direction))

#img = model(cam_pos, cam_look_at, torch.ones(199, device=pyredner.get_device()), torch.ones(199, device=pyredner.get_device()), torch.ones(3), torch.zeros(3))
#pyredner.imwrite(img.cpu(), 'img.png')
import numpy.random as nprd
for j in range(4):
    shape_coe = 25 * torch.randn(199, device=pyredner.get_device(), dtype = torch.float32)
    for i in range(len(cam_poses)):
        cam_pos = torch.tensor(cam_poses[i])
        (img, obj) = model(cam_pos, cam_look_at, shape_coe,
                    torch.tensor(1.4*nprd.randn(199), device=pyredner.get_device(), dtype = torch.float32), torch.zeros(3), dir_light_intensity, dir_light_direction)
        pyredner.imwrite(img.cpu(), 'generated/dataset{}/target_img{:0>2d}.png'.format(j, i))
    pyredner.save_obj(obj, 'generated/dataset{}/object{:0>2d}.obj'.format(j, i))
    np.save("generated/dataset{}/env_data.npy".format(j), env_data)


(img, obj) = model(cam_pos, cam_look_at, torch.tensor(0*nprd.randn(199), device=pyredner.get_device(), dtype = torch.float32),
                torch.tensor(0*nprd.randn(199), device=pyredner.get_device(), dtype = torch.float32), torch.zeros(3), dir_light_intensity, dir_light_direction)
pyredner.imwrite(img.cpu(), 'generated/average.png')
pyredner.save_obj(obj, 'generated/average.obj')






















'''
import urllib

#urllib.request.urlretrieve('https://raw.githubusercontent.com/BachiLi/redner/master/tutorials/mona-lisa-cropped-256.png', 'target.png')
target = pyredner.imread('target.png').to(pyredner.get_device())
cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277], requires_grad=True)
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918], requires_grad=True)
shape_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
ambient_color = torch.ones(3, device=pyredner.get_device(), requires_grad=True)
dir_light_intensity = torch.zeros(3, device=pyredner.get_device(), requires_grad=True)

optimizer = torch.optim.Adam([shape_coeffs, color_coeffs, ambient_color, dir_light_intensity], lr=0.1)
cam_optimizer = torch.optim.Adam([cam_pos, cam_look_at], lr=0.5)

import matplotlib.pyplot as plt
#% matplotlib inline
from IPython.display import display, clear_output
import time

plt.figure()
imgs, losses = [], []
# Run 500 Adam iterations
num_iters = 500
for t in range(num_iters):
    optimizer.zero_grad()
    cam_optimizer.zero_grad()
    img = model(cam_pos, cam_look_at, shape_coeffs, color_coeffs, ambient_color, dir_light_intensity)
    # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
    # Both img and target are in linear color space, so no gamma correction is needed.
    loss = (img - target).pow(2).mean()
    loss = loss + 0.0001 * shape_coeffs.pow(2).mean() + 0.001 * color_coeffs.pow(2).mean()
    loss.backward()
    optimizer.step()
    cam_optimizer.step()
    ambient_color.data.clamp_(0.0)
    dir_light_intensity.data.clamp_(0.0)
    # Plot the loss
    f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
    losses.append(loss.data.item())
    # Only store images every 10th iterations
    if t % 10 == 0:
        imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        pyredner.imwrite(imgs[t // 10].cpu(), 'process/img{}.png'.format(t // 10))
    clear_output(wait=True)
    ax_loss.plot(range(len(losses)), losses, label='loss')
    ax_loss.legend()
    ax_diff_img.imshow((img - target).pow(2).sum(dim=2).data.cpu())
    ax_img.imshow(torch.pow(img.data.cpu(), 1.0 / 2.2))
    plt.show()


'''










