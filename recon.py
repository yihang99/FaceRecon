# This program reconstruct the face from single/multi images

import torch
import pyredner
import h5py
import numpy as np
import os

os.system("rm -rf process")

# Load the Basel face model
shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
indices = triangle_list.permute(1, 0).contiguous()
print("finish loading")

def model(cam_pos, cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction):
    #vertices = (shape_mean + shape_basis @ torch.zeros(199, device=pyredner.get_device())).view(-1, 3)
    normals = pyredner.compute_vertex_normal(vertices, indices)
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)
    #m = pyredner.Material(use_vertex_color=True)
    m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
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
    return img, obj


#cam_pos = torch.tensor([-80.2697, -55.7891, 373.9277])
#cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918])
#img = model(cam_pos, cam_look_at, torch.zeros(199, device=pyredner.get_device()), torch.zeros(199, device=pyredner.get_device()), torch.ones(3), torch.zeros(3))
#pyredner.imwrite(img.cpu(), 'img.png')
data_path = "generated/dataset2/"
c_p, cam_look_at, dir_light_intensity, dir_light_direction = np.load(data_path+"env_data.npy", allow_pickle=True)
cam_poses = torch.tensor(c_p, requires_grad=True)

#target = pyredner.imread('generated/img03.png').to(pyredner.get_device())
target = []
for i in range(len(cam_poses)):
    target.append(pyredner.imread(data_path+'target_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))
pyredner.imwrite(target[i].cpu(), 'process/target_img{:0>2d}.png'.format(i))


#cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277], requires_grad=True)
#cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918], requires_grad=True)
#shape_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=True)
#dir_light_intensity = torch.ones(3, device=pyredner.get_device(), requires_grad=True)
#dir_light_direction = torch.tensor([0.0, 0.0, -1.0], device=pyredner.get_device(), requires_grad=True)
vertices = (shape_mean + shape_basis @ torch.zeros(199, device=pyredner.get_device())).view(-1, 3)
vertices.requires_grad = True

light_optimizer = torch.optim.Adam([ambient_color, dir_light_intensity, dir_light_direction], lr=0.1)
ver_optimizer = torch.optim.Adam([vertices], lr=0.02)
cam_optimizer = torch.optim.Adam([cam_poses, cam_look_at], lr=1.0)

import matplotlib.pyplot as plt
#% matplotlib inline
#from IPython.display import display, clear_output
#import time

plt.figure()
losses = []
for i in range(len(cam_poses)):
    losses.append([])
# Run 500 Adam iterations
num_iters_1 = 1
num_iters_2 = 1

'''
for t in range(num_iters_1):
    light_optimizer.zero_grad()
    cam_optimizer.zero_grad()
    ver_optimizer.zero_grad()
    img, obj = model(cam_pos, cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction)
    loss = (img - target).pow(2).mean()
    loss.backward()
    light_optimizer.step()
    cam_optimizer.step()
    ambient_color.data.clamp_(0.0)
    dir_light_intensity.data.clamp_(0.0)
    # Plot the loss
    #f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
    losses.append(loss.data.item())
    # Only store images every 10th iterations
    if (t+1) % 20 == 0:
        #imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        pyredner.imwrite(img.data.cpu(), 'process/process1_img{:0>2d}.png'.format(t // 20))
    print("{:.^20}".format(t))
'''

for t in range(num_iters_2):
    for i in range(len(cam_poses)):
        light_optimizer.zero_grad()
        cam_optimizer.zero_grad()
        ver_optimizer.zero_grad()
        img, obj = model(cam_poses[i], cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction)
        # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, so no gamma correction is needed.
        #imgs.append(img.numpy())
        loss = (img - target[i]).pow(2).mean()
        loss.backward()
        #optimizer.step()
        ver_optimizer.step()
        #cam_optimizer.step()
        ambient_color.data.clamp_(0.0)
        dir_light_intensity.data.clamp_(0.0)
        # Plot the loss
        #f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
        losses[i].append(loss.data.item())
        # Only store images every 10th iterations

    if t % 5 == 0:
  #      imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        pyredner.imwrite(img.data.cpu(), 'process/process2_img{:0>2d}.png'.format(t // 5))
    print("{:.^20}".format(t))

    if t == 30:
        ver_optimizer = torch.optim.Adam([vertices], lr=0.01)


#for x in losses:
   # print('0{:0>7.6f}'.format(x), end=' ')

pyredner.save_obj(obj, 'process/final.obj')

for i in range(len(cam_poses)):
    img, obj = model(cam_poses[i], cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction)
    pyredner.imwrite(img.data.cpu(), 'process/result/view0{}.png'.format(i))

    plt.plot(losses[i])

plt.ylabel("loss")
plt.xlabel("iterations")
plt.savefig("process/result/lossCurve.png", dpi=800)

print(losses)