# This program reconstruct the face from single/multi images

import torch
import pyredner
import h5py
import numpy as np
import os

os.chdir('..')
os.system("rm -rf process")

#pyredner.set_print_timing(False)

shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
indices = triangle_list.permute(1, 0).contiguous()
print("finish loading")

def stupid_smooth(vertices, indices, lmd):
    map = {}
    for triangle in indices:
        for i in range(3):
            map.setdefault(triangle[i].item(), set()).add(triangle[(i + 1) % 3].item())
            map.setdefault(triangle[i].item(), set()).add(triangle[(i + 2) % 3].item())
    new_v = torch.zeros(vertices.shape, device=vertices.device)
    for i in range(len(new_v)):
        weight, sum = 0, 0
        for neighbor in map[i]:
            weight += 1. / (vertices[i] - vertices[neighbor]).pow(2).sum()
            sum += weight * vertices[neighbor]
        shift = sum / weight
        new_v[i] = shift * lmd
        if i % 1000 == 0:
            print(new_v[i])
    return new_v + vertices * (1 - lmd)

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
                          resolution=(640, 640))
    scene = pyredner.Scene(camera=cam, objects=[obj])
    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(dir_light_direction, dir_light_intensity)
    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light])
    return img, obj

target_data_path = "generated/dataset1/"
c_p, cam_look_at, dir_light_intensity, dir_light_direction = np.load(target_data_path+"env_data.npy", allow_pickle=True)
cam_poses = torch.tensor(c_p[:], requires_grad=False)

#target = pyredner.imread('generated/img03.png').to(pyredner.get_device())
target = []
for i in range(len(cam_poses)):
    target.append(pyredner.imread(target_data_path+'target_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))
    pyredner.imwrite(target[i].cpu(), 'process/target_img{:0>2d}.png'.format(i))


color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=False)
ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=False)

vertices = (shape_mean).view(-1, 3)

#vertices, indices, uvs, normals = pyredner.generate_sphere(theta_steps = 128, phi_steps = 256)
#vertices = vertices * 110

vertices.requires_grad = True

#light_optimizer = torch.optim.Adam([ambient_color, dir_light_intensity, dir_light_direction], lr=0.1)
#cam_optimizer = torch.optim.Adam([cam_poses, cam_look_at], lr=1.0)
ver_optimizer = torch.optim.Adam([vertices], lr=0.15)

import matplotlib.pyplot as plt

plt.figure()
losses, imgs = [], []
for i in range(len(cam_poses)):
    losses.append([])
    imgs.append([])

num_iters_1 = 200
num_iters_2 = 10

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
    total_loss = 0
    # vertices = pyredner.laplacian_smooth(vertices, indices, 1.0).detach()
    # vertices.requires_grad = True
    ver_optimizer.zero_grad()
    # ver_optimizer = torch.optim.Adam([vertices], lr=0.15)

    for i in range(len(cam_poses)):
        img, obj = model(cam_poses[i], cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction)
        # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, so no gamma correction is needed.
        #imgs.append(img.numpy())
        loss = (img - target[i]).pow(2).mean()
        losses[i].append(loss.data.item())
        total_loss += loss

        imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        #pyredner.imwrite(abs(img - target[4]).data.cpu(), 'process/process2_img{:0>2d}.png'.format(t // 5))
    print("{:.^16}total_loss = {:.6f}".format(t, total_loss))

    total_loss.backward()
    ver_optimizer.step()

pyredner.save_obj(obj, 'process/final.obj')

for i in range(len(cam_poses)):
    img, obj = model(cam_poses[i], cam_look_at, vertices, color_coeffs, ambient_color, dir_light_intensity, dir_light_direction)
    pyredner.imwrite(img.data.cpu(), 'process/view0{}.png'.format(i))

    plt.plot(losses[i], label='view0{}'.format(i))

plt.legend()
plt.ylabel("loss")
plt.xlabel("iterations")
plt.savefig("process/lossCurve.png", dpi=800)

for i in range(len(cam_poses)):
    from matplotlib import animation

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    # Clamp to avoid complains
    im = plt.imshow(imgs[i][0].clamp(0.0, 1.0), animated=True)
    def update_fig(x):
        im.set_array(imgs[i][x].clamp(0.0, 1.0))
        return im,
    anim = animation.FuncAnimation(fig, update_fig, frames=len(imgs[i]), interval=300, blit=True)
    anim.save('process/anim0{}.gif'.format(i), writer='imagemagick')

print("Finish running!")