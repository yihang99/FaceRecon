# This program reconstruct the face from multi images and try to smooth
# output form changed

import torch
import pyredner
import numpy as np
import os
import sys

output_path = 'p_color'#sys.argv[1][:3] + '_' + sys.argv[2][:3] + '_' + sys.argv[3][:3]
normal_scheme = sys.argv[1]
smooth_scheme = sys.argv[2]
smooth_lmd = eval(sys.argv[3])


os.chdir('..')
os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)

shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
indices = triangle_list.permute(1, 0).contiguous()
vertices = shape_mean.view(-1, 3)
if False:
    vertices, indices, uvs, normals = pyredner.generate_sphere(theta_steps = 256, phi_steps = 512)
    vertices = vertices * 120
vertices.requires_grad = True

target_data_path = "generated/dataset5/"
c_p, cam_look_at, dir_light_intensity, dir_light_direction = np.load(target_data_path+"env_data.npy", allow_pickle=True)
cam_poses = torch.tensor(c_p[:], requires_grad=False)

print("finish loading")

colors = color_mean.view(-1, 3)
colors.requires_grad = True
def model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors):
   #normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
    #m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    m = pyredner.Material(use_vertex_color=True)
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)

    cam = pyredner.Camera(position=cam_pos,
                          look_at=cam_look_at,  # Center of the vertices
                          up=torch.tensor([0.0, 1.0, 0.0]),
                          fov=torch.tensor([45.0]),
                          resolution=(1000, 1000))
    scene = pyredner.Scene(camera=cam, objects=[obj])
    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(dir_light_direction, dir_light_intensity)
    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light], aa_samples=1)
    return img

target = []
for i in range(len(cam_poses)):
    target.append(pyredner.imread(target_data_path+'target_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))

ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=False)

bound = 1. * pyredner.bound_vertices(vertices, indices) - 0.
ver_optimizer = torch.optim.Adam([vertices], lr=0.2)
colors_optimizer = torch.optim.Adam([colors], lr=0.01)

import matplotlib.pyplot as plt

plt.figure()
losses, imgs, diffimgs = [], [], []
for i in range(len(cam_poses)):
    losses.append([])
    imgs.append([])
    diffimgs.append([])

num_iters_1 = 10
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
        pyredner.imwrite(img.data.cpu(), output_path + '/process1_img{:0>2d}.png'.format(t // 20))
    print("{:.^20}".format(t))
'''

for t in range(num_iters_1):
    total_loss = 0
    ver_optimizer.zero_grad()
    colors_optimizer.zero_grad()
    #normals_optimizer.zero_grad()
    normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

    for i in range(len(cam_poses)):
        img = model(cam_poses[i], cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors)
        # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, so no gamma correction is needed.
        #imgs.append(img.numpy())
        loss = (img - target[i]).pow(2).mean()
        losses[i].append(loss.data.item())
        total_loss += loss

        imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        diffimgs[i].append(torch.where((img.data - target[i]).cpu() > 0,
                                       (img.data - target[i]).cpu() * torch.tensor([10., 10., 10.]),
                                       (img.data - target[i]).cpu() * torch.tensor([-10., -10., -10.])))
        #pyredner.imwrite(abs(img - target[4]).data.cpu(), 'process/process2_img{:0>2d}.png'.format(t // 5))

    total_loss.backward()
    ver_optimizer.step()
    colors_optimizer.step()
    vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + shape_mean.view(-1, 3) * (1. - bound).reshape(-1, 1).expand(-1, 3)
    #normals_optimizer.step()
    #normals.data = normals.data / normals.data.norm(dim=1).reshape(-1, 1).expand(-1, 3)
    if smooth_scheme != 'None':# and t > 20:
        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
        pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
        pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

    print("{:.^16}total_loss = {:.6f}".format(t, total_loss))

print()
m = pyredner.Material(use_vertex_color=True)
obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)
pyredner.save_obj(obj, output_path + '/final.obj')

for t in range(num_iters_2):
    total_loss = 0
    normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

    for i in range(len(cam_poses)):
        img = model(cam_poses[i], cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors)
        loss = (img - target[i]).pow(2).mean()
        losses[i].append(loss.data.item())
        total_loss += loss

        imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        diffimgs[i].append(torch.where((img.data - target[i]).cpu() > 0,
                                       (img.data - target[i]).cpu() * torch.tensor([10., 10., 10.]),
                                       (img.data - target[i]).cpu() * torch.tensor([-10., -10., -10.])))

    #vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + shape_mean.view(-1, 3) * (1. - bound).reshape(-1, 1).expand(-1, 3)

    if smooth_scheme != 'None':# and t > 20:
        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)

    print("{:.^16}total_loss = {:.6f}".format(num_iters_1 + t, total_loss))

obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)
pyredner.save_obj(obj, output_path + '/final_s.obj')
print(output_path + '/final_s.obj')

for i in range(len(cam_poses)):
    img = model(cam_poses[i], cam_look_at, vertices, ambient_color, dir_light_intensity, torch.tensor([1., -1., -1.]), normals, colors)
    pyredner.imwrite(img.data.cpu(), output_path + '/view0{}.png'.format(i))

    plt.plot(losses[i], label='view0{}'.format(i))

plt.legend()
plt.ylabel("loss")
plt.xlabel("iterations")
plt.savefig(output_path + "/lossCurve.png", dpi=800)
xlim = plt.xlim()
ylim = plt.ylim()

for i in range(len(cam_poses)):
    from matplotlib import animation

    fig, (img_plot, diff_plot, loss_curve) = plt.subplots(1, 3, figsize=(20, 10))
    im = img_plot.imshow(imgs[i][0].clamp(0.0, 1.0), animated=True)
    im_diff = diff_plot.imshow(diffimgs[i][0].clamp(0.0, 1.0), animated=True)
    loss_curve.set_xlim(xlim)
    loss_curve.set_ylim(ylim)
    lc, = loss_curve.plot([], [], 'b')

    def update_fig(x):
        im.set_array(imgs[i][x].clamp(0.0, 1.0))
        im_diff.set_array(diffimgs[i][x].clamp(0.0, 1.0))
        lc.set_data(range(x), losses[i][:x])
        return im, im_diff, lc

    anim = animation.FuncAnimation(fig, update_fig, frames=len(imgs[i]), interval=600, blit=True)
    anim.save(output_path + '/anim0{}.gif'.format(i), writer='imagemagick')

print("Finish running!")