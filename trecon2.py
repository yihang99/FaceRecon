# This program reconstruct the face from multi images and try to smooth
# output form changed

import torch
import pyredner
import numpy as np
import os
import sys
import argparse

description = 'mouth stitched'

parser = argparse.ArgumentParser(prog='face recon',
                                 description='reconstruct the face from images',
                                 epilog='enjoy!')
parser.add_argument('-ns',
                    '--normal_scheme',
                    action='store',
                    choices=['max', 'cotangent'],
                    type=str,
                    help='scheme of computing vertex normal',
                    default='max')
parser.add_argument('-ss',
                    '--smooth_scheme',
                    action='store',
                    choices=['uniform', 'reciprocal', 'cotangent', 'none'],
                    type=str,
                    help='scheme of smoothing',
                    default='uniform')
parser.add_argument('-lmd',
                    '--smooth_lmd',
                    action='store',
                    type=float,
                    help='lambda value in smoothing',
                    default=0.5)
parser.add_argument('-p',
                    '--output_path',
                    action='store',
                    type=str,
                    help='path to output results',
                    default='p_default')
parser.add_argument('-ni',
                    '--num_iters_1',
                    action='store',
                    type=int,
                    help='number of iterations',
                    default=30)

args = parser.parse_args()
normal_scheme = args.normal_scheme
smooth_scheme = args.smooth_scheme
smooth_lmd = args.smooth_lmd
output_path = args.output_path
num_iters_1 = args.num_iters_1
print(vars(args))

os.chdir('..')
os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)

shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
#indices = triangle_list.permute(1, 0).contiguous()
#vertices = shape_mean.view(-1, 3)
obj = pyredner.load_obj("new_init.obj", return_objects=True)[0]
indices = obj.indices.detach()
vertices = obj.vertices.detach()

if 0:
    vertices, indices, uvs, normals = pyredner.generate_sphere(theta_steps=256, phi_steps=512)
    vertices = vertices * 120
vertices.requires_grad = True

target_data_path = "generated/dataset2/"
cam_poses, cam_look_at, dir_light_intensity, dir_light_directions = np.load(target_data_path + "env_data.npy",
                                                                            allow_pickle=True)
#cam_poses = cam_poses[:1]

def model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals):
    # normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
    m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m)  # , colors=colors)

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
    target.append(pyredner.imread(target_data_path + 'target_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))
    # pyredner.imwrite(target[i].cpu(), output_path + '/target_img{:0>2d}.png'.format(i))

ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=False)

bound = 1. * pyredner.bound_vertices(vertices, indices) - 0.
boundary = vertices.data * (1. - bound).reshape(-1, 1).expand(-1, 3)
ver_optimizer = torch.optim.Adam([vertices], lr=0.2)

import matplotlib.pyplot as plt

plt.figure()
losses, img_losses, smooth_losses, total_losses, imgs, diffimgs = [], [], [], [], [], []
for i in range(len(cam_poses)):
    losses.append([])
    imgs.append([])
    diffimgs.append([])

#num_iters_1 = 30
num_iters_2 = 0

print('Finish loading')

for t in range(num_iters_1):
    img_loss = 0
    ver_optimizer.zero_grad()
    # normals_optimizer.zero_grad()
    normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

    for i in range(len(cam_poses)):
        cam_pos = torch.tensor(cam_poses[i])
        dir_light_direction = torch.tensor(dir_light_directions[i % len(dir_light_directions)])
        img = model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals)
        # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, so no gamma correction is needed.
        # imgs.append(img.numpy())
        loss = (img - target[i]).pow(2).mean()
        losses[i].append(loss.data.item())
        img_loss += loss

        imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        diffimgs[i].append(torch.where((img.data - target[i]).cpu() > 0,
                                       (img.data - target[i]).cpu() * torch.tensor([0., 10., 0.]),
                                       (img.data - target[i]).cpu() * torch.tensor([0., 0., -10.])))
        # pyredner.imwrite(abs(img - target[4]).data.cpu(), 'process/process2_img{:0>2d}.png'.format(t // 5))

    smooth_loss = 0.03 * pyredner.smooth(vertices, indices, 0., smooth_scheme, bound, True)
    #smooth_loss = 0.000001 * pyredner.area(vertices, indices)
    total_loss = img_loss #+ smooth_loss
    total_losses.append(total_loss.data.item())
    img_losses.append(img_loss.data.item())
    smooth_losses.append(smooth_loss.data.item())

    total_loss.backward()
    ver_optimizer.step()
    vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + boundary
    # normals_optimizer.step()
    # normals.data = normals.data / normals.data.norm(dim=1).reshape(-1, 1).expand(-1, 3)
    if 1:#smooth_scheme != 'none':  # and t > 20:smi
        for num_of_smooth in range(2):
            pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
            pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

    print("{:.^10}total_loss:{:.6f}...img_loss:{:.6f}...smooth_loss:{:.6f}".format(t, total_loss, img_loss, smooth_loss))

print()
m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m)  # , colors=colors)
pyredner.save_obj(obj, output_path + '/final.obj')

if num_iters_2 > 0:
    for t in range(num_iters_2):
        total_loss = 0
        normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

        for i in range(len(cam_poses)):
            cam_pos = torch.tensor(cam_poses[i])
            dir_light_direction = torch.tensor(dir_light_directions[i % len(dir_light_directions)])
            img = model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals)
            loss = (img - target[i]).pow(2).mean()
            losses[i].append(loss.data.item())
            total_loss += loss

            imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
            diffimgs[i].append(torch.where((img.data - target[i]).cpu() > 0,
                                           (img.data - target[i]).cpu() * torch.tensor([0., 10., 0.]),
                                           (img.data - target[i]).cpu() * torch.tensor([0., 0., -10.])))

        # vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + shape_mean.view(-1, 3) * (1. - bound).reshape(-1, 1).expand(-1, 3)

        if smooth_scheme != 'None':  # and t > 20:
            pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)

        print("{:.^16}total_loss = {:.6f}".format(num_iters_1 + t, total_loss))

    m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m)  # , colors=colors)
    pyredner.save_obj(obj, output_path + '/final_s.obj')
    print(output_path + '/final_s.obj')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=13, color='blue')

plt.text(0.33, 0.9, 'txt')
plt.sca(ax1)

for i in range(len(cam_poses)):
    cam_pos = torch.tensor(cam_poses[i])
    dir_light_direction = torch.tensor(dir_light_directions[i % len(dir_light_directions)])
    img = model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals)
    pyredner.imwrite(img.data.cpu(), output_path + '/view{:0>2d}.png'.format(i))

    plt.plot(losses[i], label='view{:0>2d}:{:.6f}'.format(i, losses[i][-1]))

plt.legend()
plt.ylabel("losses")
plt.xlabel("iterations")

plt.ylim(ymin=0.)
xlim = plt.xlim()
ylim = plt.ylim()

plt.sca(ax2)
plt.plot(total_losses, label="total_loss:{:.6f}".format(total_losses[-1]))
plt.plot(img_losses, label="img_loss:{:.6f}".format(img_losses[-1]))
plt.plot(smooth_losses, label="smooth_loss:{:.6f}".format(smooth_losses[-1]))
plt.legend()
plt.xlabel("iterations")
plt.ylim(ymin=0.)
plt.savefig(output_path + "/lossCurve.png", dpi=800)

print("lossCurve printed")

for i in range(len(cam_poses)):
    from matplotlib import animation

    fig, (img_plot, diff_plot, loss_curve) = plt.subplots(1, 3, figsize=(16, 8))

    plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=16, color='blue')

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
    anim.save(output_path + '/anim{:0>2d}.gif'.format(i), writer='imagemagick')

    print('anim{:0>2d}.gif generated'.format(i))




up = torch.tensor([0., 1., 0.])
pos = cam_poses[0] - cam_look_at
relx = -torch.cross(pos, up)
relx = relx / relx.norm()
rely = torch.cross(pos, relx)
rely = rely / rely.norm()
len = pos.norm()
angle = torch.tensor(0.5236)
dir_light_directions = [[-1., 0., -1.], [1., 0., -1.]]
for i in range(-1, 2):
    pos_i = pos * torch.cos(i * angle) + len * relx * torch.sin(i * angle)
    for j in range(-1, 2):
        pos_j = pos_i * torch.cos(j * angle) + len * rely * torch.sin(j * angle)

        for k in range(2):
            dir_light_direction = torch.tensor(dir_light_directions[k])
            cam_pos = cam_look_at + pos_j
            img = model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors)
            pyredner.imwrite(img.data.cpu(), output_path+'/views/view{}_{}_{}.png'.format(i+2, j+2, k))

print("Finish running!")