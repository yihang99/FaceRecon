# This program reconstruct the face from multi images and try to smooth
# output form changed

import torch
import pyredner
import numpy as np
import os
import sys
import argparse

description = 'change rsl'

#<editor-fold desc="PARSE ARGUMENTS">
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

#</editor-fold>

os.chdir('..')
#os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)

#shape_mean, shape_basis, triangle_list, color_mean, color_basis = np.load("3dmm.npy", allow_pickle=True)
#indices = triangle_list.permute(1, 0).contiguous()
#vertices = shape_mean.view(-1, 3)
obj = pyredner.load_obj("p_ones30/final.obj", return_objects=True)[0]
indices = obj.indices.detach()
vertices = obj.vertices.detach()

tex = obj.material.diffuse_reflectance.texels

texels = pyredner.imresize(tex, (200, 200))
print('texels size: ', texels.size())
texels.requires_grad = True
m = pyredner.Material(diffuse_reflectance=texels)

vertices.requires_grad = True

target_data_path = "generated/dataset_ones30/"
cam_poses, cam_look_at, lights_list = np.load(target_data_path + "env_data.npy", allow_pickle=True)
#cam_poses = cam_poses[:1]

uvs = obj.uvs
uv_indices = obj.uv_indices

def model(cam_poses,
          cam_look_at,
          vertices,
          lights_list,
          normals,
          material):

    # m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices,
                          indices=indices,
                          normals=normals,
                          material=material,
                          uvs=uvs,
                          uv_indices=uv_indices)  # , colors=colors)
    imgs = []
    for i in range(cam_poses.size(0)):
        cam = pyredner.Camera(position=cam_poses[i],
                              look_at=cam_look_at,  # Center of the vertices
                              up=torch.tensor([0.0, 1.0, 0.0]),
                              fov=torch.tensor([45.0]),
                              resolution=(1000, 1000))
        scene = pyredner.Scene(camera=cam, objects=[obj])
        img = pyredner.render_deferred(scene=scene, lights=lights_list[i % len(lights_list)], aa_samples=1)
        imgs.append(img)
    return imgs


target = []
for i in range(len(cam_poses)):
    target.append(pyredner.imread(target_data_path + 'tgt_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))
    # pyredner.imwrite(target[i].cpu(), output_path + '/target_img{:0>2d}.png'.format(i))

ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=False)

bound = 1. * pyredner.bound_vertices(vertices, indices) - 0.
boundary = vertices.data * (1. - bound).reshape(-1, 1).expand(-1, 3)
ver_optimizer = torch.optim.Adam([vertices], lr=0.2)
tex_optimizer = torch.optim.Adam([texels], lr=0.05)

all_losses, img_losses, smooth_losses, total_losses, all_texels, all_imgs = [], [], [], [], [], []



print('Finish loading')

for t in range(num_iters_1):

    ver_optimizer.zero_grad()
    tex_optimizer.zero_grad()
    normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

    m = pyredner.Material(diffuse_reflectance=texels)
    imgs = model(cam_poses, cam_look_at, vertices, lights_list, normals, m)
    # imgs of all viewpoints
    #all_imgs.append(imgs)
    # record all imgs
    losses = torch.stack([(imgs[i] - target[i]).pow(2).mean() for i in range(len(imgs))])
    # losses of all imgs in this single iteration
    all_losses.append(losses)
    # all_losses records the losses in all iterations
    img_loss = losses.sum()

    smooth_loss = 0.1 * pyredner.smooth(vertices, indices, 0., smooth_scheme, bound, True).pow(2).mean()

    total_loss = img_loss # + smooth_loss
    total_losses.append(total_loss)
    img_losses.append(img_loss)
    smooth_losses.append(smooth_loss)
    all_texels.append(texels.data.cpu())

    total_loss.backward()
    ver_optimizer.step()
    tex_optimizer.step()
    vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + boundary

    if 1:#smooth_scheme != 'none':  # and t > 20:smi
        for num_of_smooth in range(2):
            pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
            pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

    print("{:.^8}total_loss:{:.6f}...img_loss:{:.6f}...smooth_loss:{:.6f}".format(t, total_loss, img_loss, smooth_loss))

    if 0:#(2 * (t + 1)) % num_iters_1 == 0 and t + 1 < num_iters_1:
        print(t)
        texels = pyredner.imresize(texels, scale=2.).detach()
        texels.requires_grad = True
        tex_optimizer = torch.optim.Adam([texels], lr=0.05)
        print(texels.size())

all_losses = torch.stack(all_losses).detach().cpu()
total_losses = torch.stack(total_losses).detach().cpu()
img_losses = torch.stack(img_losses).detach().cpu()
smooth_losses = torch.stack(smooth_losses).detach().cpu()

print()

obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, uvs=uvs, uv_indices=uv_indices)  # , colors=colors)
pyredner.save_obj(obj, output_path + '/final.obj')
#pyredner.imwrite(texels.data.cpu(), output_path + '/texels.png')
'''
import matplotlib.pyplot as plt

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=13, color='blue')

plt.sca(ax1)

imgs = model(cam_poses, cam_look_at, vertices, lights_list, normals, m)

for i in range(len(cam_poses)):
    pyredner.imwrite(imgs[i].data.cpu(), output_path + '/final_views/view{:0>2d}.png'.format(i))
    plt.plot(all_losses[:, i], label='view{:0>2d}:{:.6f}'.format(i, all_losses[-1, i]))

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
xxlim = plt.xlim()
yylim = plt.ylim()
plt.savefig(output_path + "/lossCurve.png", dpi=800)

print("lossCurve printed")

from matplotlib import animation

for i in range(len(cam_poses)):

    fig, (img_plot, diff_plot, loss_curve) = plt.subplots(1, 3, figsize=(16, 8))

    plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=16, color='blue')

    im = img_plot.imshow(torch.pow(all_imgs[0][i], 1.0 / 2.2).clamp(0.0, 1.0).detach().cpu(), animated=True)
    im_diff = diff_plot.imshow(abs(all_imgs[0][i] - target[i]).clamp(0.0, 1.0).detach().cpu() * 10., animated=True)
    loss_curve.set_xlim(xlim)
    loss_curve.set_ylim(ylim)
    lc, = loss_curve.plot([], [], 'b')

    def update_fig(x):
        im.set_array(torch.pow(all_imgs[x][i], 1.0 / 2.2).clamp(0.0, 1.0).detach().cpu())
        im_diff.set_array(abs(all_imgs[x][i] - target[i]).clamp(0.0, 1.0).detach().cpu())
        lc.set_data(range(x), all_losses[:x, i])
        return im, im_diff, lc

    anim = animation.FuncAnimation(fig, update_fig, frames=len(all_imgs), interval=300, blit=True)
    anim.save(output_path + '/anim{:0>2d}.gif'.format(i), writer='imagemagick')
    print('anim{:0>2d}.gif generated'.format(i))



fig, (tex_plot, loss_curve) = plt.subplots(1, 2, figsize=(16, 8))

plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=16, color='blue')

im = tex_plot.imshow(all_texels[i].clamp(0.0, 1.0), animated=True)
loss_curve.set_xlim(xxlim)
loss_curve.set_ylim(yylim)
lc, = loss_curve.plot([], [], 'b')

def update_fig(x):
    im.set_array(all_texels[x].clamp(0.0, 1.0))
    lc.set_data(range(x), total_losses[:x])
    return im, lc

anim = animation.FuncAnimation(fig, update_fig, frames=len(all_texels), interval=300, blit=True)
anim.save(output_path + '/anim_texels.gif'.format(i), writer='imagemagick')
print('anim_texels.gif generated'.format(i))

print("Finish running!")
'''