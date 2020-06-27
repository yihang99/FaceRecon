# This program reconstruct the face from multi images and try to smooth
# output form changed

import torch
import pyredner
import numpy as np
import os
import sys
import argparse

description = 'env_map'
target_data_path = "generated/env_dataset_30_5/"

# <editor-fold desc="PARSE ARGUMENTS">
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


# </editor-fold>

def visual_vertex_grad(vertices: torch.Tensor, indices: torch.Tensor, cam: pyredner.Camera = None):
    if not hasattr(visual_vertex_grad, 'x'):
        visual_vertex_grad.x = 0
    else:
        visual_vertex_grad.x += 1
    cam = pyredner.Camera(position=cam_pos,
                          look_at=cam_look_at,  # Center of the vertices
                          up=torch.tensor([0.0, 1.0, 0.0]),
                          fov=torch.tensor([45.0]),
                          resolution=(1000, 1000))
    m = pyredner.Material(use_vertex_color=True)
    obj = pyredner.Object(vertices=vertices, indices=indices, material=m)
    coe = 500000.
    color_reps = torch.tensor([[[1., 0., 0.],
                                [0., -1., -1.]],

                              [[0., 1., 0.],
                               [-1., 0., -1.]],

                              [[0., 0., 1.],
                               [-1., -1., 0.]]]).to(pyredner.get_device())
    grad_imgs = []
    for d in range(3):
        colors = torch.where(vertices.grad[:, d:d+1].expand(-1, 3) > 0,
                             vertices.grad[:, d:d+1].expand(-1, 3) * color_reps[d, 0],
                             vertices.grad[:, d:d+1].expand(-1, 3) * color_reps[d, 1]) * coe

        obj.colors = colors
        scene = pyredner.Scene(camera=cam, objects=[obj])
        grad_imgs.append(pyredner.render_albedo(scene=scene))
    for d in range(3):
        pyredner.imwrite(grad_imgs[d].cpu(), output_path + '/grad_imgs/{:0>2d}{:0>2d}.png'.format(d, visual_vertex_grad.x))
    return grad_imgs



# <editor-fold desc="LOADING DATA">
os.chdir('..')
os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)

obj = pyredner.load_obj("init/final.obj", return_objects=True)[0]
indices = obj.indices.detach().clone()
vertices = obj.vertices.detach().clone()
v = vertices.clone()
ideal_shift = pyredner.smooth(vertices, indices, 0., smooth_scheme, return_shift=True)
ideal_quad_shift = pyredner.smooth(ideal_shift, indices, 0., smooth_scheme, return_shift=True)
tex = obj.material.diffuse_reflectance.texels
texels = pyredner.imresize(tex, (200, 200))
print('texels size: ', texels.size())
texels.requires_grad = True
uvs = obj.uvs
uv_indices = obj.uv_indices

m = pyredner.Material(diffuse_reflectance=texels)

vertices.requires_grad = True


cam_pos, cam_look_at, num_views, all_euler_angles, center, resolution = np.load(target_data_path + "env_data.npy",
                                                                                allow_pickle=True)
# cam_poses = cam_poses[:1]
center = center.to(pyredner.get_device())

target = []
for i in range(num_views):
    target.append(pyredner.imread(target_data_path + 'tgt_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))

envmap_img = pyredner.imread(target_data_path + 'env_map.png')
envmap = pyredner.EnvironmentMap(envmap_img)

print('Finish loading')
# </editor-fold>


# <editor-fold desc="CORRECTING POSITION">
euler_list, trans_list = [], []
euler = torch.tensor([0.0, 0., 0.], device=pyredner.get_device(), requires_grad=True)
trans = torch.tensor([-0., -0., -0.], device=pyredner.get_device(), requires_grad=True)
# eul_optimizer = torch.optim.SGD([euler], lr=2)
# tra_optimizer = torch.optim.SGD([trans], lr=2000)
# eul_optimizer = torch.optim.Adam([euler], lr=0.02)
cam = pyredner.Camera(position=cam_pos,
                      look_at=cam_look_at,  # Center of the vertices
                      up=torch.tensor([0.0, 1.0, 0.0]),
                      fov=torch.tensor([20.0]),
                      resolution=resolution)
for i in range(0):  # num_views):
    print("correcting position {:0>2d}".format(i))
    eul_optimizer = torch.optim.SGD([euler], lr=2)
    tra_optimizer = torch.optim.SGD([trans], lr=5000)
    for t in range(20):
        eul_optimizer.zero_grad()
        tra_optimizer.zero_grad()
        rotation_matrix = pyredner.gen_rotate_matrix(euler)
        obj.vertices = (vertices - center) @ torch.t(rotation_matrix) \
                       + center + trans * torch.tensor([1., 1., 3.], device=pyredner.get_device())
        scene = pyredner.Scene(objects=[obj], camera=cam, envmap=envmap)
        img = pyredner.render_pathtracing(scene=scene, num_samples=(64, 4), use_secondary_edge_sampling=True)
        print('f')
        loss = (img - target[i]).pow(2).mean()
        loss.backward()
        eul_optimizer.step()
        tra_optimizer.step()
        if t % 2 == 1:
            print('    iteration', t, 'loss:{:.6f}'.format(loss), euler.data.cpu(),
                  trans.data.cpu() * torch.tensor([1., 1., 3.]))

    euler_list.append(euler.data.clone())
    trans_list.append(trans.data.clone())

    #pyredner.imwrite(img.cpu(), output_path + '/view_positions/{:0>2d}.png'.format(i))

euler_list = torch.stack(all_euler_angles).to(pyredner.get_device())\
             + 0.01*torch.ones(num_views, 3).to(pyredner.get_device())
#euler_list = euler_list[0:2]
#num_views = 2

euler_list.requires_grad = True
trans_list = torch.zeros((num_views, 3), device=pyredner.get_device())

trans_list.requires_grad = True

print('view positions corrected!')
# </editor-fold>

bound = pyredner.bound_vertices(vertices, indices)
boundary = vertices.data * (1.0 - bound).reshape(-1, 1).expand(-1, 3)

ver_optimizer = torch.optim.Adam([vertices], lr=0.2e0)
tex_optimizer = torch.optim.Adam([texels], lr=0.03)
eul_optimizer = torch.optim.SGD([euler_list], lr=1*3)
tra_optimizer = torch.optim.SGD([trans_list], lr=2000*3)
#eul_optimizer = torch.optim.Adam([euler_list], lr=0.005)
#tra_optimizer = torch.optim.Adam([trans_list], lr=0.2)
all_losses, img_losses, smooth_losses, total_losses, all_texels, all_imgs = [], [], [], [], [], []
print((vertices - v).pow(2).mean())
for t in range(num_iters_1):
    eul_optimizer.zero_grad()
    tra_optimizer.zero_grad()
    ver_optimizer.zero_grad()
    tex_optimizer.zero_grad()
    obj.normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
    obj.material = pyredner.Material(diffuse_reflectance=texels)
    imgs = []

    for i in range(num_views):
        euler = euler_list[i]
        trans = trans_list[i]
        rotation_matrix = pyredner.gen_rotate_matrix(euler)
        obj.vertices = (vertices - center) @ torch.t(rotation_matrix) \
                       + center + trans * torch.tensor([1., 1., 3.], device=pyredner.get_device())
        scene = pyredner.Scene(objects=[obj], camera=cam, envmap=envmap)
        img = pyredner.render_pathtracing(scene=scene, num_samples=(32, 4), use_secondary_edge_sampling=False)
        imgs.append(img)


    all_imgs.append(imgs)
    # record all imgs
    losses = torch.stack([(imgs[i] - target[i]).pow(2).mean() for i in range(len(imgs))])
    # losses of all imgs in this single iteration
    all_losses.append(losses)
    # all_losses records the losses in all iterations
    img_loss = losses.sum()

    shift = pyredner.smooth(vertices, indices, 0., smooth_scheme, bound, True)
    smooth_loss = num_views*0.005*(shift - ideal_shift).pow(2).mean()
    # ideal_shift = (ideal_shift * 0.9 + shift * 0.1).detach()
    # smooth_loss = 0.1 * num_views * (pyredner.smooth(normals, indices, 0., smooth_scheme, bound, True)).pow(2).mean()

    total_loss = img_loss  + smooth_loss
    total_losses.append(total_loss)
    img_losses.append(img_loss)
    smooth_losses.append(smooth_loss)
    all_texels.append(texels.data.cpu())

    total_loss.backward()
    #vertices.grad.data = vertices.grad.data.clamp(-5.0e-7, 5.0e-7)
    if t > 5:
        ver_optimizer.step()
        tex_optimizer.step()
        eul_optimizer = torch.optim.SGD([euler_list], lr=0.2*5)
        tra_optimizer = torch.optim.SGD([trans_list], lr=500*5)
    eul_optimizer.step()
    tra_optimizer.step()
    #visual_vertex_grad(vertices, indices)
    #vertices.data = vertices.data * bound.reshape(-1, 1).expand(-1, 3) + boundary

    if t > 5:  # smooth_scheme != 'none':  # and t > 20:smi
        for num_of_smooth in range(2):
            pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
            pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

    print("{:.^8}total_loss:{:.6f}...img_loss:{:.6f}...smooth_loss:{:.6f}".format(t, total_loss, img_loss, smooth_loss))
    print('{:.6f}'.format(shift.pow(2).mean().item()))
    #print(euler_list)
    #print(trans_list)
    if 0:  # (2 * (t + 1)) % num_iters_1 == 0 and t + 1 < num_iters_1:
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

obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, uvs=uvs,
                      uv_indices=uv_indices)  # , colors=colors)
pyredner.save_obj(obj, output_path + '/final.obj')
# pyredner.imwrite(texels.data.cpu(), output_path + '/texels.png')

import matplotlib.pyplot as plt

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=13, color='blue')

plt.sca(ax1)

for i in range(num_views):
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

for i in range(num_views):
    fig, (img_plot, diff_plot, loss_curve) = plt.subplots(1, 3, figsize=(16, 8))

    plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=16, color='blue')

    im = img_plot.imshow(torch.pow(all_imgs[0][i], 1.0 / 2.2).clamp(0.0, 1.0).detach().cpu(), animated=True)
    im_diff = diff_plot.imshow(abs(all_imgs[0][i] - target[i]).clamp(0.0, 0.1).detach().cpu() * 10., animated=True)
    loss_curve.set_xlim(xlim)
    loss_curve.set_ylim(ylim)
    lc, = loss_curve.plot([], [], 'b')


    def update_fig(x):
        im.set_array(torch.pow(all_imgs[x][i], 1.0 / 2.2).clamp(0.0, 1.0).detach().cpu())
        im_diff.set_array(abs(all_imgs[x][i] - target[i]).clamp(0.0, 0.1).detach().cpu() * 10.)
        lc.set_data(range(x), all_losses[:x, i])
        return im, im_diff, lc


    anim = animation.FuncAnimation(fig, update_fig, frames=len(all_imgs), interval=300, blit=True)
    anim.save(output_path + '/anim{:0>2d}.gif'.format(i), writer='imagemagick')
    print('anim{:0>2d}.gif generated'.format(i))

fig, (tex_plot, loss_curve) = plt.subplots(1, 2, figsize=(16, 8))

plt.suptitle(description + '\n' + str(vars(args))[1:-1].replace("'", ""), fontsize=16, color='blue')

im = tex_plot.imshow(all_texels[0].clamp(0.0, 1.0), animated=True)
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
