# This program reconstruct the face from multi images and try to smooth
# fixed camera position, changing face position
import math

import torch
import pyredner
import numpy as np
import os
import sys
import argparse

description = 'envmap_changing camera position'
target_data_path = "generated/senv_dataset_5views2/"

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
parser.add_argument('-anim',
                    '--store_animation',
                    action='store_true',
                    help='store_animation or not')

args = parser.parse_args()
normal_scheme = args.normal_scheme
smooth_scheme = args.smooth_scheme
smooth_lmd = args.smooth_lmd
output_path = args.output_path
num_iters_1 = args.num_iters_1
store_animation = args.store_animation
print(vars(args))


# </editor-fold>
def compute_lap(texels):
    a = torch.cat((texels[-1:, :], texels[:-1, :]), dim=0)  # down
    b = torch.cat((texels[1:, :], texels[:1, :]), dim=0)  # up
    c = torch.cat((texels[:, -1:, :], texels[:, :-1, :]), dim=1)  # right
    d = torch.cat((texels[:, 1:, :], texels[:, :1, :]), dim=1)  # left
    return (a + b + c + d) / 4.0 - texels

def compute_nab(texels):
    b = torch.cat((texels[1:, :], texels[:1, :]), dim=0)  # up
    c = torch.cat((texels[:, -1:, :], texels[:, :-1, :]), dim=1)  # right
    return ((c - texels).pow(2) + (b - texels).pow(2)).sum(dim=2)

# <editor-fold desc="LOADING DATA">
os.chdir('..')
#os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)

obj = pyredner.load_obj("init2/final.obj", return_objects=True)[0]
indices = obj.indices.detach().clone()
vertices = obj.vertices.detach().clone()
v = vertices.clone()
ideal_shift = pyredner.smooth(vertices, indices, 0., smooth_scheme, return_shift=True)

tex = obj.material.diffuse_reflectance.texels
texels = tex.clone()  #pyredner.imresize(tex, (200, 200))
ideal_lap = compute_lap(texels)
ideal_nab = compute_nab(texels)
pyredner.imwrite(ideal_nab.unsqueeze(2).expand(-1, -1, 3), 'nab.png')

print('texels size: ', texels.size())
texels.requires_grad = True
uvs = obj.uvs
uv_indices = obj.uv_indices

m = pyredner.Material(diffuse_reflectance=texels, specular_reflectance=torch.tensor([0.05, 0.05, 0.05]), roughness=torch.tensor([0.02]))

vertices.requires_grad = True


cam_poses, cam_look_ats, resolution = np.load(target_data_path + "env_data.npy", allow_pickle=True)
num_views = len(cam_poses)

target = []
for i in range(num_views):
    target.append(pyredner.imread(target_data_path + 'tgt_img{:0>2d}.png'.format(i)).to(pyredner.get_device()))

tgt_envmap_img = pyredner.imread(target_data_path + 'env_map.png')
envmap_img = (torch.zeros((64, 128, 3), dtype=torch.float32) + 0.5).detach()
envmap_img.requires_grad = True
envmap = pyredner.EnvironmentMap(envmap_img)

print('Finish loading')
# </editor-fold>

def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    #deringed_coeffs[:, 9:9 + 7] += \
       # coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
    return deringed_coeffs

coeffs = torch.zeros(3, 9).to(pyredner.get_device())
coeffs[:, 0] += 0.5
coeffs.requires_grad = True

bound = pyredner.bound_vertices(vertices, indices)
boundary = vertices.data * (1.0 - bound).reshape(-1, 1).expand(-1, 3)

ver_optimizer = torch.optim.Adam([vertices], lr=0.1e0)
tex_optimizer = torch.optim.Adam([texels], lr=0.01)

env_optimizer = torch.optim.SGD([coeffs], lr=5)

all_losses, img_losses, smooth_losses, total_losses, all_texels, all_imgs = [], [], [], [], [], []
print((vertices - v).pow(2).mean())

nab_loss=0

for t in range(num_iters_1):

    ver_optimizer.zero_grad()
    tex_optimizer.zero_grad()
    env_optimizer.zero_grad()
    #normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
    obj.material = pyredner.Material(diffuse_reflectance=texels, specular_reflectance=torch.tensor([0.05, 0.05, 0.05]), roughness=torch.tensor([0.02]))
    imgs = []
    deringed_coeffs = deringing(coeffs, 6.0)
    # envmap_img = pyredner.SH_reconstruct(deringed_coeffs, (64, 128))
    envmap_img = tgt_envmap_img
    envmap = pyredner.EnvironmentMap(envmap_img)
    pyredner.imwrite(envmap_img.cpu(), output_path + '/env{:0>2d}.png'.format(t))
    pyredner.imwrite(texels.cpu(), output_path + '/tex{:0>2d}.png'.format(t))
    obj.vertices = vertices
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices)
    for i in range(num_views):
        cam = pyredner.Camera(position=cam_poses[i],
                              look_at=cam_look_ats[i % len(cam_look_ats)],  # Center of the vertices
                              up=torch.tensor([0.0, 1.0, 0.0]),
                              fov=torch.tensor([45.0]),
                              resolution=resolution)
        scene = pyredner.Scene(objects=[obj], camera=cam, envmap=envmap)
        img = pyredner.render_pathtracing(scene=scene, num_samples=(32, 4), use_secondary_edge_sampling=False)
        imgs.append(img)

        pyredner.imwrite(img.cpu(), output_path + '/iter{:0>2d}_{:0>2d}.png'.format(t, i))

    all_imgs.append(imgs)
    # record all imgs
    losses = torch.stack([(imgs[i] - target[i]).pow(2).mean() for i in range(len(imgs))])
    # losses of all imgs in this single iteration
    all_losses.append(losses)
    # all_losses records the losses in all iterations
    img_loss = losses.sum()

    shift = pyredner.smooth(vertices, indices, 0., smooth_scheme, bound, True)
    smooth_loss = num_views*0.05*(shift - ideal_shift).pow(2).mean()

    lap = compute_lap(texels)
    nab = compute_nab(texels)
    print(nab.shape)
    lap_loss = 0.8 * (lap - ideal_lap).pow(2).mean()
    nab_loss = 0.05 * (nab - ideal_nab).clamp(min=0.0).mean()
    pyredner.imwrite(nab.unsqueeze(2).expand(-1, -1, 3), output_path + '/nab{:0>2d}.png'.format(t))
    total_loss = img_loss + smooth_loss # + nab_loss


    total_losses.append(total_loss)
    img_losses.append(img_loss)
    smooth_losses.append(smooth_loss)
    all_texels.append(texels.data.cpu())

    total_loss.backward()


    #visual_vertex_grad(vertices, indices)

    ver_optimizer.step()
    tex_optimizer.step()
    env_optimizer.step()
    for num_of_smooth in range(2):
        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
        pyredner.smooth(vertices, indices, -smooth_lmd, smooth_scheme, bound)

    print("{:.^8}total_loss:{:.6f}...img_loss:{:.6f}...smooth_loss:{:.6f}...nab_loss:{:.6f}".format(t, total_loss,
                                                                                                    img_loss,
                                                                                                    smooth_loss,
                                                                                                    nab_loss))
    print('average shift amount: {:.6f}'.format(shift.pow(2).mean().item()))

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
normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, uvs=uvs,
                      uv_indices=uv_indices)  # , colors=colors)
pyredner.save_obj(obj, output_path + '/final.obj')
# pyredner.imwrite(texels.data.cpu(), output_path + '/texels.png')

import matplotlib.pyplot as plt

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
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

if store_animation:

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
        anim.save(output_path + '/anims/anim{:0>2d}.gif'.format(i), writer='imagemagick')
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
    anim.save(output_path + '/anims/anim_texels.gif'.format(i), writer='imagemagick')
    print('anim_texels.gif generated'.format(i))

print("Finish running!")
