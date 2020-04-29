import torch
import pyredner
import numpy as np
import os
import sys

#obj = pyredner.load_obj("../process_no_smoothing/final.obj", return_objects=True)[0]
#obj = pyredner.load_obj("../cube.obj", return_objects=True)[0]

#bound = pyredner.bound_vertices(obj.vertices, obj.indices)

#for i in range(60):
#pyredner.smooth(obj.vertices, obj.indices, weighting_scheme='cotangent', lmd=0.5, control=bound)

#obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, weighting_scheme='cotangent')

#pyredner.save_obj(obj, "../new.obj")

# This program reconstruct the face from multi images and try to smooth
# output form changed



output_path = 'ppp'#sys.argv[1][:3] + '_' + sys.argv[2][:3] + '_' + sys.argv[3][:3]
normal_scheme = sys.argv[1]
smooth_scheme = sys.argv[2]
smooth_lmd = eval(sys.argv[3])


os.chdir('..')
os.system("rm -rf " + output_path)

pyredner.set_print_timing(False)
'''
tgt_vertices = torch.tensor([[-5, -1, -3],
                         [0, -5, 2],
                         [0, 4, 3],
                         [4, 0, -4]], dtype=torch.float32, device=pyredner.get_device(), requires_grad=False)
vertices = torch.tensor([[-2.4, -2.6, 0.2],
                         [3.1, -1.8, 0],
                         [0, 2.8, 1.1],
                         [4.1, 6.8, 2.1]], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
indices = torch.tensor([[0, 1, 2],
                        [2, 1, 3]], dtype=torch.int32, device=pyredner.get_device())
'''
obj = pyredner.load_obj('cube.obj', return_objects=True)[0]
vertices = obj.vertices * 8. - 4.
vertices.requires_grad = True
indices = obj.indices
tgt_vertices = vertices.detach() * 1.1 + 0.5 * torch.randn(vertices.shape, device=pyredner.get_device())
print(tgt_vertices)
cam_poses = torch.tensor([[0, 0, 20], [-12, 0, 16], [12, 0, 16]], dtype=torch.float32, device=pyredner.get_device(), requires_grad=False)
cam_look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=pyredner.get_device(), requires_grad=False)
dir_light_direction = torch.tensor([-0.0, -0.0, -1.0], device=pyredner.get_device(), requires_grad=False)
dir_light_intensity = torch.ones(3, device=pyredner.get_device(), requires_grad=False)
normals = pyredner.compute_vertex_normal(tgt_vertices, indices, normal_scheme)
print("finish loading")

def model(cam_pos, cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals):
   #normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)
    m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m)#, colors=colors)

    cam = pyredner.Camera(position=cam_pos,
                          look_at=cam_look_at,  # Center of the vertices
                          up=torch.tensor([0.0, 1.0, 0.0]),
                          fov=torch.tensor([45.0]),
                          resolution=(1000, 1000))
    scene = pyredner.Scene(camera=cam, objects=[obj])
    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(dir_light_direction, dir_light_intensity)
    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light], aa_samples=1)
    return img, obj

target = []
for i in range(len(cam_poses)):
    target.append(model(cam_poses[i], cam_look_at, tgt_vertices, torch.zeros(3), dir_light_intensity, dir_light_direction, normals)[0])
    #pyredner.imwrite(target[i].cpu(), output_path + '/target_img{:0>2d}.png'.format(i))

ambient_color = torch.zeros(3, device=pyredner.get_device(), requires_grad=False)

bound = 1. * pyredner.bound_vertices(vertices, indices) - 0.
ver_optimizer = torch.optim.Adam([vertices], lr=0.02)
#normals_optimizer = torch.optim.Adam([normals], lr=0.01)

import matplotlib.pyplot as plt

plt.figure()
losses, imgs, diffimgs = [], [], []
for i in range(len(cam_poses)):
    losses.append([])
    imgs.append([])
    diffimgs.append([])

num_iters_1 = 200
num_iters_2 = 60

for t in range(num_iters_2):
    total_loss = 0
    ver_optimizer.zero_grad()
    #normals_optimizer.zero_grad()
    normals = pyredner.compute_vertex_normal(vertices, indices, normal_scheme)

    for i in range(len(cam_poses)):
        img, obj = model(cam_poses[i], cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals)
        # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, so no gamma correction is needed.
        #imgs.append(img.numpy())
        loss = (img - target[i]).pow(2).mean()
        losses[i].append(loss.data.item())
        total_loss += loss

        imgs[i].append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        diffimgs[i].append(torch.where((img.data - target[i]).cpu() > 0,
                                       (img.data - target[i]).cpu() * torch.tensor([0., 10., 0.]),
                                       (img.data - target[i]).cpu() * torch.tensor([0., 0., -10.])))
        #pyredner.imwrite(abs(img - target[4]).data.cpu(), 'process/process2_img{:0>2d}.png'.format(t // 5))

    total_loss.backward()
    ver_optimizer.step()

    if smooth_scheme != 'None':# and t > 20:
        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)
        pyredner.smooth(vertices, indices, smooth_lmd, smooth_scheme, bound)

    if t == 200:
        ver_optimizer=torch.optim.Adam([vertices], lr=0.05)

    print("{:.^16}total_loss = {:.6f}".format(t, total_loss))
    print((normals - pyredner.compute_vertex_normal(vertices, indices, max)).pow(2).sum())

pyredner.save_obj(obj, output_path + '/final.obj')
print(output_path + '/final.obj')

for i in range(len(cam_poses)):
    img, obj = model(cam_poses[i], cam_look_at, vertices, ambient_color, dir_light_intensity, dir_light_direction, normals)
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

