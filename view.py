import torch
import pyredner
import numpy as np
import os
import sys

def add(x):
    return x+1

def view_face(cam_pos, cam_look_at, vertices, indices, ambient_color, dir_light_intensity, dir_light_directions, normals, colors):
    cam_pos = torch.tensor(cam_pos)
    cam_look_at = torch.tensor(cam_look_at)
    up = torch.tensor([0., 0., 1.])
    pos = cam_pos - cam_look_at
    relx = -torch.cross(pos, up)
    relx = relx / relx.norm()
    rely = torch.cross(pos, relx)
    rely = rely / rely.norm()
    len = pos.norm()
    angle = torch.tensor(0.5236)
    for i in range(-2, 3):
        pos_i = pos * torch.cos(i * angle) + len * relx * torch.sin(i * angle)
        for j in range(-2, 3):
            pos_j = pos_i * torch.cos(j * angle) + len * rely * torch.sin(i * angle)

            for k in range(1):
                dir_light_direction = torch.tensor(dir_light_directions[k])
                img = model(cam_look_at + pos_j, cam_look_at, vertices, indices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors)
                pyredner.imwrite(img.data.cpu(), 'views/view{}_{}_{}.png'.format(i+2, j+2, k))



def model(cam_pos, cam_look_at, vertices, indices, ambient_color, dir_light_intensity, dir_light_direction, normals, colors):
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

'''
def generate_views():

    m = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5]))
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m)#, colors=colors)
    pyredner.save_obj(obj, output_path + '/final_s.obj')
    print(output_path + '/final_s.obj')'''