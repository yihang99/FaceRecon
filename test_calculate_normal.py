import torch
import pyredner
import math

import urllib
import zipfile
def smooth(vertices, indices, lmd):
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

def dot(v1, v2):
    return torch.sum(v1 * v2, dim=1)

def squared_length(v):
    return torch.sum(v * v, dim=1)

def length(v):
    return torch.sqrt(squared_length(v))

def safe_asin(v):
    # Hack: asin(1)' is infinite, so we want to clamp the contribution
    return torch.asin(v.clamp(0, 1 - 1e-6))

objs = pyredner.load_obj('smoothed.obj', return_objects=True)
obj = objs[0]
vertices = obj.vertices
indices = obj.indices

normals = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)
v = [vertices[indices[:, 0].long(), :],  # all the 0th vertices of triangles
     vertices[indices[:, 1].long(), :],  # all the 1st vertices of triangles
     vertices[indices[:, 2].long(), :]]
for i in range(3):  # 0th, 1st and 2nd
    v0 = v[i]
    v1 = v[(i + 1) % 3]
    v2 = v[(i + 2) % 3]
    e1 = v1 - v0
    e2 = v2 - v0
    e1_len = length(e1)  # lengths of the first edges
    e2_len = length(e2)  # lengths of the second edges
    side_a = e1 / torch.reshape(e1_len, [-1, 1])
    side_b = e2 / torch.reshape(e2_len, [-1, 1])
    if i == 0: # compute normal for one time
        n = torch.cross(side_a, side_b)
        n = torch.where(length(n).reshape(-1, 1).expand(-1, 3) > 0,  # usually all true
                        n / torch.reshape(length(n), [-1, 1]),
                        torch.zeros(n.shape, dtype=n.dtype, device=n.device))
    angle = torch.where(dot(side_a, side_b) < 0,  # usually all false
                        math.pi - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                        2.0 * safe_asin(0.5 * length(side_b - side_a)))
    sin_angle = torch.sin(angle)

    # XXX: Inefficient but it's PyTorch's limitation
    e1e2 = e1_len * e2_len
    # contrib is 0 when e1e2 is 0
    contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                          n * (sin_angle / e1e2).reshape(-1, 1).expand(-1, 3),
                          # n is 3d vector for each triangle, so expand the latter term weight into 3d vectors
                          torch.zeros(n.shape, dtype=torch.float32, device=vertices.device))
    # contrib is a specific tensor matrix for say, the 0th vertex of all triangles, the total contribution (in 3d vector)
    # of this triangle to the normal of this vertex. So it is a (num of triangles) * 3 tensor
    index = indices[:, i].long().reshape(-1, 1).expand(-1, 3)
    # index is the indexes of the (say 0th) vertex of all triangles, (num * 3) tensor
    normals.scatter_add_(0, index, contrib)
    # the normals without normalization:
    # normals[index[i][j]] += contrib[i]  # if dim == 0
    # index[i][j] gives the jth vertex of ith triangle, normals[index[i][j]] is the corresponding normal vector

# Assign 0, 0, 1 to degenerate faces
degenerate_normals = torch.zeros(normals.shape, dtype=torch.float32, device=vertices.device)
degenerate_normals[:, 2] = 1.0
normals = torch.where(length(normals).reshape(-1, 1).expand(-1, 3) > 0,  # usually all true
                      normals / torch.reshape(length(normals), [-1, 1]),  # normalize the normals
                      degenerate_normals)
assert (torch.isfinite(normals).all())


'''
obj.laplacian_smooth(1)
pyredner.save_obj(obj, 'smoothed2.obj')
'''

'''this is a note'''