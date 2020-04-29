import torch
import pyredner
import numpy as np
import os
import sys

obj = pyredner.load_obj("../process_test9/final.obj", return_objects=True)[0]
#obj = pyredner.load_obj("../cube.obj", return_objects=True)[0]

bound = pyredner.bound_vertices(obj.vertices, obj.indices)

for i in range(3):
    pyredner.smooth(obj.vertices, obj.indices, weighting_scheme='uniform', lmd=0.5, control=bound)
    pyredner.smooth(obj.vertices, obj.indices, weighting_scheme='uniform', lmd=-0.5, control=bound)

#obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, weighting_scheme='cotangent')

pyredner.save_obj(obj, "../nnew.obj")

# This program reconstruct the face from multi images and try to smooth
# output form changed