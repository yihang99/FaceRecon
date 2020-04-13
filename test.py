import torch
import pyredner
import math

import urllib
import zipfile


objs = pyredner.load_obj('smoothed.obj', return_objects=True)
obj = objs[0]
vertices = obj.vertices
indices = obj.indices




'''
obj.laplacian_smooth(1)
pyredner.save_obj(obj, 'smoothed2.obj')
'''

'''this is a note'''