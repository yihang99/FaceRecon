<<<<<<< HEAD
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

=======
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

>>>>>>> 6c3d57d8a450b2afaf03b7a9487e3766d28f4e75
'''this is a note'''