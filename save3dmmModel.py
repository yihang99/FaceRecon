<<<<<<< HEAD
import torch
import pyredner
import numpy as np
import h5py
import matplotlib.pyplot as plt
with h5py.File(r'model2017-1_bfm_nomouth.h5', 'r') as hf:
    shape_mean = torch.tensor(hf['shape/model/mean'], device=pyredner.get_device())
    shape_basis = torch.tensor(hf['shape/model/pcaBasis'], device=pyredner.get_device())
    triangle_list = torch.tensor(hf['shape/representer/cells'], device=pyredner.get_device())
    color_mean = torch.tensor(hf['color/model/mean'], device=pyredner.get_device())
    color_basis = torch.tensor(hf['color/model/pcaBasis'], device=pyredner.get_device())

_3dmm = np.array((shape_mean, shape_basis, triangle_list, color_mean, color_basis))
=======
import torch
import pyredner
import numpy as np
import h5py
import matplotlib.pyplot as plt
with h5py.File(r'model2017-1_bfm_nomouth.h5', 'r') as hf:
    shape_mean = torch.tensor(hf['shape/model/mean'], device=pyredner.get_device())
    shape_basis = torch.tensor(hf['shape/model/pcaBasis'], device=pyredner.get_device())
    triangle_list = torch.tensor(hf['shape/representer/cells'], device=pyredner.get_device())
    color_mean = torch.tensor(hf['color/model/mean'], device=pyredner.get_device())
    color_basis = torch.tensor(hf['color/model/pcaBasis'], device=pyredner.get_device())

_3dmm = np.array((shape_mean, shape_basis, triangle_list, color_mean, color_basis))
>>>>>>> 6c3d57d8a450b2afaf03b7a9487e3766d28f4e75
np.save("3dmm.npy", _3dmm)