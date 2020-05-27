import torch
import pyredner
import numpy as np
import os
import sys
import argparse

dir_light_directions = torch.tensor([[-1.0, -1.0, -1.0],
                                     [1.0, -0.0, -1.0],
                                     [0.0, 0.0, -1.0]])
dir_light_intensities = torch.ones(3, dtype=torch.float32).expand(3, 3)

dir_lights = [pyredner.DirectionalLight(dir_light_directions[i], dir_light_intensities[i]) for i in range(len(dir_light_directions))]

data = np.array(dir_lights)

np.save("data.npy", data)
