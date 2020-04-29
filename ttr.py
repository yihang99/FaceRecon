import os
for normal_scheme in [' max', ' cotangent', ' else']:
    for smooth_scheme in [' reciprocal', ' cotangent', ' uniform']:
        for lmd in [0.1, 0.5]:
            os.system('CUDA_VISIBLE_DEVICES=0 python trecon.py' + normal_scheme + smooth_scheme + ' ' + str(lmd))