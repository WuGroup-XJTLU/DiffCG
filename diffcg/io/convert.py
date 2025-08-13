# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from ase.io import read,write

def ase2xyz(ase_traj,xyz_traj):
    traj=read(ase_traj,index=':')
    for i,frame in enumerate(traj):
        write(f'{xyz_traj}{i}.xyz',frame)