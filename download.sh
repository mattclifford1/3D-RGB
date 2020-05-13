#!/bin/sh

mkdir 3d-inpainting/checkpoints
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
mv color-model.pth 3d-inpainting/checkpoints/.
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
mv depth-model.pth 3d-inpainting/checkpoints/.
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
mv edge-model.pth 3d-inpainting/checkpoints/.
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt
mv model.pt 3d-inpainting/MiDaS/.
