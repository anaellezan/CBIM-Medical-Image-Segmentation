import os
from tqdm import tqdm
from utils import ResampleXYZAxis, ResampleLabelToRef
import SimpleITK as sitk
import yaml


def ResampleCMRImage(imImage, imLabel, tgt_path, file, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()

    if not os.path.exists('%s'%(tgt_path)):
        os.mkdir('%s'%(tgt_path))
    
    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)

    print(os.path.join(tgt_path, file.replace('_0000.mha','_0.mha')))
    print(os.path.join(tgt_path, file.replace('_0000.mha','_0_gt.mha')))

    sitk.WriteImage(re_img_xyz, os.path.join(tgt_path, file.replace('_0000.mha','_0.mha')))
    sitk.WriteImage(re_lab_xyz, os.path.join(tgt_path, file.replace('_0000.mha','_0_gt.mha')))



def main():

    src_path = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset002_LA_CT00/'
    tgt_path = '/media/sharedata/atriumCT/atrium_medFormer/LA_CT00_pretraining_dataset/'

    imagestr = os.path.join(src_path, "imagesTr")
    labelstr = os.path.join(src_path, "labelsTr")

    images_files = [f for f in os.listdir(imagestr) if f[-4:]=='.mha']

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump([file.replace('_0000.mha','') for file in images_files], f)
    
    for file in tqdm(images_files):
        img = sitk.ReadImage(os.path.join(imagestr, file))
        lab = sitk.ReadImage(os.path.join(labelstr, file.replace('_0000.mha','.mha')))
        ResampleCMRImage(img, lab, tgt_path, file, (0.5, 0.388672, 0.388672)) # median spacing defined in nnUnet Dataset002


if __name__ == '__main__':
    main()