import os
import numpy as np
import SimpleITK as sitk
import utils
import math


# .nrrd files have been used as data format, but probably other formats might be possible.
# input order of imgs: tra, cor, sag
def createInputArray( multiplane_array, *imgs):

    print('... save images to numpy array ...')
    if multiplane_array:
        outArray = np.zeros([1, 168, 168, 168, 3])
    else:
        outArray = np.zeros([1, 168, 168, 168, 1])

    # transversal image
    outArray[0, :, :, :, 0] = sitk.GetArrayFromImage(imgs[0])
    if multiplane_array:
        # coronal image
        outArray[0, :, :, :, 1] = sitk.GetArrayFromImage(imgs[1])
        # sagittal image
        outArray[0, :, :, :, 2] = sitk.GetArrayFromImage(imgs[2])

    return outArray


# crop images to overlapping ROI and resample to isotropic transversal space
# input order of imgs: tra, cor, sag
def getCroppedIsotropicImgs(outputDirectory, *imgs):

    img_tra = imgs[0]
    img_cor = imgs[1]
    img_sag = imgs[2]

    # normalize intensities
    print('... normalize intensities ...')
    img_tra, img_cor, img_sag = utils.normalizeIntensitiesPercentile(img_tra, img_cor, img_sag)

    # get intersecting region (bounding box)
    print('... get intersecting region (ROI) ...')

    # upsample transversal image to isotropic voxel size (isotropic transversal image coordinate system is used as reference coordinate system)
    tra_HR = utils.resampleImage(img_tra, [0.5, 0.5, 0.5], sitk.sitkLinear,0)
    tra_HR = utils.sizeCorrectionImage(tra_HR, factor=6, imgSize=168)

    # resample coronal and sagittal to tra_HR space
    # resample coronal to tra_HR and obtain mask (voxels that are defined in coronal image )
    cor_toTraHR = utils.resampleToReference(img_cor, tra_HR, sitk.sitkLinear,-1)
    cor_mask = utils.binaryThresholdImage(cor_toTraHR, 0)

    tra_HR_Float = utils.castImage(tra_HR, sitk.sitkFloat32)
    cor_mask_Float = utils.castImage(cor_mask, sitk.sitkFloat32)
    # mask transversal volume (set voxels, that are defined only in transversal image but not in coronal image, to 0)
    coronal_masked_traHR = sitk.Multiply(tra_HR_Float, cor_mask_Float)

    # resample sagittal to tra_HR and obtain mask (voxels that are defined in sagittal image )
    sag_toTraHR = utils.resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)
    sag_mask = utils.binaryThresholdImage(sag_toTraHR, 0)
    # mask sagittal volume
    sag_mask_Float = utils.castImage(sag_mask, sitk.sitkFloat32)

    # masked image contains voxels, that are defined in tra, cor and sag images
    maskedImg = sitk.Multiply(sag_mask_Float, coronal_masked_traHR)
    boundingBox = utils.getBoundingBox(maskedImg)

    # correct the size and start position of the bounding box according to new size
    start, size = sizeCorrectionBoundingBox(boundingBox, newSize=168, factor=6)
    start[2] = 0
    size[2] = tra_HR.GetSize()[2]

    # resample cor and sag to isotropic transversal image space
    cor_traHR = utils.resampleToReference(img_cor, tra_HR, sitk.sitkLinear, -1)
    sag_traHR = utils.resampleToReference(img_sag, tra_HR, sitk.sitkLinear,-1)

    ## extract bounding box for all planes
    region_tra = sitk.RegionOfInterest(tra_HR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_tra)
    region_tra = utils.thresholdImage(region_tra, 0, maxVal, 0)

    region_cor = sitk.RegionOfInterest(cor_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_cor)
    region_cor = utils.thresholdImage(region_cor, 0, maxVal, 0)

    region_sag = sitk.RegionOfInterest(sag_traHR, [size[0], size[1], size[2]],
                                       [start[0], start[1], start[2]])
    maxVal = utils.getMaximumValue(region_sag)
    region_sag = utils.thresholdImage(region_sag, 0, maxVal, 0)

    # save cropped images to output directory
    # if not os.path.exists(outputDirectory+ '/ROI/'):
    #     os.makedirs(outputDirectory+ '/ROI/')
    #
    # sitk.WriteImage(region_tra, outputDirectory + '/ROI/'+ 'croppedIsotropic_tra.nrrd')
    # sitk.WriteImage(region_cor, outputDirectory + '/ROI/'+ 'croppedIsotropic_cor.nrrd')
    # sitk.WriteImage(region_sag, outputDirectory + '/ROI/'+ 'croppedIsotropic_sag.nrrd')
    return region_tra, region_cor, region_sag, start, size


# adapt the start index of the ROI to the manual bounding box size
# (assumes that all ROIs are smaller than newSize pixels in length and width)
def sizeCorrectionBoundingBox(boundingBox, newSize, factor):
    # correct the start index according to the new size of the bounding box
    start = boundingBox[0:3]
    start = list(start)
    size = boundingBox[3:6]
    size = list(size)
    start[0] = start[0] - math.floor((newSize - size[0]) / 2)
    start[1] = start[1] - math.floor((newSize - size[1]) / 2)

    # check if BB start can be divided by the factor (essential if ROI needs to be extracted from non-isotropic image)
    if (start[0]) % factor != 0:
        cX = (start[0] % factor)
        newStart = start[0] - cX
        start[0] = int(newStart)

    # y-direction
    if (start[1]) % factor != 0:
        cY = (start[1] % factor)
        start[1] = int(start[1] - cY)

    size[0] = newSize
    size[1] = newSize

    return start, size