import UNET3D
import preprocessing
import utils

import os
from keras import backend as K
import tensorflow as tf
import numpy as np
import SimpleITK as sitk

# expectations for input data structure: multi plane images must be provided (for extraction of ROI)
# plane must be defined in img name with keywords 'tra', 'sag' or 'cor'
# all images must be in same directory
def segment(inputDirectory, outputDirectory, multistream = True):


    # load images
    files = os.listdir(inputDirectory)
    for file in files:
        if 'tra' in file:
            img_tra = sitk.ReadImage(inputDirectory  + '/' + file)
        if 'cor' in file:
            img_cor = sitk.ReadImage(inputDirectory  + '/' + file)
        if 'sag' in file:
            img_sag = sitk.ReadImage(inputDirectory + '/' + file)

    # make directory of the output
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # save original and upsampled version of transversal image
    img_tra_original = img_tra
    img_tra_HR = utils.resampleImage(img_tra, [0.5, 0.5, 0.5], sitk.sitkLinear,0)
    img_tra_HR = utils.sizeCorrectionImage(img_tra_HR, 6, 168)
    sitk.WriteImage(img_tra_HR, outputDirectory+'/tra_HR.nrrd')

    # save upsampled version of coronal image
    cor_toTraHR = utils.resampleToReference(img_cor, img_tra_HR, sitk.sitkLinear, 0)
    sitk.WriteImage(cor_toTraHR, outputDirectory + '/cor_HR.nrrd')

    # save upsampled version of sagittal image
    sag_toTraHR = utils.resampleToReference(img_sag, img_tra_HR, sitk.sitkLinear, 0)
    sitk.WriteImage(sag_toTraHR, outputDirectory + '/sag_HR.nrrd')

    # preprocess and save to numpy array
    print('... preprocess images and save to array...')
    img_tra, img_cor, img_sag, startROI, sizeROI  = preprocessing.getCroppedIsotropicImgs(outputDirectory, img_tra, img_cor, img_sag)
    input_array = preprocessing.createInputArray(multistream, img_tra, img_cor, img_sag)
    #TODO: delete directory with cropped images?

    # get net and model
    if multistream:
        weightFile = 'weights/weights_multiStream.h5'
        model = UNET3D.get_net_multiPlane()
    else:
        weightFile= 'weights/weights_singleStream.h5'
        model = UNET3D.get_net_singlePlane()

    #TODO: compile network here
    print('... load weights into model...')
    model.load_weights(weightFile)

    # predict image with CNN (either multistream or single stream)
    print('... predict image ...')
    if multistream:
        img_labels = model.predict([input_array[0:1, :, :, :, 0:1], input_array[0:1, :, :, :, 1:2],
                                    input_array[0:1, :, :, :, 2:3]], verbose=1)
    else:
        img_labels = model.predict([input_array[0:1, :, :, :, 0:1]], verbose=1)


    print('... save predicted image...')

    # transform prediction back to original and upsampled transversal input space
    # upsampled transversal space
    output_predicted_original = sitk.Image(img_tra_HR.GetSize(), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(output_predicted_original)
    arr[startROI[2]:startROI[2]+sizeROI[2], startROI[1]:startROI[1]+sizeROI[1],startROI[0]:startROI[0]+sizeROI[0]] = img_labels[0,:,:,:,0]
    output_predicted = sitk.GetImageFromArray(arr)
	output_predicted = utils.binaryThresholdImage(output_predicted, 0.5)
    output_predicted = utils.getLargestConnectedComponents(output_predicted)
    output_predicted.SetOrigin(img_tra_HR.GetOrigin())
    output_predicted.SetDirection(img_tra_HR.GetDirection())
    output_predicted.SetSpacing(img_tra_HR.GetSpacing())
    sitk.WriteImage(output_predicted, outputDirectory + '/predicted_HR.nrrd')

    # original transversal space (high slice thickness), transform perdiction with shape-based interpolation (via distance transformation)
    segm_dis = sitk.SignedMaurerDistanceMap(output_predicted, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
    segm_dis = utils.resampleToReference(segm_dis, img_tra_original, sitk.sitkLinear, -1)
    #smoothed = sitk.DiscreteGaussian(gt_traHR, variance=1.0)
    thresholded = utils.binaryThresholdImage(segm_dis, 0)
    sitk.WriteImage(thresholded, outputDirectory + '/predicted_transversal_space.nrrd')



    K.clear_session()
    tf.reset_default_graph()

if __name__ == '__main__':

    segment(inputDirectory = 'data/ProstateX-0029/', outputDirectory='output_single', multistream = False)
    segment(inputDirectory='data/ProstateX-0029/', outputDirectory='output_multi', multistream=True)


