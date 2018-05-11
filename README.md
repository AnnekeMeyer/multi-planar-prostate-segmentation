# multi-planar-prostate-segmentation
sourcecode of multi-planar prostate segmentation (ISBI publication in 2018)

Multi-Planar Prostate Segmentation with multi-stream CNN
This algorithm enables the high resolution segmentation of the prostate in MRI images from orthogonal T2-weighted scans (transversal, sagittal and coronal). With this multi-planar approach, improvements on segmentation predictions could be achieved comparing the results of single volume (transversal) based segmentation.

Algorithm description
Details on the algorithm can be found in the attached paper. The inputs to the algorithm are the three orthogonal volumes, which should be registered to each other (example data is provided in the ‘data’ directory). Data used in this research was obtained from the ProstateX Challenge [1-3]. The outputs of the algorithm are the high resolution segmentation, its downsampled version (downsampled to transversal input space) as well as the linearly upsampled input volumes.
The algorithm is started in the main function of ‘segmentation.py’. The function arguments are the input directory of the volumes, the output directory and the option if multi-planar or single transversal segmentation is applied. Please note that all three planes must be provided for the extraction of the ROI (input to the convolutional neural network). The ROI can also be extracted in other ways, e.g. interactive or with a detection algorithm. But this needs to be implemented in future. Furthermore, the input size of the network can not be arbitrary. At the moment, only volumes of size 168x168x168 voxels are taken as input. 

Further remarks
The model was trained on publicly available data from The Cancer Imaging Archive (TCIA) sponsored by the SPIE, NCI/NIH, AAPM, and Radboud University [1]. Whether it works on other data has not been tested yet. A training refinement of the model might be necessary. 
The computation time of the algorithm is about 70 s for single volume prediction and 120 s for multi-planar segmentation on CPU (i7-6820HQ CPU @ 2.70 GHz). The tested speed with NVIDIA TitanX GPU was 1-2s for single and multi-planar prediction.
More segmentations of the challenge dataset can be found on http://isgwww.cs.uni-magdeburg.de/cas/isbi2018/. Please refer to our work (ISBI paper) when using the segmentations.


[1] G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer, and H. Huisman. "ProstateX Challenge data", The Cancer Imaging Archive (2017). https://doi.org/10.7937/K9TCIA.2017.MURS5CL
[2] G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer and H. Huisman. "Computer-aided detection of prostate cancer in MRI", IEEE Transactions on Medical Imaging 2014;33:1083-1092.
[3] Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.
