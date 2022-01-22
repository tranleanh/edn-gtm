# EDN-GTM: A Novel Encoder-Decoder Network with Guided Transmission Map for Single Image Dehazing

Official Implementation of "EDN-GTM: A Novel Encoder-Decoder Network with Guided Transmission Map for Single Image Dehazing"

Paper: (to be updated)

Medium article: (will be updated)

## I. Requiremtents
- CUDA 10.0
- CUDNN 7.6
- OpenCV
- Tensorflow 1.14.0
- Keras 2.1.3
<!-- - [more](https://pjreddie.com/darknet/yolo/) -->

You can simply run:
```bashrc
$ pip install -r requirements.txt
```

## II. Test using Pre-trained Weights

#### 1. Download pre-trained weights
- Download weight files from this link

- Locate weight files in folder: weights/

#### 2. Edit Image Folder in Script
- Image Folder

#### 3. Run Dehazing

- Run: to be updated
```bashrc
$ python test_on_images.py
```

## III. Train Network

#### 1. Prepare Dataset
- Each image in a clean-hazy image pair must have the same name

- Make Folder 'A' and Folder 'B' containing hazy and clean images, respectively

#### 2. Edit Image Folder Path in Training Script
- Image Folder

#### 3. Run Training

- Run: to be updated
```bashrc
$ python train.py
```


## IV. Results

#### 1. Quantitative Results
#### (a) I-HAZE Dataset

|     |  DCP (TPAMI’10)  | CAP (TIP’15) | MSCNN (ECCV’16) | NLID (CVPR’1) | AOD-Net (ICCV’17) | PPD-Net (CVPRW’18) | EDN-GTM           | 
| :---:     |   :---:          |   :---:      |    :----:       |    :---:      |          :---:    |    :----:          |    :---:           |
|  PSNR     |   14.43          |    12.24     |    15.22        |    14.12      |         13.98     |   22.53 (2nd)      |   22.90 (1st)      |
|  SSIM     |   0.7516         |    0.6065    |    0.7545       |    0.6537     |         0.7323    |   0.8705 (1st)      |  0.8270 (2nd)      |

#### (b) O-HAZE Dataset

|     |  DCP (TPAMI’10)  | CAP (TIP’15) | MSCNN (ECCV’16) | NLID (CVPR’1) | AOD-Net (ICCV’17) | PPD-Net (CVPRW’18) |  EDN-GTM      | 
| :---:     |   :---:          |   :---:      |    :----:       |    :---:      |          :---:    |    :----:          |     :---:           |
|  PSNR     |   16.78          |    16.08     |    17.56        |    15.98      |         15.03     |    24.24 (1st)    |     23.46 (2nd)     |
|  SSIM     |   0.6532         |    0.5965    |    0.6495       |    0.5849     |         0.5385    |   0.7205 (2nd)     |      0.8198 (1st)   |

#### (c) Dense-HAZE Dataset

|     |  DCP (TPAMI’10)  | DehazeNet (TIP’16) | AOD-Net (ICCV’17) | MSBDN (CVPR’20) | KDDN (CVPR’20)    | AECR-Net (CVPR’21)  | EDN-GTM       | 
| :---:     |   :---:          |   :---:            |    :----:         |    :---:        |          :---:    |    :----:           |    :---:           |
|  PSNR     |   10.06          |    13.84           |    13.14          |    15.37        |         14.28     |    15.80 (1st)      |  15.43 (2nd)       |
|  SSIM     |   0.3856         |    0.4252          |    0.4144         |    0.4858       |         0.4074    |  0.4660 (2nd)       |     0.5200 (1st)   |

#### (d) NH-HAZE Dataset

|     |  DCP (TPAMI’10)  | DehazeNet (TIP’16) | AOD-Net (ICCV’17) | MSBDN (CVPR’20) | KDDN (CVPR’20)    | AECR-Net (CVPR’21)  | EDN-GTM   | 
| :---:     |   :---:          |   :---:            |    :----:         |    :---:        |          :---:    |    :----:           |    :---:        |
|  PSNR     |   10.57          |    16.62           |    15.40          |    19.23        |      17.39        |  19.88 (2nd)        |   20.24 (1st)   |
|  SSIM     |   0.5196         |    0.5238          |    0.5693         |    0.7056       |      0.5897       |  0.7173 (2nd)       |   0.7178 (1st)  |

#### 2. Qualitative Results

#### (a) I-HAZE Dataset

<img src="docs/dehazed-ihaze.jpg" width="800">

#### (b) O-HAZE Dataset

<img src="docs/dehazed-ohaze.jpg" width="800">

#### (c) Dense-HAZE Dataset

<img src="docs/dehazed-densehaze.jpg" width="800">

#### (d) NH-HAZE Dataset

<img src="docs/dehazed-nhhaze.jpg" width="800">


Have fun!

LA Tran

12.2021
