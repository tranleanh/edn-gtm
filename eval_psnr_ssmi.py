import os 
import cv2
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import glob


def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


total_psnr = 0
total_ssmi = 0

# offset = 51
img_size = 512

testset = glob.glob("./ohaze_valset_gt/*.jpg")

output_folder = "predict_ohaze_unet_spp_swish_deeper_gan_4c"
txtfile = open(f"./{output_folder}/test_log.txt", "w+")

for path in testset:

	fname = get_file_name(path)

	pred = cv2.imread(f"./{output_folder}/{fname}.jpg")

	gt = cv2.imread(path)

	psnr = cv2.PSNR(pred, gt)
	ssmi = cal_ssim(gt, pred, data_range=pred.max() - pred.min(), multichannel=True)

	print(fname, psnr, ssmi)
	print(fname, psnr, ssmi, file=txtfile)

	total_psnr += psnr
	total_ssmi += ssmi

average_psnr = total_psnr/len(testset)
average_ssmi = total_ssmi/len(testset)

print("PSNR:", average_psnr)
print("SSMI:", average_ssmi)

print("PSNR:", average_psnr, file=txtfile)
print("SSMI:", average_ssmi, file=txtfile)
