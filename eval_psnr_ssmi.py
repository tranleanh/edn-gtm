import os 
import cv2
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import glob


#s = ssim(imageA, imageB)
# s = measure.compare_ssim(imageA, imageB)

# def cal_ssim(y_true , y_pred):
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     std_true = np.sqrt(var_true)
#     std_pred = np.sqrt(var_pred)
#     c1 = np.square(0.01*7)
#     c2 = np.square(0.03*7)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom


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
	pred = cv2.resize(pred, (img_size,img_size))

	gt = cv2.imread(path)
	gt = cv2.resize(gt, (img_size,img_size))

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