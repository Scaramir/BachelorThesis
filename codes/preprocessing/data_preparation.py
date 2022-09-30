"""
Maximilian Otto, 2022, maxotto45@gmail.com
Prepare the imaging data for the neural network.
"""

# ----------------------------------------------------------------------------------------------- #
# Global parameters:
# Set the working directory, where all the data is stored:
wd = "S:/mdc_work/mdc_huntington/images/Cortical Organoids"

# The pre-processing will be applied to the following folders/conditions:
folders_list = ["combined"]

# Merge three fluorescent channels of the same plate area into one image.
# Channel name prefix:
#  Example: "c01" + "c02" + "c03" -> "combined"
#  -> ch_prefix = "c0"
ch_prefix = "C00"

# Channel name suffices:
ch_1_suf = "1"
ch_2_suf = "2"
ch_3_suf = "3"

# File format:
input_file_format = ".tiff"

# Combine the images into one RGB-image?
merge_to_rgb = True
output_file_format = ".bmp"

# Min. bit-depth we want to check for:
# Has the microscope used a >12bit-color camera and an according sensitivity? 
# If so, at least some pixels of the images should contain intensities with a higher value than the required min_bit_depth. 
# FOr 12bit-color images, the max. value of a pixel is 4095, so min_bit_depth = 8 (-> pixel brighter than 255 should exist) 
min_bit_depth = 8
# ----------------------------------------------------------------------------------------------- #

import exifread
import glob, os
import cv2
from tqdm import tqdm

# Input: amount of bits, e.g. the amount of bits used to store a greyscale image.
# Return: Max. value of a certain amount of bits. 
# Note: used to get the max. value of a certain "bit-depth"of an image.
def max_bits(min_bit_depth):
    return (1 << min_bit_depth) - 1

def is_it_really_16_bit(file, max_value_of_min_bit_depth):
    """
    Does the tif-file contain a pixel value higher than the max. value of the min. bit-depth we want?
    input: current file name: string, max_value_of_min_bit_depth: int
    return: Boolean
    """
    f = open(file, 'rb')                        # Open image file for reading (binary mode)
    tags = exifread.process_file(f)             # Get Exif tags
    # Check max. brightness of the image (Larger than max(8bit)):
    if "Image SMaxSampleValue" in tags and str(tags["Image SMaxSampleValue"].values) > max_value_of_min_bit_depth:
        pic_brighter_than_min_bit_depth = True
    return pic_brighter_than_min_bit_depth

def check_bit_depth(pic_folder_path):
    """
    Check each image of a folder for its brightness. 
    Print the number of images that are darker than the min_bit_depth we want.
    Imaging data in 16bit-tif-format should have one pixel value higher than 12bit-color values. 
    otherwise the image can be considered as too dark and needs to be treated in a different way.
    (This can have different reasons: low sensitivity of the microscope, made an 8bit-image but saved it as 16bit, etc.)
    """
    pics_brighter_than_16_bit = 0
    pics_total = 0
    pics_too_dark = []
    max_value_of_min_bit_depth = max_bits(min_bit_depth)
    os.chdir(pic_folder_path)
    print("Searching for pictures with a brightness indicating that they are truly 16 bits and not too dark:")
    print(os.getcwd())
    for file in tqdm(glob.glob("*"+input_file_format), desc = "Checking for bit depth"):
        pics_total += 1
        if is_it_really_16_bit(file, max_value_of_min_bit_depth) == True:
            pics_brighter_than_16_bit += 1
        else: 
            pics_too_dark.append(os.path.basename(file))
    print("{0} {1} {2} {3}".format(pics_brighter_than_16_bit, "of", pics_total, "are stored in the correct format."))
    if len(pics_too_dark) > 0:
        print("The following files are too dark or they could have been saved in another data format:")
        if len(pics_too_dark) == len([name for name in os.listdir() if os.path.isfile(name)]):
            print("All images are too dark")
        else: 
            for pic in pics_too_dark:
                print(pic)
    return pics_brighter_than_16_bit
#check_bit_depth(pic_folder_path)                
# none ok so far. --> they are stored and interpreted as rgb8 images even though they are just greyscale 
# (why not save them as greyscales with a higher bit-depth?)

def image_merger_to_rgb(pic_folder_path): 
    """
    Merge the three channels of the same image into one image.
    input: "picture folder path": string
    """
    pics_total = 0
    base_channel = ch_prefix + ch_1_suf
    for file in tqdm(glob.glob(pic_folder_path+"/*"+base_channel+"*"+input_file_format), desc = "Merging three channels into one rgb8 file"):
        if os.path.isfile(file.replace(base_channel, "combined")):
            continue
        ch1 = cv2.imread(file, -1)
        ch2 = cv2.imread(file.replace(base_channel, ch_prefix + ch_2_suf), -1)
        ch3 = cv2.imread(file.replace(base_channel, ch_prefix + ch_3_suf), -1)

        # Combine the channels into one image
        combined_img = cv2.merge((ch1[:,:,0], ch2[:,:,0], ch3[:,:,0]))
        file_replaced = file.replace(base_channel, "combined_")
        cv2.imwrite(file_replaced.replace(input_file_format, output_file_format), combined_img)
        pics_total += 1
    print("Created {0} rgb-images".format(pics_total), end = "\r")


# -------------------------------------- MAIN ------------------------------------------- #
for sub_folder in folders_list:
    pic_folder_path = os.path.join(wd, sub_folder)
    os.chdir(pic_folder_path)
    check_bit_depth(pic_folder_path)
    if merge_to_rgb:
        image_merger_to_rgb(pic_folder_path)
