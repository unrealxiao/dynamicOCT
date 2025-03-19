from natsort import natsorted
import os
from pydicom import dcmread
import re
import numpy as np
import pathlib 
import h5py
import multiprocessing
import cv2
#this script is used to convert processed dicom file into H5 file and put it in the globus shared directory

# def parallel_convert(variables):
#     s_name, dcm_list, arr_shape, source_path, target_path, data_name = variables
#     volume = np.zeros((len(dcm_list), arr_shape[0], arr_shape[1]))
#     non_sub_list = [f for f in os.listdir(os.path.join(source_path, s_name, 'pic')) if re.match(r'.*\.dcm', f)]
#     dcm_sub_list = natsorted(non_sub_list)
#     for i in range(len(dcm_sub_list)):
#         d_name = dcm_sub_list[i]
#         image = dcmread(os.path.join(source_path, s_name, 'pic', d_name))
#         array = image.pixel_array
#         volume[i, :, :] = array
    
#     target_folder = os.path.join(target_path, data_name, s_name)#the folder where picke file will be saved
#     pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)#create the folder if it was not created
#     target_addressname = os.path.join(target_folder, f'{s_name}.h5')
#     with h5py.File(target_addressname, 'w') as h5f:
#         h5f.create_dataset(
#             "volume",
#             data=volume,
#             compression="gzip",
#             compression_opts=5,
#             chunks=True
#         )
#     print("completed processing ", s_name)

# if __name__ == "__main__":

#     source_path = "I:/Cells_without_H2o2"
#     data_name = os.path.basename(source_path)#name of the main image set
#     target_path = "D:/globus slate shared data Tankam Lab"
#     scan_list = [f for f in natsorted(os.listdir(source_path)) if re.match('scan[0-9]+', f)]
#     scan_name = scan_list[0]
#     nonorganize_list = [f for f in os.listdir(os.path.join(source_path, scan_name, 'pic')) if re.match(r'.*\.dcm', f)]
#     # nonorganize_list = [f for f in os.listdir(os.path.join(source_path, scan_name)) if re.match(r'^image_[0-9]+\.asc$', f)]
#     dcm_list = natsorted(nonorganize_list)
#     dcm_name = dcm_list[0]
#     ds = dcmread(os.path.join(source_path, scan_name, 'pic', dcm_name))
#     arr = ds.pixel_array
#     arr_shape = arr.shape
#     print("start to convert dicm to pickle")

#     with multiprocessing.Pool(processes=4) as pool:k
#         tasks = [(name, dcm_list, arr_shape, source_path, target_path, data_name) 
#                  for name in scan_list
#                 ]
#         pool.map(parallel_convert, tasks)


#use asc file to save instead


def parallel_convert(variables):
    s_name, asc_list, arr_shape, source_path, target_path, data_name = variables
    volume = np.zeros((len(asc_list), arr_shape[0], arr_shape[1]))
    non_sub_list = [f for f in os.listdir(os.path.join(source_path, s_name)) if re.match(r'^image_[0-9]+\.asc$', f)]
    asc_sub_list = natsorted(non_sub_list)
    for i in range(len(asc_sub_list)):
        d_name = asc_sub_list[i]
        image = np.loadtxt(os.path.join(source_path, s_name, d_name))
        new_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        calibrate = new_img.astype(np.uint16)
        volume[i, :, :] = calibrate
    
    target_folder = os.path.join(target_path, data_name, s_name)#the folder where picke file will be saved
    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)#create the folder if it was not created
    target_addressname = os.path.join(target_folder, f'{s_name}.h5')
    with h5py.File(target_addressname, 'w') as h5f:
        h5f.create_dataset(
            "volume",
            data=volume,
            compression="gzip",
            compression_opts=5,
            chunks=True
        )
    print("completed processing ", s_name)

if __name__ == "__main__":

    source_path = "I:/IR_card_glass_side_3_13_2025"
    data_name = os.path.basename(source_path)#name of the main image set
    target_path = "D:/globus slate shared data Tankam Lab"
    scan_list = [f for f in natsorted(os.listdir(source_path)) if re.match('scan[0-9]+', f)]
    scan_name = scan_list[0]
    nonorganize_list = [f for f in os.listdir(os.path.join(source_path, scan_name)) if re.match(r'^image_[0-9]+\.asc$', f)]
    asc_name = nonorganize_list[0]
    image = np.loadtxt(os.path.join(source_path, scan_name, asc_name))
    new_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    calibrate = new_img.astype(np.uint16)
    arr_shape = calibrate.shape
    print("start to convert asc to h5")

    with multiprocessing.Pool(processes=4) as pool:
        tasks = [(name, nonorganize_list, arr_shape, source_path, target_path, data_name) 
                 for name in scan_list
                ]
        pool.map(parallel_convert, tasks)
