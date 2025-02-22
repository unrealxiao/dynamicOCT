from natsort import natsorted
import os
import pickle
from pydicom import dcmread
import re
import numpy as np
import pathlib 
import h5py
# import multiprocessing
#this script is used to convert processed dicom file into pickle file and put it in the globus shared directory
#
# def parallel_convert(variables):
#     s_name, dcm_list, arr_shape, source_path, target_path, data_name = variables
#     print("processing ", s_name)
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
#     with open(os.path.join(target_folder, f'{s_name}.pickle'), 'wb') as handle:
#         pickle.dump(volume, handle, protocol=pickle.HIGHEST_PROTOCOL)

# if __name__ == "__main__":

#     source_path = "I:/cells_with_h2o2_after 1hour"
#     data_name = os.path.basename(source_path)#name of the main image set
#     target_path = "D:/globus slate shared data Tankam Lab"
#     scan_list = [f for f in natsorted(os.listdir(source_path)) if re.match('scan[0-9]+', f)]
#     scan_name = scan_list[0]
#     nonorganize_list = [f for f in os.listdir(os.path.join(source_path, scan_name, 'pic')) if re.match(r'.*\.dcm', f)]
#     dcm_list = natsorted(nonorganize_list)
#     dcm_name = dcm_list[0]
#     ds = dcmread(os.path.join(source_path, scan_name, 'pic', dcm_name))
#     arr = ds.pixel_array
#     arr_shape = arr.shape
#     print("start to convert dicm to pickle")

#     with multiprocessing.Pool(processes=4) as pool:
#         tasks = [(name, dcm_list, arr_shape, source_path, target_path, data_name) 
#                  for name in scan_list
#                 ]
#         pool.map(parallel_convert, tasks)

source_path = "H:/Hadiya_5_16_2024"
data_name = os.path.basename(source_path)#name of the main image set
target_path = "D:/globus slate shared data Tankam Lab"
scan_list = [f for f in natsorted(os.listdir(source_path)) if re.match('scan[0-9]+', f)]
scan_name = scan_list[0]
nonorganize_list = [f for f in os.listdir(os.path.join(source_path, scan_name, 'pic')) if re.match(r'.*\.dcm', f)]
dcm_list = natsorted(nonorganize_list)
dcm_name = dcm_list[0]
ds = dcmread(os.path.join(source_path, scan_name, 'pic', dcm_name))
arr = ds.pixel_array
arr_shape = arr.shape
target_folder = os.path.join(target_path, data_name)#the folder where picke file will be saved
pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)#create the folder if it was not created
print("start to convert dicm to pickle")
target_addressname = os.path.join(target_folder, "fulldata.h5")#address and name of the h5 compressed file
for s_name in scan_list:
    volume = np.zeros((len(dcm_list), arr_shape[0], arr_shape[1]))
    non_sub_list = [f for f in os.listdir(os.path.join(source_path, s_name, 'pic')) if re.match(r'.*\.dcm', f)]
    dcm_sub_list = natsorted(non_sub_list)
    for i in range(len(dcm_sub_list)):
        d_name = dcm_sub_list[i]
        image = dcmread(os.path.join(source_path, s_name, 'pic', d_name))
        array = image.pixel_array
        volume[i, :, :] = array
    
    # with open(os.path.join(target_folder, f'{s_name}.pickle'), 'wb') as handle:
    #     pickle.dump(volume, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with h5py.File(target_addressname, 'a') as h5f:
        h5f.create_dataset(
            s_name,
            data=volume,
            compression="gzip",
            compression_opts=5,
            chunks=True
        )
    print("completed processing ", s_name)

with h5py.File(target_addressname, 'r') as h5f:
    for name in h5f.keys():
        dset = h5f[name]
        print(f"Dataset: {name}, Compression: {dset.compression}, Shape: {dset.shape}")