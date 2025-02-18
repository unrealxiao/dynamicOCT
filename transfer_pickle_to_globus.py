from natsort import natsorted
import os
import pickle
from pydicom import dcmread
import re
import numpy as np
import pathlib 

#this script is used to convert processed dicom file into pickle file and put it in the globus shared directory
#
source_path = "I:/cells with H2O2 after 3hours of application"
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
print("start to convert dicm to pickle")
for s_name in scan_list:
    print("processing ", s_name)
    volume = np.zeros((len(dcm_list), arr_shape[0], arr_shape[1]))
    non_sub_list = [f for f in os.listdir(os.path.join(source_path, s_name, 'pic')) if re.match(r'.*\.dcm', f)]
    dcm_sub_list = natsorted(non_sub_list)
    for i in range(len(dcm_sub_list)):
        d_name = dcm_sub_list[i]
        image = dcmread(os.path.join(source_path, s_name, 'pic', d_name))
        array = image.pixel_array
        volume[i, :, :] = array
    
    target_folder = os.path.join(target_path, data_name, s_name)#the folder where picke file will be saved
    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)#create the folder if it was not created
    with open(os.path.join(target_folder, f'{s_name}.pickle'), 'wb') as handle:
        pickle.dump(volume, handle, protocol=pickle.HIGHEST_PROTOCOL)