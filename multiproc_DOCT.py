import numpy as np #numpy for math operation
import os
import utils
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
# from time import time
from tqdm import tqdm
import cv2
from natsort import natsorted
from pydicom import dcmread
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
from multiprocessing import Pool, shared_memory
import h5py

folder_name = 'IR_noshift'
data_name = 'standard_inter/'
data_path = '../../../../../../../../N/project/OCT_preproc/IR_card_glass_side_3_13_2025_without_shift_correction/intervolume_registered/' + data_name
# data_path = '/Users/akapatil/Documents/OCT/timelapse/inter_volume_registered/' + data_name

average_LivCurve = True
fitting_method = "CPU"
frameSeparationTime = 0.001 #1ms frame to frame time interval
alivInitial  = 1 #initla guess for aliv parameter curve fitting
swiftInitial = 1 # initial guess for swiftness parameter curve fitting
bounds = (0, np.inf)
# save_dcm_pickle(data_path)

def swift_aliv(data):
    concatenate_set = utils.concatenate_scan_set(data)
    blockRepeat = data.shape[0]
    blockPerVolume = 1 #only 1 block is used in our protocal
    bscanLocationPerBlock = data.shape[1] #the number of B-scan in one 3D volume
    numLocation = bscanLocationPerBlock * blockPerVolume # Number of total B-scan
    # print('Processing: ' + data_path)
    ## OCT intensity
    height = concatenate_set.shape[1]
    width = concatenate_set.shape[2]
    aliv = np.zeros((numLocation, height, width))
    swift = np.zeros((numLocation, height, width))
    oct_db = np.zeros((numLocation, height, width))
    for floc in range(0,numLocation):
        sparseSequence, timePoints, frameindex = utils.makeSparseDataFromRasterRepeat(concatenate_set, bscanLocationPerBlock, blockRepeat, floc)
        if floc == 0: #for save VLIV array
            VLIV_save = np.zeros((numLocation, timePoints.shape[0]-1, height, width))
        oct_db[floc] = 10*np.log10(np.nanmean(sparseSequence + 1e-8, axis=0))
        ## Compute VLIV
        VLIV , possibleMtw , VoV = utils.computeVLIV(sparseSequence, timePoints, maxTimeWidth =  np.nan, compute_VoV = False)
        ## Average LIV curve
        if average_LivCurve == True:
            twIdx = 0
            for twIdx in range(0, VLIV.shape[0]):                
                VLIV[twIdx,:,:] = cv2.blur(VLIV[twIdx,:,:], (3,3))
                twIdx = twIdx + 1
        if fitting_method == 'CPU':
            mag, tau = utils.vlivCPUFitExp(VLIV, possibleMtw, frameSeparationTime, alivInitial, swiftInitial, bounds, use_constraint = False)
        aliv[floc] = mag ## aLIV
        swift[floc] = 1/ tau ## Swiftness
        # VLIV_save[floc,:,:,:] = VLIV ## LIV curve (VLIV)
    octRange = (np.min(oct_db), np.max(oct_db))
    alivRange = (0, 10)
    swiftRange = (0, 3)
    # slice_index = 5
    # swift_slice = swift[:, slice_index, :]
    # swift_slice[swift_slice > np.percentile(swift_slice, 95)] = np.median(swift_slice)
    # swift_normalized = cv2.normalize(swift_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    # fourD_normalized = cv2.normalize(fourD_image[10, :, slice_index, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    aliv_rgb = generate_RgbImage(doct = aliv, dbInt = oct_db, doctRange = alivRange, octRange = octRange, scale=0.7)
    swift_rgb = generate_RgbImage(doct = swift, dbInt = oct_db, doctRange = swiftRange, octRange = octRange, scale=0.7)
    # swift_rgb_slice = swift_rgb[:, slice_index, :]
    # print(swift_rgb)
    return swift_rgb, aliv_rgb
        

def load_nested_data_h5(path):
    pic_paths = []
    for scan_num in natsorted(os.listdir(path)):
        if scan_num.startswith('scan'):
            pic_paths.append(os.path.join(path,scan_num,f'{scan_num}.h5'))

    with h5py.File(pic_paths[0], 'r') as hf:
        b = np.array(hf['volume'])
    data = np.zeros((len(pic_paths),b.shape[0],b.shape[1],b.shape[2]))
    for idx,img_path in enumerate(pic_paths):
        with h5py.File(img_path, 'r') as hf:
            temp = np.array(hf['volume'])
        data[idx]=(temp.copy())
    return data

def generate_RgbImage(doct, dbInt, doctRange, octRange, scale):
    hsvImage = np.stack([utils.scale_clip(doct, *doctRange, scale), np.ones_like(doct),
                       utils.scale_clip(dbInt, *octRange)], axis=-1)
    rgbImage = hsv_to_rgb(hsvImage)
    return rgbImage

def process_batch_data(args):
    batch, shm_name, window_size, data_shape = args
    shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(data_shape, dtype=np.float64, buffer=shm.buf)
    result = swift_aliv(data[batch:batch+window_size, :, :, :])
    shm.close()
    return result

if __name__ == '__main__':
    image_list = os.listdir(data_path)
    fourD_image_volume_complete = load_nested_data_h5(data_path)
    fourD_image_volume_complete = fourD_image_volume_complete[:, :, 48:52, :]
    if fourD_image_volume_complete.min()<0:
        fourD_image_volume_complete -= fourD_image_volume_complete.min()
    # fourD_image = fourD_image_volume_complete[:20, :, 20:30, 20:]
    shm = shared_memory.SharedMemory(create=True, size=fourD_image_volume_complete.nbytes)
    shared_data = np.ndarray(fourD_image_volume_complete.shape, dtype=np.float64, buffer=shm.buf)
    np.copyto(shared_data, fourD_image_volume_complete)
    window_size = 20
    num_vols = fourD_image_volume_complete.shape[0] if fourD_image_volume_complete.shape[0] % 2 == 0 else fourD_image_volume_complete.shape[0] + 1
    swift_rgb_mask_shape = ((num_vols - window_size) // 2, fourD_image_volume_complete.shape[1], fourD_image_volume_complete.shape[2], fourD_image_volume_complete.shape[3],3)
    swift_rgb_mask = np.zeros(swift_rgb_mask_shape, dtype=np.float64)

    aliv_rgb_mask_shape = ((num_vols - window_size) // 2, fourD_image_volume_complete.shape[1], fourD_image_volume_complete.shape[2], fourD_image_volume_complete.shape[3],3)
    aliv_rgb_mask = np.zeros(aliv_rgb_mask_shape, dtype=np.float64)
    tasks = []
    for batch in range(0, fourD_image_volume_complete.shape[0] - window_size, 2):
        tasks.append((batch, shm.name, window_size, fourD_image_volume_complete.shape))
    with Pool(processes=120) as pool:
        results = list(pool.imap(process_batch_data, tasks))
    idx = 0
    for batch_number, batch in enumerate(range(0, fourD_image_volume_complete.shape[0] - window_size, 2)):
        swift_rgb_mask[batch_number] = results[idx][0]
        aliv_rgb_mask[batch_number] = results[idx][1]
        idx += 1
    shm.close()
    shm.unlink()
    os.makedirs(f'swift/{folder_name}_{data_name}',exist_ok=True)
    with open(f'swift/{folder_name}_{data_name}/swift_rgb.pickle', 'wb') as handle:
        pickle.dump(swift_rgb_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(f'aliv/{folder_name}_{data_name}',exist_ok=True)
    with open(f'aliv/{folder_name}_{data_name}/aliv_rgb.pickle', 'wb') as handle:
        pickle.dump(aliv_rgb_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
