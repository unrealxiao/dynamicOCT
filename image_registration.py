import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as scp
from natsort import natsorted
import cv2
import multiprocessing
from time import time
# from itertools import repeat


# Run the script using python testing.py 0/1/2
# 0 means before, 1 means after, 2 means after2min data

def load_data(path_num,range_frames=None):
    if path_num==0:
        path = '../../data/before/'
    elif path_num==1:
        path = '../../data/after/'
    elif path_num==2:
        path = 'D:/xiaoliu_onedrive/OneDrive - Indiana University/lab/Dynamic_OCT/registration_png/'
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.dcm') or  i.endswith('.DCM') or i.endswith('.PNG'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)
    if range_frames:
        pic_paths = pic_paths[range_frames-50:range_frames+50]
    pics_without_line = []

    for i in pic_paths:
        aa = cv2.imread(path+i,cv2.IMREAD_UNCHANGED)
        pics_without_line.append(aa.copy())

    pics_without_line = np.array(pics_without_line).astype(np.float32)
    zero_line_down= []
    zero_line_up = []
    zero_line_left = []
    zero_line_right = []
    for i in range(pics_without_line.shape[0]):
        for down in range(pics_without_line[i].shape[0]-1,-1,-1):
            if np.any(pics_without_line[i][down,:]!=0):
                zero_line_down.append(down)
                break
        for up in range(0,pics_without_line[i].shape[0]):
            if np.any(pics_without_line[i][up,:]!=0):
                zero_line_up.append(up)
                break
        for left in range(0,pics_without_line[i].shape[1]):
            if np.any(pics_without_line[i][:,left]!=0):
                zero_line_left.append(left)
                break
        for right in range(pics_without_line[i].shape[1]-1,-1,-1):
            if np.any(pics_without_line[i][:,right]!=0):
                zero_line_right.append(right)
                break
    zero_line_down = np.min(zero_line_down)
    zero_line_up = np.min(zero_line_up)
    zero_line_left = np.min(zero_line_left)
    zero_line_right = np.min(zero_line_right)
    pics_without_line[:,zero_line_down:,:] =  0
    pics_without_line[:,:zero_line_up,:] =  0
    pics_without_line[:,:,:zero_line_left] =  0
    pics_without_line[:,:,zero_line_right:] =  0
    return pics_without_line


def min_max(data1):
    if np.all(data1 == data1[0]):
        return data1
    else:
        data1 = (data1-np.min(data1))/(np.max(data1)-np.min(data1))
        return data1





# def slope_mask_10batch(arr,p1):
#     mask1 = np.zeros_like(arr[0],dtype=np.float32)
#     arr = arr.astype(np.float32)
#     std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)
#     arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
#     for x in range(arr.shape[1]):
#         for y in range(arr.shape[2]):
#             data1 = arr[:,x,y].astype(np.float32).copy()
#             slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
#             mask1[x, y] = -np.abs(slope1)
#     return mask1*std_mask

## image without using shared memory
# def image(x, y, arr):
#     data1 = arr[:,x,y].astype(np.float32).copy()
#     slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
#     return x, y, slope1


def initpool(mask_arr):
    global X_mask
    X_mask = mask_arr


def image(x, y, X_shape):
    X_np = np.frombuffer(X_mask, dtype=np.float32).reshape(X_shape)
    data1 = X_np[:,x,y].astype(np.float32).copy()
    slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
    return x, y, -np.abs(slope1)


# def image(x, y, X_shape):
#     X_np = np.frombuffer(X_mask.get_obj(), dtype=np.float32).reshape(X_shape)
#     data1 = X_np[:,x,y].astype(np.float32).copy()
#     slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
#     return x, y, -np.abs(slope1)

# def image(x, y, X_np):
#     #X_np = np.frombuffer(X_mask.get_obj(), dtype=np.float32).reshape(X_shape)
#     data1 = X_np[:,x,y].astype(np.float32).copy()
#     slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
#     return x, y, -np.abs(slope1)

if __name__ == '__main__':
    tic = time()
    num=2
    print("begin load data")
    data = load_data(num)
    data = data.astype(np.float32)
    #mask = np.zeros((len(range(50,2500-50,2)),data.shape[1],data.shape[2]),dtype=np.float32)
    # j=0
    # for i in range(50,150,2):
    arr = data
    # X_shape = (arr.shape[1], arr.shape[2])
    mask1 = np.zeros_like(arr[0], dtype=np.float32)
    ##create shared memory X_np
    # X = multiprocessing.Array('f', X_shape[0] * X_shape[1], lock=True)
    # X_np = np.frombuffer(X.get_obj(), dtype=np.float32).reshape(X_shape)
    # np.copyto(X_np, mask1)

    arr = arr.astype(np.float32)
    std_mask = np.apply_along_axis(func1d=np.std,arr=arr,axis=0)   
    arr = np.apply_along_axis(func1d=min_max,arr=arr,axis=0)
    


    print("start multicore processing")
    # pool = multiprocessing.Pool(processes=8, initializer=initpool, initargs=(X,))
    # pool.starmap(image, zip(range(arr.shape[1]), repeat(arr)))
    # for x in range(arr.shape[1]):
    #     for y in range(arr.shape[2]):
    #         # report = 'coordinate [' + repr(x) + ',' + repr(y) + '] start'
    #         # print(report)
    #         pool.apply_async(image, (x, y,))
    # F_mask = np.frombuffer(X.get_obj(), dtype=np.float32).reshape(X_shape)
    # pool.close()
    # pool.join()
       # mask = F_mask*std_mask 

    X_shape = (arr.shape[0], arr.shape[1], arr.shape[2])
    X = multiprocessing.Array('f', X_shape[0] * X_shape[1] * X_shape[2], lock=False)
    # Pa_shape = multiprocessing.Array('i', X_shape)#to share the shape data
    # X_np = np.frombuffer(X.get_obj(), dtype=np.float32).reshape(X_shape)
    X_np = np.frombuffer(X, dtype=np.float32).reshape(X_shape)
    np.copyto(X_np, arr)
    pool = multiprocessing.Pool(processes=3, initializer=initpool, initargs=(X,))
    res = [pool.apply_async(image, args=(x, y, X_shape)) for x in range(arr.shape[1]) for y in range(arr.shape[2])]
    pool.close()
    pool.join()
    for r in res:
        x, y, value = r.get()
        mask1[x, y] = value
    
    mask = mask1*std_mask
    toc = time()
    print(mask)

    # with multiprocessing.Pool(processes=8, initializer=initpool, initargs=(X,)) as pool:
    #     res = [pool.apply_async(image, args=(x, y)) for x in range(arr.shape[1]) for y in range(arr.shape[2])]
    #     for r in res:
    #         x, y, value = r.get()
    #         mask1[x, y] = value
    #     print(mask1)

        # j+=1
    
    print('Done in {:.4f} seconds'.format(toc-tic))
    # with open(f'D:/xiaoliu_onedrive/OneDrive - Indiana University/lab/Dynamic_OCT/registration_png/mask_{num}.pickle', 'wb') as handle:
    #     pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
