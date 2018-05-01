from __future__ import print_function

import os
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from step1 import step1_python
import warnings
import time

import multiprocessing
import Queue

class ParallelProcessCaller(object):
    def __init__(self, cmd, args):
        self.cmd = cmd
        self.__result = None
        self.__queue = multiprocessing.Queue()
        self.proc = multiprocessing.Process(target=ParallelProcessCaller.process_target, args=(self.__queue, cmd) + tuple(args))
        self.proc.start()
    
    @staticmethod
    def process_target(queue, cmd, *args):
        queue.put(cmd(*args))
        
    @property
    def result(self):
        if self.proc is not None:
            while True:
                try:
                    self.__result = self.__queue.get(timeout=1)
                except Queue.Empty:
                    if self.proc.is_alive:
                        continue
                    else:
                        raise Exception("subprocess died unexpectedly")
                break
            self.proc.join()
            self.proc = None
        return self.__result


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id,filelist,prep_folder,data_path,use_existing=True):      
    resolution = np.array([1,1,1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
        
        st = time.time()
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')
        
        print("pstep1, block1", time.time() - st)



        convex_mask = m1
        
        pc1 = ParallelProcessCaller(process_mask, (m1,))
        pc2 = ParallelProcessCaller(process_mask, (m2,))
        dm1 = pc1.result
        dm2 = pc2.result
        
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        print("pstep1, block2", time.time() - st)
        
        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]

        print("pstep1, block3", time.time() - st)
        
        print("  -> Writing", os.path.join(prep_folder, name))
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
    except:
        print('bug in '+name)
        raise
    print(name+' done')

    
# Thanks to: http://mindcache.io/2015/08/09/python-multiprocessing-module-daemonic-processes-are-not-allowed-to-have-children.html   
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

import multiprocessing.pool
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def full_prep(data_path,prep_folder,n_worker = 1, use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

            
    print('starting preprocessing')
    filelist = [f for f in os.listdir(data_path)]
    print("Processing", len(filelist), "files with", n_worker, "workers")

    if n_worker <= 1:
        for i in range(len(filelist)):
            savenpy(i, filelist, prep_folder, data_path, use_existing=use_existing)
    else:
        partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
                                  data_path=data_path,use_existing=use_existing)
        N = len(filelist)
        pool = NoDaemonProcessPool(n_worker)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        
    print('end preprocessing')
    return filelist
