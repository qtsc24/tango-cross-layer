# -*- coding: utf-8 -*-
from __future__ import division
#import rados
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy import stats
import numpy as np
import math
import time
import struct
import sys
import os
import subprocess
from sklearn.cluster import KMeans
import scipy.fftpack as fft
from threading import Thread
import threading
import argparse
import datetime
from pytz import timezone
import gc
from ctypes import *
from numpy.ctypeslib import ndpointer
from multiprocessing import Process
from multiprocessing import Pool
np.set_printoptions(threshold=np.inf)

def accuracy_tranfer(error_metric, accuracy):
    if error_metric == "NRMSE":
        pseudo_accuracy = -math.log10(accuracy)
    elif error_metric == "PSNR":
        pseudo_accuracy = accuracy
    return pseudo_accuracy

def get_weight(error_metric, datasize, priority, accuracy, max_size, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag):
    max_weight = 1000
    min_weight = 100
    min_size = 1/1024/1024
    if error_metric == "NRMSE":
        min_a = -math.log10(min_accuracy)
        max_a = -math.log10(max_accuracy)
    elif error_metric == "PSNR":
        min_a = min_accuracy
        max_a = max_accuracy
    k = (max_weight - min_weight)/((max_size * max_priority/min_a) - (min_size * min_priority/max_a))
    b = min_weight - k * min_size * min_priority/max_a


    if weight_tag == "size":
        weight = (k * datasize * 3)/accuracy_tranfer(error_metric, max_accuracy) + b
    elif weight_tag == "size_priority":
        weight = (k * datasize * priority)/accuracy_tranfer(error_metric, max_accuracy) + b
    elif weight_tag == "size_priority_accuracy":
        weight = (k * datasize * priority)/accuracy_tranfer(error_metric, accuracy) + b

    return weight

def data_size_vs_weight(size,fulldata_len):
    weight_high = 1000
    weight_low = 100
    size_high = fulldata_len
    size_low = 1*1024*1024
    k = (weight_high - weight_low)/(size_high - size_low)
    b = weight_high - k*size_high
    weight = k * size + b
    if weight > 1000:
        print("weight larger than 1000, check datasize_vs_weight function!\n")
    return int(weight)

def find_large_elements(source,threshold):
    max_gradient=0.0
    for i in range(len(source)):
        if math.fabs(source[i])> max_gradient:
            max_gradient = math.fabs(source[i])
     
    return max_gradient * threshold 

def find_augment_points_gradient(base,chosn_index,threshold):
    if threshold == 0.0:
        return [range(len(base))]
    elif threshold == 1.0:
        return []
    chosn_points=[]

    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    #print("-1\n"
    #print(sys.getsizeof(base)/1024/1024
    base_gradient=np.gradient(base)
    #print("0\n"
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(base_gradient[j])
    #print("1\n"   
    thre = find_large_elements(chosn_points, threshold)

    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if math.fabs(base_gradient[j])> thre:
                #print("VIP=",j
                temp_1.append(j)
        #if len(temp_1)>1:
        temp_index.append(temp_1)
        temp_1=[]
    #print("temp_index=",temp_index                                          
    for i in temp_index:
        if len(i)>1:
            for j in range(1,len(i)):
                if i[j]-i[j-1]>1:
                    temp_interval.append(i[j]-i[j-1]) 
    #print("temp_interval=",temp_interval
    #print("temp_interval",temp_interval
    if len(temp_interval) ==1:
        max_intv = temp_interval[0]
    else: 
        max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    #print("max_intv=",max_intv
    #print(temp_index                  
    temp_2=[]

    for i in temp_index:
        temp_2.append(i[0])
        if len(i) > 1:
            for j in range(1,len(i)):
                if i[j]-i[j-1] <= max_intv:
                    temp_2.append(i[j])
                else:
                    #if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
                    temp_2=[]
                    temp_2.append(i[j])
            #if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
            temp_2=[]
    #Not Finish
    last_tag = chosn_index_finer[0][-1]
    for i in chosn_index_finer:
        #before_len = len(i)
        tm = i[-1]
        if i[-1]!=len(dpot_L1)-1 and i[0]-last_tag>1:
            i.append(i[-1]+1)
        if i[0]!=0:
            i.append(i[0]-1)
        i.sort()
        last_tag = tm
    return chosn_index_finer
def partial_refinement_new1(timestep, peak_noise, no_noise, ctag, chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
    #base_index = range(len(base))
    #print("len(base_index)=",len(base_index)
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)

    full_delta=np.array(range(finer_len))
    if len(chosn_index) != finer_len:
        non_refine = np.array(list(set(full_delta).difference(set(chosn_index))))
        zero_delta = np.zeros(finer_len - len(chosn_data), dtype = np.float64)
        print(np.shape(chosn_index), np.shape(non_refine))
        chosn_index = np.concatenate((chosn_index, non_refine), axis=0)
        chosn_data = np.concatenate((chosn_data, zero_delta), axis=0)
        chosn_r = np.concatenate((chosn_r, zero_delta), axis=0)
        chosn_z = np.concatenate((chosn_z, zero_delta), axis=0)
    print("len(chosn_index) = %s, len(chosn_data) = %s, len(chosn_r) = %s, len(chosn_z) = %s\n"%(len(chosn_index), len(chosn_data), len(chosn_r), len(chosn_z)))
        #print("np.shape(chosn_index) = %s, np.shape(chosn_data) = %s, np.shape(chosn_r) = %s, np.shape(chosn_z) = %s\n"%(np.shape(chosn_index), np.shape(chosn_data), np.shape(chosn_r), np.shape(chosn_z))
    #print("np.shape(chosn_index)=",np.shape(chosn_index)
    for i in range(len(chosn_index)):
        #if i%10000 ==0:
        #    print(i
        index1 = chosn_index[i] // deci_ratio  
        index2 = chosn_index[i] % deci_ratio
        if index2 == 0:
            finer[chosn_index[i]] = base[index1]
            finer_r[chosn_index[i]] = base_r[index1]
            finer_z[chosn_index[i]] = base_z[index1]
        else:
            if index1 != len(base)-1:
                finer[chosn_index[i]]=(base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i]
                #finer_p.append((base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i])
                #finer_r_p.append((base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i])
                #finer_z_p.append((base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i])
            else:
                finer[chosn_index[i]]= 2* base[index1]*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=2 * base_r[index1]*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=2 * base_z[index1]*index2/deci_ratio + chosn_z[i]
                #finer_p.append(2* base[index1]*index2/deci_ratio + chosn_data[i])
                #finer_r_p.append(2 * base_r[index1]*index2/deci_ratio + chosn_r[i])
                #finer_z_p.append(2 * base_z[index1]*index2/deci_ratio + chosn_z[i])
    print("len(finer) = %d, len(finer_r) = %d, len(finer_z) = %d"%(len(finer), len(finer_r), len(finer_z)))
    finer_name = ssd_path+application_name+"_"+str(peak_noise)+"_"+str(no_noise)+"_"+str(timestep)+"_"+str(ctag)+"_psnr.npz"
    np.savez(finer_name, psnr_finer = finer, psnr_finer_r = finer_r, psnr_finer_z = finer_z)
    finer_name = ssd_path+application_name+"_"+str(peak_noise)+"_"+str(no_noise)+"_"+str(timestep)+"_"+str(ctag)+"_plot.npz"
    np.savez(finer_p_name, finer = finer_p, finer_r = finer_r_p, finer_z = finer_z_p)
    del finer
    del finer_r
    del finer_z

    del chosn_index
    del chosn_data
    del chosn_r
    del chosn_z
    del base
    del base_r
    del base_z
    #print("reference value:", sys.getrefcount(finer), sys.getrefcount(finer_r),sys.getrefcount(finer_z),sys.getrefcount(chosn_index),sys.getrefcount(chosn_data),sys.getrefcount(chosn_r),sys.getrefcount(chosn_z),sys.getrefcount(base),sys.getrefcount(base_r),sys.getrefcount(base_z)
    del gc.garbage[:]
    gc.collect()
    gc.collect()
    gc.collect()
    print("Finish timestep", timestep)
    #return finer, finer_r, finer_z,finer_p, finer_r_p, finer_z_p
def partial_refinement_new(timestep, peak_noise, no_noise, ctag, chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
 
    base = tuple(base)
    base_r = tuple(base_r)
    base_z = tuple(base_z)
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    full_delta=np.array(range(finer_len))
    refine_len = len(chosn_data)
    if len(chosn_index) != finer_len:
        non_refine = np.array(list(set(full_delta).difference(set(chosn_index))))
        zero_delta = np.zeros(finer_len - len(chosn_data), dtype = np.float64)
        print(np.shape(chosn_index), np.shape(non_refine))
        chosn_index = tuple(np.concatenate((chosn_index, non_refine), axis=0))
        chosn_data = tuple(np.concatenate((chosn_data, zero_delta), axis=0))
        chosn_r = tuple(np.concatenate((chosn_r, zero_delta), axis=0))
        chosn_z = tuple(np.concatenate((chosn_z, zero_delta), axis=0))

    for i in range(len(chosn_index)):
        #if i%10000 ==0:
        #    print(i
        index1 = chosn_index[i] // deci_ratio
        index2 = chosn_index[i] % deci_ratio
        if index2 == 0:
            finer[chosn_index[i]] = base[index1]
            finer_r[chosn_index[i]] = base_r[index1]
            finer_z[chosn_index[i]] = base_z[index1]
            finer_p.append(base[index1])
            finer_r_p.append(base_r[index1])
            finer_z_p.append(base_z[index1])
        else:
            if index1 != len(base)-1:
                finer[chosn_index[i]]=(base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i]
                if i < refine_len:
                    finer_p.append((base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i])
                    finer_r_p.append((base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i])
                    finer_z_p.append((base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i])
            else:
                finer[chosn_index[i]]= 2* base[index1]*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=2 * base_r[index1]*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=2 * base_z[index1]*index2/deci_ratio + chosn_z[i]
                if i < refine_len:
                    finer_p.append(2* base[index1]*index2/deci_ratio + chosn_data[i])
                    finer_r_p.append(2 * base_r[index1]*index2/deci_ratio + chosn_r[i])
                    finer_z_p.append(2 * base_z[index1]*index2/deci_ratio + chosn_z[i])
    #print("len(finer) = %d, len(finer_r) = %d, len(finer_z) = %d"%(len(finer), len(finer_r), len(finer_z))
    finer_name = ssd_path+application_name+"_"+str(peak_noise)+"_"+str(no_noise)+"_"+str(timestep)+"_"+str(ctag)+"_psnr.npz"
    np.savez(finer_name, psnr_finer = finer, psnr_finer_r = finer_r, psnr_finer_z = finer_z)
    finer_p_name = ssd_path+application_name+"_"+str(peak_noise)+"_"+str(no_noise)+"_"+str(timestep)+"_"+str(ctag)+"_plot.npz"
    np.savez(finer_p_name, finer = finer_p, finer_r = finer_r_p, finer_z = finer_z_p)

    del finer
    del finer_r
    del finer_z

    del chosn_index
    del chosn_data
    del chosn_r
    del chosn_z
    del base
    del base_r
    del base_z
    #print("reference value:", sys.getrefcount(finer), sys.getrefcount(finer_r),sys.getrefcount(finer_z),sys.getrefcount(chosn_index),sys.getrefcount(chosn_data),sys.getrefcount(chosn_r),sys.getrefcount(chosn_z),sys.getrefcount(base),sys.getrefcount(base_r),sys.getrefcount(base_z)
    del gc.garbage[:]
    gc.collect()
    print("Finish timestep", timestep)
          
def partial_refinement(timestep, chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    start = 0
    inc = 0
    i=0
    num_refine=0
    refine_index=[]
    while(i<len(chosn_index)):
        
        for m in range(start,chosn_index[i]):
            index1 = m // deci_ratio
            index2 = m % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    finer[m]=(base[index1]+base[index1+1])*index2/deci_ratio
                    finer_r[m]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                    finer_z[m]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
            else:
                if index2!=0:
                    finer[m]=2*base[index1]*index2/deci_ratio
                    finer_r[m]=2*base_r[index1]*index2/deci_ratio
                    finer_z[m]=2*base_z[index1]*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
        for n in range(chosn_index[i],chosn_index[i+1]+1):
            index1 = n // deci_ratio
            index2 = n % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)

            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)

            inc+=1
        start = chosn_index[i+1]+1
        if i == len(chosn_index)-2:
            for j in range(start,finer_len):
                index1 = j // deci_ratio
                index2 = j % deci_ratio
                if index1!=len(base)-1:
                    if index2!=0:
                        finer[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                        finer_r[j]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                        finer_z[j]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]

                else:
                    if index2!=0:
                        finer[j]=2*base[index1]*index2/deci_ratio
                        finer_r[j]=2*base_r[index1]*index2/deci_ratio
                        finer_z[j]=2*base_z[index1]*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]

        i+=2

    print("finish timestep ", timestep)
    return finer, finer_r, finer_z

def k_means(source,savefig,k):
    if k == "true":
        find_k(source,savefig)
    #print(source
    y=np.array(source).reshape(-1,1)
        
    km=KMeans(n_clusters=2)
    km.fit(y)
    km_label=km.labels_

    if len(km_label)!=len(source):
        print("length issue")
    sorted_cluster_index=np.argsort(km.cluster_centers_.reshape(-1,))

    group=sorted_cluster_index[:1]

    limit = 0
    for i in range(len(source)):
        if km_label[i] in group:
            #print(km_label[i]
            if source[i]>limit:
                #print(source[i],limit
                limit = source[i]
    return limit

def noise_threshold(noise, peak_noise, no_noise):
    print("Actual peak_noise=", peak_noise)
    print("Actual no_noise=", no_noise)

    k=(1.0-0.0)/(no_noise - peak_noise)
    b =1.0 - k * no_noise

    if noise < peak_noise:
        threshold = 0.0
    elif noise > no_noise:
        threshold = 1.0 
    else:
        threshold = noise*k+b
        #thre.append(threshold)

    return threshold

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def prediction_noise_wave(samples, hi_freq_ratio,timestep_interval):

    sample_rate = 1/timestep_interval
    Nsamples = len(samples)
    print("sample rate = ",sample_rate)
    #amp = fft.fft(samples)/(Nsamples/2.0)
    amp = fft.fft(samples)/Nsamples
    #amp_complex_h = amp[range(int(len(samples)/2))]
    amp_complex_h = amp
    amp_h = np.absolute(amp_complex_h)

    freq=fft.fftfreq(amp.size,1/sample_rate)
    freq_h = freq
    #freq_h = freq[range(int(len(samples)/2))] 

    if amp_h[0]>1e-10:
        threshold = np.max(np.delete(amp_h,0,axis=0))*hi_freq_ratio
        dc = amp_h[0]
        start_index = 1
    else:
        threshold = np.max(amp_h)*hi_freq_ratio
        dc = 0.0
        start_index = 0

    selected_freq = []
    selected_amp = []
    selected_complex=[]
    for i in range(start_index,len(amp_h)):
        if amp_h[i]>=threshold:
            selected_freq.append(freq_h[i])
            selected_amp.append(amp_h[i])
            selected_complex.append(amp_complex_h[i])

    selected_phase = np.arctan2(np.array(selected_complex).imag,np.array(selected_complex).real)

    for i in range(len(selected_phase)):
        if np.fabs(selected_phase[i])<1e-10:
            selected_phase[i]=0.0

    return dc, selected_amp, selected_freq, selected_phase   

 
def get_prediction_threshold(dc, selected_amp, selected_freq, selected_phase, time, peak_noise, no_noise):
    sig = dc
    for i in range(len(selected_freq)):
        sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*time+ selected_phase[i])
    if sig < 0 :
        sig = 1.1
    print("Noise amplitude=",sig)
    threshold = noise_threshold(sig, peak_noise, no_noise)
    return threshold, sig

def get_chosn_data_index(threshold):
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    #r_L1_compressed=f.read(2975952*8)
    #r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    #z_L1_compressed=f.read(2516984*8)
    #z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
     
    if threshold == 0.0:
        chosn_index_L1 = []            
    elif threshold == 1.0:
        chosn_index_L1 = [range(len(dpot_L1))]
    else:
        chosn_index_L1 = find_augment_points_gradient(dpot_L1,[range(len(dpot_L1))], 1-threshold)

    return chosn_index_L1    

def calc_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1,p2,p3
    #return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)
    area = abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))*0.5)
    return area

def high_potential_area(dpot,R,Z,thre):
    start = time.time()
    points = np.transpose(np.array([Z,R]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    area=0.0
    for i in range(len(conn)):
        index1=conn[i][0]
        index2=conn[i][1]
        index3=conn[i][2]
        #if (dpot[index1]>thre and dpot[index2]>thre) or (dpot[index1]>thre and dpot[index3]>thre) or (dpot[index2]>thre and dpot[index3]>thre):
        if (dpot[index1]+dpot[index2]+dpot[index3])/3.0 > thre:
            each_area=calc_area((R[index1],Z[index1]),(R[index2],Z[index2]),(R[index3],Z[index3]))
            area = area + each_area
    end = time.time()
    print("High potential analysis time = ", end - start)
    return area

class myThread (threading.Thread):
    def __init__(self, ssd_read_time, full_read_time, full_all_delta_read_time, deci_ratio,timestep,tag,time_tail,weight_bool, fulldata_len, reduced_len,docker_path_id, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag):
        threading.Thread.__init__(self)
        self.ssd_read_time = ssd_read_time
        self.full_read_time = full_read_time
        self.full_all_delta_read_time = full_all_delta_read_time
        self.deci_ratio = deci_ratio
        self.timestep = timestep
        self.tag = tag 
        self.time_tail = time_tail
        self.weight_bool = weight_bool
        self.fulldata_len = fulldata_len
        self.reduced_len = reduced_len
        self.docker_path_id = docker_path_id
        self.error_metric = error_metric
        self.error_target = error_target
        self.nrmse_target_list = nrmse_target_list
        self.psnr_target_list = psnr_target_list
        self.priority = priority
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.weight_tag = weight_tag 
    def run(self):                   
        fully_refine(ssd_read_time, full_read_time, full_all_delta_read_time, deci_ratio,timestep,tag,time_tail,weight_bool, fulldata_len, reduced_len,docker_path_id, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag)

def fully_refine(full_weight_bool, ssd_read_time, full_read_time, full_all_delta_read_time, deci_ratio,timestep,tag,time_tail,weight_bool, fulldata_len, reduced_len,docker_path_id, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag):
    a = time.time()
    
    start_ssd = time.time()
    print("Application name = " , application_name)
    filename = ssd_path+"reduced_data_"+application_name+"_16.bin"
    f = open(filename, "rb")
    dpot_L1_str=f.read(reduced_len*8)
    r_L1_str=f.read(reduced_len*8)
    z_L1_str=f.read(reduced_len*8)
    f.close()
    end_ssd = time.time()
    dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
    r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
    z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)

    read_ssd_time = end_ssd- start_ssd
    ssd_read_time.append(read_ssd_time)

    if timestep !=-1:    
        delta_L0_L1_str = b""
        delta_r_L0_L1_str = b""
        delta_z_L0_L1_str = b""
        docker_cgroup_path = "/sys/fs/cgroup/blkio/docker/" + docker_path_id + "/blkio.bfq.weight"
        io_time = 0.0
        if full_weight_bool == True:
            read_len = []
            read_len_r = []
            read_len_z = []
            bitplane_size = fulldata_len//64 + 1
            n_bitplanes_decode = r_n_bitplanes_decode = z_n_bitplanes_decode = 64

            if error_metric == "NRMSE":
                bitplanes_len_name = ssd_path+"metadata_nrmse_"+application_name+".bin"
                error_target_list = nrmse_target_list
            elif error_metric == "PSNR":
                bitplanes_len_name = ssd_path+"metadata_psnr_"+application_name+".bin"
                error_target_list = psnr_target_list
            error_target_len = len(error_target_list)
            f = open(bitplanes_len_name, "rb")
            meta_len = error_target_len * 3 + 3
            meta_str=f.read(meta_len * 4)
            f.close()
            meta = struct.unpack(str(meta_len)+'i',meta_str)
            #print("meta = ", meta
            delta_n_bitplane_nrmse = list(meta[:error_target_len])
            delta_r_n_bitplane_nrmse = list(meta[error_target_len:error_target_len*2])
            delta_z_n_bitplane_nrmse = list(meta[error_target_len*2:error_target_len*3])

            delta_level_exp = meta[error_target_len*3]
            delta_r_level_exp = meta[error_target_len*3 + 1]
            delta_z_level_exp = meta[error_target_len*3 + 2]

            delta_n_bitplane_nrmse[-1] = n_bitplanes_decode - sum(delta_n_bitplane_nrmse[:-1])
            delta_r_n_bitplane_nrmse[-1] = r_n_bitplanes_decode - sum(delta_r_n_bitplane_nrmse[:-1])
            delta_z_n_bitplane_nrmse[-1] = z_n_bitplanes_decode - sum(delta_z_n_bitplane_nrmse[:-1])

            print("delta_n_bitplane_nrmse = ", delta_n_bitplane_nrmse)
            print("delta_r_n_bitplane_nrmse = ", delta_r_n_bitplane_nrmse)           
            print("delta_z_n_bitplane_nrmse = ", delta_z_n_bitplane_nrmse)

            for i in range(error_target_len):
                read_len.append(delta_n_bitplane_nrmse[i] * bitplane_size)
                read_len_r.append(delta_r_n_bitplane_nrmse[i] * bitplane_size)
                read_len_z.append(delta_z_n_bitplane_nrmse[i] * bitplane_size)
            
            print("Read len =",read_len)
            print("Read len R =",read_len_r)
            print("Read len Z =",read_len_z)

            delta_name = hdd_path+"delta_bitplane_"+application_name+".bin"
            delta_r_name = hdd_path+"delta_r_bitplane_"+application_name+".bin"
            delta_z_name = hdd_path+"delta_z_bitplane_"+application_name+".bin"

            f = open(delta_name, "rb")

            for i in range(len(read_len)):
                if read_len[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len[i] * 8/1024/1024, priority, error_target, bitplane_size * 64 *8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                #print("weight = ", weight
                delta_low_start = time.time()
                delta_L0_L1_str += f.read(read_len[i]*8)
                delta_low_end = time.time()
                io_time += delta_low_end - delta_low_start
            f.close()

            f = open(delta_r_name, "rb")
            for i in range(len(read_len_r)):
                if read_len_r[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len_r[i] * 8/1024/1024, priority, error_target, bitplane_size * 64 * 8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                #print("R weight = ", weight
                delta_low_start = time.time()
                delta_r_L0_L1_str += f.read(read_len_r[i]*8)
                delta_low_end = time.time()
                io_time += delta_low_end - delta_low_start
            f.close()

            f = open(delta_z_name, "rb")
            for i in range(len(read_len_z)):
                if read_len_z[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len_z[i] * 8/1024/1024, priority, error_target, bitplane_size * 64 * 8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                #print("Z weight = ", weight
                delta_low_start = time.time()
                delta_z_L0_L1_str += f.read(read_len_z[i]*8)
                delta_low_end = time.time()
                io_time += delta_low_end - delta_low_start
            f.close()
        else:
            weight = 100

            docker_cgroup_path = "/sys/fs/cgroup/blkio/docker/" + docker_path_id + "/blkio.bfq.weight"
            os.system("echo %d > %s"%(weight, docker_cgroup_path))
            os.system("cat %s"%(docker_cgroup_path))

            start=time.time()

            f = open(hdd_path+"delta_"+application_name+"_o.bin", "rb")
            delta_L0_L1_str=f.read()
            f.close()

            f = open(hdd_path+"delta_r_"+application_name+"_o.bin", "rb")
            delta_r_L0_L1_str=f.read()
            f.close()

            f = open(hdd_path+"delta_z_"+application_name+"_o.bin", "rb")
            delta_z_L0_L1_str=f.read()
            f.close()

            end = time.time()
            io_time = end - start

    full_all_delta_read_time.append(io_time)
    delta_read_time = io_time
    print("Data read = %d Mb"%int((len(delta_L0_L1_str)+len(delta_r_L0_L1_str)+len(delta_z_L0_L1_str))/1024/1024))
    print("All delta reading time = ", delta_read_time)
    if timestep !=-1:
        delta_L0_L1 = struct.unpack(str(int(len(delta_L0_L1_str)/8))+'d',delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack(str(int(len(delta_r_L0_L1_str)/8))+'d',delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack(str(int(len(delta_z_L0_L1_str)/8))+'d',delta_z_L0_L1_str)
        sample_bandwidth = (len(delta_L0_L1_str)+ len(delta_r_L0_L1_str)+ len(delta_z_L0_L1_str))/1024/1024/delta_read_time
    else:
        delta_L0_L1 = struct.unpack('d',delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack('d',delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack('d',delta_z_L0_L1_str)
        sample_bandwidth = (8*3)/1024/1024/delta_read_time
    #print("Delta reading bandwidth = ", (ioctx_2.stat("delta_L0_L1")[0]+ ioctx_2.stat("delta_r_L0_L1")[0] + ioctx_2.stat("delta_z_L0_L1")[0])/1024/1024/delta_time
    fname = ssd_path+"sample_"+str(tag)+"_"+application_name+".npz"
        
    if timestep != 0:
        fp = np.load(fname)
        sample_bd = fp['sample_bandwidth']
        sample_read_time = fp['sample_read_time']
    else:
        sample_bd = np.array([]) 
        sample_read_time = np.array([])
    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()
    
    sample_bd.append(sample_bandwidth)
    sample_read_time.append(delta_read_time)

    print("Total bandwidth samples=",sample_bd)

    print("Total time samples=",sample_read_time)
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))
 
    refine_start = time.time()

    refine_end = time.time()
    print("Refinement time = ", refine_end- refine_start)
    #print("start plot\n"
    #a=time.time()
    b=time.time()

    del dpot_L1_str
    del r_L1_str
    del z_L1_str
    del dpot_L1
    del r_L1
    del z_L1
    #if full_weight_bool == True:
    #    del trunk_len_str
    #    del trunk_len
    del delta_L0_L1_str
    del delta_r_L0_L1_str
    del delta_z_L0_L1_str
    del delta_L0_L1
    del delta_r_L0_L1
    del delta_z_L0_L1
    #print("reference value:", sys.getrefcount(dpot_L1_str), sys.getrefcount(r_L1_str),sys.getrefcount(z_L1_str),sys.getrefcount(dpot_L1),sys.getrefcount(r_L1),sys.getrefcount(z_L1),sys.getrefcount(delta_index_L0_L1_str),sys.getrefcount(delta_L0_L1_str),sys.getrefcount(delta_r_L0_L1_str),sys.getrefcount(delta_z_L0_L1_str), sys.getrefcount(delta_L0_L1),sys.getrefcount(delta_r_L0_L1),sys.getrefcount(delta_z_L0_L1)
    del gc.garbage[:]
    gc.collect()
    gc.collect()
    gc.collect()

    print("Fully refinement function time = ", b-a)
    #high_p_area = high_potential_area(finer, finer_r,finer_z,thre)
    #return high_p_area

def plot(data,r,z,filename):
    points = np.transpose(np.array([z,r]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    fig,ax=plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=26)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)

    axis_font = {'fontname':'Arial', 'size':'38'}

    #plt.xlabel('R', **axis_font)
    #plt.ylabel('Z',**axis_font )
    #plt.triplot(r,z,conn)
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=25));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig(filename, format='png')

def get_read_len(n_bitplanes_decode, delta_n_bitplane_nrmse, error_target_len, bitplane_size):
    if n_bitplanes_decode > delta_n_bitplane_nrmse[-1]:
        delta_n_bitplane_nrmse[-1] = n_bitplanes_decode
        read_end = error_target_len - 1
    else:
        for i in range(error_target_len):
            if n_bitplanes_decode <= delta_n_bitplane_nrmse[i]:
                delta_n_bitplane_nrmse[i] = n_bitplanes_decode
                read_end = i
    read_len = []
    prev = 0
    for i in range(read_end + 1):
        read_len.append((delta_n_bitplane_nrmse[i] - prev) * bitplane_size)
        prev = delta_n_bitplane_nrmse[i]
    return read_len

def partial_refine(low_accuracy_read_time, ssd_read_time, decode_time, full_read_time, deci_ratio, timestep, ctag, time_tail, peak_noise, no_noise, time_interval, fulldata_len, reduced_len, docker_path_id, weight_bool, error_control_bool, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag):
    
    start_ssd = time.time()

    filename = ssd_path+"reduced_data_"+application_name+"_16.bin"
    f = open(filename, "rb")
    dpot_L1_str=f.read(reduced_len*8)
    r_L1_str=f.read(reduced_len*8)
    z_L1_str=f.read(reduced_len*8)
    f.close()
    end_ssd = time.time()
    dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
    r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
    z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)        

    read_ssd_time = end_ssd- start_ssd
    ssd_read_time.append(read_ssd_time)
    #print("Time for reading reduced data from SSD =", read_ssd_time

    #f_ssd = np.load("/ssd/ssd_time_"+application_name+".npz") 
    #ssd_time=f_ssd["ssd_time"]
    #ssd_time = ssd_time.tolist()
    #ssd_time.append(read_ssd_time)
    #print("Total ssd_time = ",  ssd_time
    #np.savez("/ssd/ssd_time_"+application_name+".npz", ssd_time = np.array(ssd_time))
    fname = ssd_path+"recontructed_noise_"+ str(time_interval) + "_" + str(ctag-1)+"_"+application_name+".npz"
    #fname = "/ssd/recontructed_noise_"+ str(time_interval) + "_" + str(ctag-1)+"_"+application_name+"_v2.npz"
    f_np = np.load(fname)
    dc = f_np["recontructed_noise_dc"]
    s_freq = f_np["recontructed_noise_freq"]
    s_amp = f_np["recontructed_noise_amp"]
    s_phase = f_np["recontructed_noise_phase"]
    peak_noise_a = f_np["peak_noise"]
    no_noise_a = f_np["no_noise"]
    print("Detected peak_noise", peak_noise_a)
    print("Detected no_noise", no_noise_a)
    if peak_noise < peak_noise_a or no_noise > no_noise_a:
        print("Peak/no noise setting error!\n")
    #peak_noise = int((peak_noise_a//10+1)*10)
    #no_noise = int((no_noise_a//10)*10)
    thre, sig = get_prediction_threshold(dc, s_amp, s_freq, s_phase, timestep, peak_noise, no_noise)
    
    print("threshold=",thre)
    thre = 0.001

    delta_index_L0_L1_str = b""
    delta_L0_L1_str = b""
    delta_r_L0_L1_str = b""
    delta_z_L0_L1_str = b""
    delta_len=0
    delta_read_time = 0.0
    
    bitplane_size = fulldata_len//64 + 1
    encoded_bitplane_len = 20 * bitplane_size
    refine_len = int(thre * encoded_bitplane_len)
    n_bitplanes_decode = refine_len//bitplane_size
    r_n_bitplanes_decode = refine_len//bitplane_size
    z_n_bitplanes_decode = refine_len//bitplane_size

    print("bitplane size =", bitplane_size)
    print("read bitplane N:", n_bitplanes_decode)
    print("read bitplane R:", r_n_bitplanes_decode)
    print("read bitplane Z:", z_n_bitplanes_decode)

    if error_metric == "NRMSE":
        bitplanes_len_name = ssd_path+"metadata_nrmse_"+application_name+".bin"
        error_target_list = nrmse_target_list
    elif error_metric == "PSNR":
        bitplanes_len_name = ssd_path+"/ssd/metadata_psnr_"+application_name+".bin"
        error_target_list = psnr_target_list
    error_target_len = len(error_target_list)
    f = open(bitplanes_len_name, "rb")
    meta_len = error_target_len * 3 + 3
    meta_str=f.read(meta_len * 4)
    f.close()
    meta = struct.unpack(str(meta_len)+'i',meta_str)
    #print("meta = ", meta
    delta_n_bitplane_nrmse = list(meta[:error_target_len])
    delta_r_n_bitplane_nrmse = list(meta[error_target_len:error_target_len*2])
    delta_z_n_bitplane_nrmse = list(meta[error_target_len*2:error_target_len*3])
    low_accuracy_len = delta_n_bitplane_nrmse[0] * bitplane_size
    low_accuracy_len_r = delta_r_n_bitplane_nrmse[0] * bitplane_size
    low_accuracy_len_z = delta_z_n_bitplane_nrmse[0] * bitplane_size
    delta_level_exp = meta[error_target_len*3]
    delta_r_level_exp = meta[error_target_len*3 + 1]
    delta_z_level_exp = meta[error_target_len*3 + 2]
    if error_control_bool == True:
        error_target_index = error_target_list.index(error_target)
        
        if n_bitplanes_decode < delta_n_bitplane_nrmse[error_target_index]:
            n_bitplanes_decode = delta_n_bitplane_nrmse[error_target_index]
            print("Need read more bitplanes to reach error %f, total number of delta bitplanes = %d\n"%(error_target, n_bitplanes_decode))
        else:
            print("No need to read more Delta bitplanes, error target %f has already been met by reading %d bitplanes"%(error_target, n_bitplanes_decode))
        if r_n_bitplanes_decode < delta_r_n_bitplane_nrmse[error_target_index]:
            r_n_bitplanes_decode = delta_r_n_bitplane_nrmse[error_target_index]
            print("Need read more bitplanes to reach error %f, total number of delta R bitplanes = %d\n"%(error_target, r_n_bitplanes_decode))
        else:
            print("No need to read more Delta R bitplanes, error target %f has already been met by reading %d bitplanes"%(error_target, r_n_bitplanes_decode))

        if z_n_bitplanes_decode < delta_z_n_bitplane_nrmse[error_target_index]:
            z_n_bitplanes_decode = delta_z_n_bitplane_nrmse[error_target_index]
            print("Need read more bitplanes to reach error %f, total number of delta Z bitplanes = %d\n"%(error_target, z_n_bitplanes_decode))
        else:
            print("No need to read more Delta Z bitplanes, error target %f has already been met  by reading %d bitplanes"%(error_target, z_n_bitplanes_decode))

    # n_bitplanes_decode = 7
    # r_n_bitplanes_decode = 9
    # z_n_bitplanes_decode = 7

    read_len = get_read_len(n_bitplanes_decode, delta_n_bitplane_nrmse, error_target_len, bitplane_size)
    read_len_r = get_read_len(r_n_bitplanes_decode, delta_r_n_bitplane_nrmse, error_target_len, bitplane_size)
    read_len_z = get_read_len(z_n_bitplanes_decode, delta_z_n_bitplane_nrmse, error_target_len, bitplane_size)
    
    print("Read len =",read_len)
    print("Read len R =",read_len_r)
    print("Read len Z =",read_len_z)
        
    delta_name = hdd_path+"delta_bitplane_"+application_name+".bin"
    delta_r_name = hdd_path+"delta_r_bitplane_"+application_name+".bin"
    delta_z_name = hdd_path+"delta_z_bitplane_"+application_name+".bin"
    io_time = 0.0    
    if sig > -1:
    #if sig > peak_noise:
        delta_L0_L1_str = b""
        delta_r_L0_L1_str = b""
        delta_z_L0_L1_str = b""
        low_accuracy_read_t = 0.0
        docker_cgroup_path = "/sys/fs/cgroup/blkio/docker/" + docker_path_id + "/blkio.bfq.weight"
        if weight_bool == True:
            f = open(delta_name, "rb")
            
            for i in range(len(read_len)):
                if read_len[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len[i] * 8/1024/1024, priority, error_target, bitplane_size * 20 *8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)            
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                print("weight = ", weight)
                delta_low_start = time.time()
                delta_L0_L1_str += f.read(read_len[i]*8)
                delta_low_end = time.time()
                if i == 0:
                    low_accuracy_read_t += delta_low_end - delta_low_start
                io_time += delta_low_end - delta_low_start
            f.close()

            f = open(delta_r_name, "rb")
            for i in range(len(read_len_r)):
                if read_len_r[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len_r[i] * 8/1024/1024, priority, error_target, encoded_bitplane_len*8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                print("R weight = ", weight)
                delta_low_start = time.time()
                delta_r_L0_L1_str += f.read(read_len_r[i]*8)
                delta_low_end = time.time()
                if i == 0:
                    low_accuracy_read_t += delta_low_end - delta_low_start
                io_time += delta_low_end - delta_low_start
            f.close()   

            f = open(delta_z_name, "rb")
            for i in range(len(read_len_z)):
                if read_len_z[i] == 0:
                    continue
                weight = get_weight(error_metric, read_len_z[i] * 8/1024/1024, priority, error_target, encoded_bitplane_len*8/1024/1024, min_priority, max_priority, error_target_list[0], error_target_list[-1], weight_tag)
                os.system("echo %d > %s"%(weight, docker_cgroup_path))
                print("Z weight = ", weight)
                delta_low_start = time.time()
                delta_z_L0_L1_str += f.read(read_len_z[i]*8)
                delta_low_end = time.time()
                if i == 0:
                    low_accuracy_read_t += delta_low_end - delta_low_start
                io_time += delta_low_end - delta_low_start
            f.close()
            low_accuracy_read_time.append(low_accuracy_read_t)
        else:
            weight = 100
            print("weight control is off, set the default weight = ", weight)
            os.system("echo %d > %s"%(weight, docker_cgroup_path))
            
            start_io = time.time()
            f = open(delta_name, "rb")
            #delta_L0_L1_str = f.read(n_bitplanes_decode * bitplane_size * 8)
            low1_s = time.time()
            delta_L0_L1_str = f.read(low_accuracy_len * 8)
            low1_e = time.time()
            if (n_bitplanes_decode * bitplane_size - low_accuracy_len) > 0:
                delta_L0_L1_str += f.read((n_bitplanes_decode * bitplane_size - low_accuracy_len) * 8)
            f.close()

            f_r = open(delta_r_name, "rb")
            #delta_r_L0_L1_str = f_r.read(r_n_bitplanes_decode * bitplane_size * 8)
            low2_s = time.time()
            delta_r_L0_L1_str = f_r.read(low_accuracy_len_r * 8)
            low2_e = time.time()
            if (r_n_bitplanes_decode * bitplane_size - low_accuracy_len_r) > 0:
                delta_r_L0_L1_str += f_r.read((r_n_bitplanes_decode * bitplane_size - low_accuracy_len_r) * 8)
            f_r.close()

            f_z = open(delta_z_name, "rb")
            #delta_z_L0_L1_str = f_z.read(z_n_bitplanes_decode * bitplane_size * 8)
            low3_s = time.time()
            delta_z_L0_L1_str = f_z.read(low_accuracy_len_z * 8)
            low3_e = time.time()
            if (z_n_bitplanes_decode * bitplane_size - low_accuracy_len_z) > 0:
                delta_z_L0_L1_str += f_z.read((z_n_bitplanes_decode * bitplane_size - low_accuracy_len_z) * 8)
            f_z.close()

            end_io = time.time()
            low_accuracy_time = low1_e - low1_s + low2_e - low2_s + low3_e - low3_s
            print("low accuracy time = ", low_accuracy_time)
            io_time = end_io - start_io
            io_time = low_accuracy_time
        #print("Read time = ", io_time
        full_read_time.append(io_time)
        delta_L0_L1_encoded = struct.unpack(str(n_bitplanes_decode * bitplane_size)+'Q',delta_L0_L1_str)
        delta_r_L0_L1_encoded = struct.unpack(str(r_n_bitplanes_decode * bitplane_size)+'Q',delta_r_L0_L1_str)
        delta_z_L0_L1_encoded = struct.unpack(str(z_n_bitplanes_decode * bitplane_size)+'Q',delta_z_L0_L1_str)

        bit = CDLL('NegaDecode.so') 

        c_ulonglong_p = POINTER(c_ulonglong)
        delta_L0_L1_encoded = np.array(delta_L0_L1_encoded, dtype = np.ulonglong)
        delta_L0_L1_encoded_p = delta_L0_L1_encoded.ctypes.data_as(c_ulonglong_p)
        delta_r_L0_L1_encoded = np.array(delta_r_L0_L1_encoded, dtype = np.ulonglong)
        delta_r_L0_L1_encoded_p = delta_r_L0_L1_encoded.ctypes.data_as(c_ulonglong_p)
        delta_z_L0_L1_encoded = np.array(delta_z_L0_L1_encoded, dtype = np.ulonglong)
        delta_z_L0_L1_encoded_p = delta_z_L0_L1_encoded.ctypes.data_as(c_ulonglong_p)
        #input_len = ctypes.c_ulonglong * bitplane_size * n_bitplanes_decode
        bit.NegaDecode.argtypes = (POINTER(c_ulonglong), c_int, c_int, c_int)
        bit.NegaDecode.restype = ndpointer(dtype = c_double, shape = (fulldata_len,))
        #delta_ptr = ctypes.cast(delta_L0_L1_encoded, ctypes.POINTER(ctypes.c_ulonglong))
        start_decode = time.time()
        delta_L0_L1 = bit.NegaDecode(delta_L0_L1_encoded_p, fulldata_len, delta_level_exp, n_bitplanes_decode)
        delta_r_L0_L1 = bit.NegaDecode(delta_r_L0_L1_encoded_p, fulldata_len, delta_r_level_exp, r_n_bitplanes_decode)
        delta_z_L0_L1 = bit.NegaDecode(delta_z_L0_L1_encoded_p, fulldata_len, delta_z_level_exp, z_n_bitplanes_decode)
        end_deccode = time.time()
        decode_time.append(end_deccode - start_decode)
        print("Decode time = ", end_deccode - start_decode)
        total_read_size = (n_bitplanes_decode + r_n_bitplanes_decode + z_n_bitplanes_decode)* bitplane_size *8 /1024/1024
        print("Total Read size = %d Mb"%(total_read_size))
        sample_bandwidth = total_read_size/io_time
    
    else:
        sample_bandwidth = sig
        full_read_time.append(io_time)
        decode_time.append(0.0)
        low_accuracy_read_time.append(0.0)
        print("Not read any delta!")
        
    fname = ssd_path+"sample_"+str(ctag)+"_"+application_name+".npz"
    
    if timestep != 0:
        fp = np.load(fname)
        sample_bd = fp['sample_bandwidth']
        sample_read_time = fp['sample_read_time']
    else:
        sample_bd = np.array([])
        sample_read_time = np.array([])

    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()
    sample_bd.append(sample_bandwidth)
    print("Perceived bandwidt = ", sample_bandwidth)
    sample_read_time.append(io_time)
    bdstr=''
    print("Total bandwidth samples=",sample_bd)
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))
    aa =time.time()
    bb =time.time()
    print("Time for one time partial refinement=",bb-aa)
    del delta_L0_L1_str
    del delta_r_L0_L1_str
    del delta_z_L0_L1_str
    del gc.garbage[:]
    gc.collect()
    gc.collect()
    gc.collect()

    # if psnr =="True":
    #     psnr_start = time.time()
    #     filename = ssd_path+"full_data.bin"
    #     f = open(filename, "rb")
    #     dpot_str = f.read(finer_len*8)
    #     r_str = f.read(finer_len*8)
    #     z_str = f.read(finer_len*8)
    #     f.close()
    #     number_of_original_elements = str(finer_len)
    #     dpot=struct.unpack(number_of_original_elements+'d',dpot_str)
    #     r=struct.unpack(number_of_original_elements+'d',r_str)
    #     z=struct.unpack(number_of_original_elements+'d',z_str)
    #     data_len=[len(dpot_L1),len(dpot)]
    #     print(data_len)
    #     #psnr_original=psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
    #     if len(finer)!=len(dpot):
    #         print("finer len error!\n")
    #         print("len(finer)=%d len(dpot)=%d\n"%(len(finer), len(dpot)))
    #     if sig > peak_noise:
    #         psnr_finer=psnr_c(dpot, finer, data_len, deci_ratio, 0)
    #     else:
    #         psnr_finer = psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
    #     print("finer PSNR=",psnr_finer)
    #     if timestep == time_interval:
    #         np.savez(ssd_path+"psnr_"+application_name+".npz", psnr = [psnr_finer])
    #         print("Total PSNR =", [psnr_finer])
    #     else:
    #         fpp = np.load(ssd_path+"psnr_"+application_name+".npz")
    #         s_psnr = fpp["psnr"]
    #         s_psnr = s_psnr.tolist()
    #         s_psnr.append(psnr_finer)
    #         np.savez(ssd_path+"psnr_"+application_name+".npz", psnr = s_psnr)
    #         print("Total PSNR =", s_psnr)
    #     psnr_end = time.time()
    #     print("Time for calculate PSNR = ",psnr_end-psnr_start)
    
    #return high_p_area

def psnr_c(original_data, base, leveldata_len, deci_ratio, level_id):
    for i in range(len(leveldata_len)-level_id,len(leveldata_len)):
        leveldata=np.zeros(leveldata_len[i])
        for j in range(leveldata_len[i]):
            index1=j//deci_ratio
            index2=j%deci_ratio
            if index1!= len(base)-1:
                if index2 != 0:
                    leveldata[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
            else:
                if index2 != 0:
                    leveldata[j]=(base[index1]*2)*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
        base=leveldata
    if len(base) !=len(original_data):
        print("len(leveldata) !=len(original_data)")

    MSE = 0.0
    for i in range(len(original_data)):
        #print(i, original_data[i]-base[i]
        MSE=(original_data[i]-base[i])*(original_data[i]-base[i])+MSE
    MSE=MSE/len(original_data)
    if MSE < 1e-6:
        MSE=0.0
    if MSE ==0.0:
        print("Couldn't get PSNR of two identical data.")
        return 0
    else:
        psnr=10*math.log(np.max(original_data)**2/MSE,10)
    #print("psnr=",psnr
    return psnr
 
def work_flow(full_weight_bool, deci_ratio, time_interval, peak_noise, no_noise, fulldata_len,reduced_len,docker_path_id, weight_bool,error_control_bool,error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag, only_partial):
    w_start = time.time()
    if error_control_bool == True:
        print("error_metric = ", error_metric)
        print("error_target = ", error_target)
    if only_partial == True:
        tag = 1
        start_time = half_time
    else:
        tag = 0
        start_time = 0
    
    if full_weight_bool == True:
        end_time = half_time
    else:
        end_time = half_time*2

    timestep = 0
    last_tag = 0
    t_sample = np.arange(time_interval,half_time+1,time_interval)
    t_tail = t_sample[-1]
    t_tail_whole = t_sample[-1]
    full_delta_read_time = []
    full_all_delta_read_time = []
    all_partial_delta_read_time = []
    ssd_read_time = []
    low_accuracy_read_time = []
    decode_time = []
    for i in range(start_time, end_time ,time_interval):
        start=time.time()
        print("time step = %d\n" %(i))
        
        # update prediction
        if i == t_tail_whole and only_partial == False:
            print("start updating prediction")
            a=time.time()
            fname = ssd_path+"sample_"+str(tag)+"_"+application_name+".npz"
            fp = np.load(fname)
            samples_bd = fp['sample_bandwidth']
            dc, s_amp, s_freq, s_phase = prediction_noise_wave(samples_bd, 0.5, time_interval)
            t = np.arange(i-half_time*(tag+1),half_time+1, time_interval)
            t_tail = t[-1]
            t_tail_whole = t[-1] + half_time*(tag+1)   
            sig = dc  
            for j in range(len(s_freq)):
                sig += s_amp[j]*np.cos(2*np.pi*s_freq[j]*t+ s_phase[j])
            print("Predicted sig bandwidth=", sig.tolist())
            print("Predicted time steps = ", t.tolist())
            print("Predicted sig bandwidth Max = %f, Min = %f"%(np.max(sig), np.min(sig)))
            noise_low_bar = 0.0
            for m in range(len(t)):
                if t[m]%90 ==0 or t[m]%100 ==0 or t[m]%120 == 0:
                #if t[m]%120 ==0 or t[m]%150 ==0 or t[m]%100 ==0:
                    if sig[m]>noise_low_bar:
                        noise_low_bar = sig[m]
            
            r_noise_name = ssd_path+"recontructed_noise_"+ str(time_interval) + "_" + str(tag)+"_"+application_name+".npz"
            np.savez(r_noise_name, recontructed_noise_dc = dc, recontructed_noise_amp = s_amp,recontructed_noise_freq = s_freq, recontructed_noise_phase = s_phase, no_noise = max(sig), peak_noise = noise_low_bar)
            tag += 1
            b=time.time()
            print("Updating prediction time = ", b-a)
        
        # full read
        if tag == 0:
            timestep = i
            w_end = time.time()
            print("Phase I time = ", w_end - start)
            fully_refine(full_weight_bool, ssd_read_time, full_delta_read_time,full_all_delta_read_time,deci_ratio,i, tag, t_tail, weight_bool, fulldata_len, reduced_len, docker_path_id, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag)
        
        # partial read
        else: 
            timestep = i-half_time*tag
            w_end = time.time()
            print("Phase I time = ", w_end - start)
            partial_refine(low_accuracy_read_time, ssd_read_time, decode_time, all_partial_delta_read_time, deci_ratio, timestep, tag, t_tail, peak_noise, no_noise, time_interval, fulldata_len, reduced_len, docker_path_id, weight_bool, error_control_bool, error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag)
        end = time.time()
        print("Analysis time = ",end-start)
        time.sleep(time_interval-(end-start)) 
    print("SSD read time = ", ssd_read_time)
    print("Decode time = ",decode_time)
    print("Full all delta read time = ", full_all_delta_read_time)
    #print("Partial delta read time = ",partial_delta_read_time
    print("All partial delta read time = ", all_partial_delta_read_time)
    print("low_accuracy_read_time = ", low_accuracy_read_time)
    
half_time = 1800
application_name = "cfd"
deci_ratio = 16
time_interval = 60
frequency_cut_off = 0.5
peak_noise = 30
no_noise = 120
if application_name == "xgc":
    fulldata_len = 44928785
    reduced_len = 2808050
elif application_name == "astro2d":
    fulldata_len = 47403736
    reduced_len = 2962734
elif application_name == "cfd":
    fulldata_len = 30764603
    reduced_len = 1922788
error_metric = "NRMSE"
error_target = 0.01
priority = 10
min_priority = 1
max_priority = 10
min_accuracy = 0.01
max_accuracy = 0.00001
psnr_target_list = [30, 50, 70]
nrmse_target_list = [0.01, 0.001, 0.0001, 0.00001]
docker_path_id = os.popen(f'cat /proc/self/cgroup | grep "docker" | sed s/\\\\//\\\\n/g | tail -1').read().strip('\n')
ssd_path = "/ssd/"
hdd_path = "/hdd/"
#weight_tag option: "size", "size_priority", "size_priority_accuracy"
weight_tag = "size_priority_accuracy"
full_weight_bool = False
only_partial = True

def main():
    print("time =", half_time)
    print("docker ID =", docker_path_id)
    print("application_name = ", application_name)
    print("fulldata_len = ", fulldata_len)
    print("reduced_len = ", reduced_len)
    print("max_priority = ", max_priority)
    print("min_priority = ", min_priority)
    print("max_accuracy = ", max_accuracy)
    print("min_accuracy = ", min_accuracy)
    print("Max data size = ", int(fulldata_len*8/1024/1024))
    print("priority = ", priority)
    print("weight tag = ", weight_tag)
    print("peak_noise = ", peak_noise)
    print("no_noise", no_noise)

    if sys.argv[1] == "on":
        error_control_bool = True
    else:
        error_control_bool = False
    if sys.argv[2] == "on":
        weight_bool = True
    else:
        weight_bool = False

    tag = 1
    while (tag):
        now_time = datetime.datetime.now(timezone('UTC'))
        hour = now_time.hour
        minutes = now_time.minute
        seconds = now_time.second
        if sys.argv[3] == 'now':
            work_flow(full_weight_bool, deci_ratio, time_interval, peak_noise, no_noise, fulldata_len,reduced_len, docker_path_id, weight_bool,error_control_bool,error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag, only_partial)
            tag = 0
            break
        if hour == int(sys.argv[3]) and minutes == int(sys.argv[4]) and seconds == int(sys.argv[5]):
            work_flow(full_weight_bool, deci_ratio, time_interval, peak_noise, no_noise, fulldata_len,reduced_len, docker_path_id, weight_bool,error_control_bool,error_metric, error_target, nrmse_target_list, psnr_target_list, priority, min_priority, max_priority, min_accuracy, max_accuracy, weight_tag, only_partial)
            tag = 0
            break

if __name__ == "__main__":
    main()
