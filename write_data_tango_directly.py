import shutil
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# matplotlib.use('Agg')

from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation as R

from utils.data_processing import *


display_graphic = True
for id_f in range(1):
    path = f"/home/huydung/devel/intern/data/4eme/{id_f}"
    tango_gt_unalign = np.loadtxt(path + '/pose.txt')
    tango_gt_unalign[:, 0] /= 1e9

    start_imu = pd.read_csv(path + '/0.csv', nrows=1, header=None).iloc[0, 1]
    imu_unalign = pd.read_csv(path + '/0.csv', header=1)
    step1_time = imu_unalign.iloc[0, 0] - start_imu

    tango_acc_unalign = np.loadtxt(path + '/acce.txt')
    tango_gyr_unalign = np.loadtxt(path + '/gyro.txt')
    tango_ori_unalign = np.loadtxt(path + '/orientation.txt')
    tango_ori_unalign[:, 1:] = tango_ori_unalign[:, 1:] * np.array([[-1,1,1,-1]])
    tango_acc_unalign[:, 0] /= 1e9
    tango_gyr_unalign[:, 0] /= 1e9
    tango_ori_unalign[:, 0] /= 1e9
    t_min_tango = tango_acc_unalign[0, 0]
    t_max_tango = tango_acc_unalign[-1, 0]

    if tango_ori_unalign[0, 0] > t_min_tango:
        t_min_tango = tango_ori_unalign[0, 0]
    if tango_gyr_unalign[0, 0] > t_min_tango:
        t_min_tango = tango_gyr_unalign[0, 0]
    if tango_gt_unalign[0, 0] > t_min_tango:
        t_min_tango = tango_gt_unalign[0, 0]

    if tango_ori_unalign[-1, 0] < t_max_tango:
        t_max_tango = tango_ori_unalign[-1, 0]
    if tango_gyr_unalign[-1, 0] < t_max_tango:
        t_max_tango = tango_gyr_unalign[-1, 0]
    if tango_gt_unalign[-1, 0] < t_max_tango:
        t_max_tango = tango_gt_unalign[-1, 0]

    tango_gyr_unalign = interpolate_3dvector_linear(tango_gyr_unalign, tango_gyr_unalign[:, 0], np.arange(t_min_tango, t_max_tango, 0.01))
    tango_ori_unalign = interpolate_3dvector_linear(tango_ori_unalign, tango_ori_unalign[:, 0], np.arange(t_min_tango, t_max_tango, 0.01))
    tango_gt_unalign = interpolate_3dvector_linear(tango_gt_unalign, tango_gt_unalign[:, 0], np.arange(t_min_tango, t_max_tango, 0.01))
    tango_acc_unalign = interpolate_3dvector_linear(tango_acc_unalign, tango_acc_unalign[:, 0], np.arange(t_min_tango, t_max_tango, 0.01))
    start_tango = np.where(np.abs(tango_gyr_unalign[int(100*step1_time)-100:int(100*step1_time)+500, 3]) > 1)[0][0] + int(100*step1_time)-100


    time_imu_unalign = imu_unalign.iloc[:, 0]
    imu_acc_unalign = imu_unalign.iloc[:, 1:7]
    imu_mag_unalign = imu_unalign.iloc[:, 7:13]
    imu_gyr_unalign = imu_unalign.iloc[:, 13:19]
    imu_ori_unalign = imu_unalign.iloc[:, 19:25]
    imu_quat_unalign = imu_unalign.iloc[:, 25:33]
    tango_position_unalign = tango_gt_unalign[:, 1:4]
    tango_ori2_unalign = tango_gt_unalign[:, 4:8]

    imu_start_measurement = np.where((time_imu_unalign.diff(1) > 0.5).to_numpy())[0][0]
    moment_imu_start_measurement = time_imu_unalign[imu_start_measurement]

    imu_acc_unalign = data_transform(imu_acc_unalign, 100)
    imu_acc_unalign.columns = ['acc_x', 'acc_y', 'acc_z']
    imu_mag_unalign = data_transform(imu_mag_unalign, 900)
    imu_mag_unalign.columns = ['mag_x', 'mag_y', 'mag_z']
    imu_gyr_unalign = data_transform(imu_gyr_unalign, 16) / 180.0 * np.pi
    imu_gyr_unalign.columns = ['gyr_x', 'gyr_y', 'gyr_z']
    imu_ori_unalign = data_transform(imu_ori_unalign, 16) / 180.0 * np.pi
    imu_ori_unalign.columns = ['ori_z', 'ori_y', 'ori_x']
    imu_quat_unalign = data_transform(imu_quat_unalign, 2**14)
    imu_quat_unalign.columns = ['q', 'p1', 'p2', 'p3']

    time_imu_unalign = time_imu_unalign.to_numpy()
    imu_gyr_unalign = imu_gyr_unalign.to_numpy()
    imu_acc_unalign = imu_acc_unalign.to_numpy()
    imu_mag_unalign = imu_mag_unalign.to_numpy()
    imu_quat_unalign = imu_quat_unalign.to_numpy()

    imu_acc_unalign = interpolate_3dvector_linear(imu_acc_unalign, time_imu_unalign, np.arange(time_imu_unalign[0], time_imu_unalign[-1], 0.01))
    imu_mag_unalign = interpolate_3dvector_linear(imu_mag_unalign, time_imu_unalign, np.arange(time_imu_unalign[0], time_imu_unalign[-1], 0.01))
    imu_quat_unalign = interpolate_3dvector_linear(imu_quat_unalign, time_imu_unalign, np.arange(time_imu_unalign[0], time_imu_unalign[-1], 0.01))
    imu_gyr_unalign = interpolate_3dvector_linear(imu_gyr_unalign, time_imu_unalign, np.arange(time_imu_unalign[0], time_imu_unalign[-1], 0.01))
    time_imu_unalign = np.arange(time_imu_unalign[0], time_imu_unalign[-1], 0.01)
    imu_start_measurement = np.where(time_imu_unalign > moment_imu_start_measurement)[0][0] + 100 # +100 to take off 0.1sec after start the measurement


    period = 200
    stop_tango = start_tango + period

    cor = -100
    idx_align_imu = 0
    for i in range(imu_gyr_unalign.shape[0] - period):
        window = imu_gyr_unalign[i:i+period, 2]
        if cor < corr2_coeff(tango_gyr_unalign[start_tango:stop_tango, 3], window):
            cor = corr2_coeff(tango_gyr_unalign[start_tango:stop_tango, 3], window)
            idx_align_imu = i
    n_pts = min(len(time_imu_unalign) - imu_start_measurement, len(tango_gyr_unalign) - (start_tango + imu_start_measurement - idx_align_imu))

    imu_acc_align = imu_acc_unalign[imu_start_measurement: imu_start_measurement + n_pts]
    imu_mag_align = imu_mag_unalign[imu_start_measurement: imu_start_measurement + n_pts]
    imu_ori_align = imu_ori_unalign[imu_start_measurement: imu_start_measurement + n_pts]
    imu_quat_align = imu_quat_unalign[imu_start_measurement: imu_start_measurement + n_pts]
    imu_gyr_align = imu_gyr_unalign[imu_start_measurement: imu_start_measurement + n_pts]
    time_imu_align = time_imu_unalign[imu_start_measurement: imu_start_measurement + n_pts]

    tango_acc_align = tango_acc_unalign[start_tango + imu_start_measurement - idx_align_imu: start_tango + imu_start_measurement - idx_align_imu + n_pts]
    tango_gyr_align = tango_gyr_unalign[start_tango + imu_start_measurement - idx_align_imu: start_tango + imu_start_measurement - idx_align_imu + n_pts]
    tango_ori_align = tango_ori_unalign[start_tango + imu_start_measurement - idx_align_imu: start_tango + imu_start_measurement - idx_align_imu + n_pts]
    tango_position_align = tango_position_unalign[start_tango + imu_start_measurement - idx_align_imu: start_tango + imu_start_measurement - idx_align_imu + n_pts]
    tango_position_align = tango_position_align - tango_position_align[0]
    tango_ori2_align = tango_ori2_unalign[start_tango + imu_start_measurement - idx_align_imu: start_tango + imu_start_measurement - idx_align_imu + n_pts]
    tango_ori2_align = tango_ori2_align * np.array([[-1,1,1,-1]])


    if display_graphic:
        plt.figure()
        plt.plot(tango_gyr_unalign[start_tango: start_tango + period, 0] - tango_gyr_unalign[start_tango, 0], tango_gyr_unalign[start_tango: start_tango + period, 3], label='tango')
        plt.plot(time_imu_unalign[idx_align_imu: idx_align_imu + period] - time_imu_unalign[idx_align_imu], imu_gyr_unalign[idx_align_imu: idx_align_imu + period, 2], label='imu')
        plt.title("Synchronize gyroscope")
        plt.legend()

        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        for i in range(3):
            axes[i].plot(tango_ori_align[:, 0] - tango_ori_align[0, 0], R.from_quat(tango_ori_align[:, [2, 1, 4, 3]]).as_euler(seq='xyz')[:, i] / np.pi * 180, label='tango' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].plot(time_imu_align[:] - time_imu_align[0], R.from_quat(imu_quat_align[:, [1, 2, 3, 0]]).as_euler(seq='xyz')[:, i] / np.pi * 180, label='imu' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].set_title('Orientation' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].legend()

        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        for i in range(3):
            axes[i].plot(tango_gyr_align[:, 0] - tango_gyr_align[0, 0], tango_gyr_align[:, i+1] / np.pi * 180, label='tango' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].plot(time_imu_align[:] - time_imu_align[0], imu_gyr_align[:, i] / np.pi * 180, label='imu' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].set_title('Vitesse angulaire' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].legend()

        _, axes = plt.subplots(1, 3, figsize=(13, 4))
        for i in range(3):
            axes[i].plot(tango_acc_align[1:, 0] - tango_acc_align[0, 0], tango_acc_align[1:, i+1], label='tango' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].plot(time_imu_align[1:] - time_imu_align[0], imu_acc_align[1:, i], label='imu'  + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].set_ylim(-3, 13)
            axes[i].set_title('Acceleration' + ' x' * (i==0) + ' y' * (i==1) + ' z' * (i==2))
            axes[i].legend()

    if display_graphic:
        g = np.zeros((tango_acc_align.shape[0], tango_acc_align.shape[1] - 1))
        g2 = np.zeros((tango_acc_align.shape[0], tango_acc_align.shape[1] - 1))
        for i in range(n_pts):
            g[i] = R.from_quat(tango_ori_align[i, [2,1,4,3]]).as_matrix() @ tango_acc_align[i, 1:]
            g2[i] = R.from_quat(imu_quat_align[i, [1,2,3,0]]).as_matrix() @ imu_acc_align[i]
        plt.figure()
        plt.plot(time_imu_align, g)
        plt.plot(time_imu_align, g2)
        print(f"g: {np.mean(g, axis=0)}")

        plt.rcParams.update({'font.size': 18})
        fig = plt.figure(figsize=[14.4, 10.8])
        ax = fig.gca(projection='3d')
        ax.plot(tango_position_align[:, 0], tango_position_align[:, 1], tango_position_align[:, 2])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.1)
        ax.set_zlim(0, 0.1)

    idxx = SHOE(np.hstack([tango_acc_align[:, 1:], tango_gyr_align[:, 1:]]),G=1e5)
    lists = []
    start = 0
    stop = 0
    for i in range(len(idxx)):
        if not idxx[i]:
            if i < stop:
                continue
            if i == 0 or idxx[i-1]:
                start = i
                for j in range(i+1, len(idxx)):
                    if idxx[j]:
                        stop = j    
                        if np.linalg.norm(tango_position_align[stop] - tango_position_align[start]) > 0.3 and stop - start < 1000:
                            lists.append([start, stop])
                        break


    try:
        shutil.rmtree(path + '/data_deep')
    except:
        pass

    try:
        os.makedirs(path + '/data_deep')
        os.makedirs(path + '/data_deep/gt')
        os.makedirs(path + '/data_deep/imu')
    except:
        pass
    
    tango_ori_align = tango_ori_align[:, [3,2,1,4]]
    tango_ori2_align = tango_ori2_align[:, [2,1,0,3]]

    tango_position_align = (R.from_quat(tango_ori_align[0, [3,0,1,2]]).as_matrix() @ R.from_quat(tango_ori2_align[0, [3,0,1,2]]).inv().as_matrix() @ tango_position_align.T).T

    if display_graphic:
        fig = plt.figure(figsize=[14.4, 10.8])
        ax = fig.gca(projection='3d')
        ax.plot(tango_position_align[idxx, 0], tango_position_align[idxx, 1], tango_position_align[idxx, 2], '.b')
        ax.plot(tango_position_align[~idxx, 0], tango_position_align[~idxx, 1], tango_position_align[~idxx, 2], 'r')
        ax.set_xlim(-0.5, 0.1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.1)
        plt.show()

    imu_acc = pd.DataFrame(imu_acc_align, index=None, columns=["acc_x", "acc_y", "acc_z"])
    imu_gyr = pd.DataFrame(imu_gyr_align, index=None, columns=["gyr_x", "gyr_y", "gyr_z"])
    imu_mag = pd.DataFrame(imu_mag_align, index=None, columns=["mag_x", "mag_y", "mag_z"])
    time_imu = pd.DataFrame(time_imu_align, index=None, columns=["time"])
    data = pd.concat([time_imu, imu_acc, imu_mag, imu_gyr], axis=1)
    data.to_csv(path + f'/imu.csv', index=False)

    tango_ori = pd.DataFrame(tango_ori_align, index=None, columns=["q", "x", "y", "z"])
    tango_position = pd.DataFrame(tango_position_align, index=None, columns=["x", "y", "z"])
    data = pd.concat([time_imu, tango_position, tango_ori], axis=1)
    data.to_csv(path + f'/gt.csv', index=False)

    for i, lst in enumerate(lists):
        print(lst[0])
        print(lst[1])
        print('-------------------------')
        imu_acc = pd.DataFrame(imu_acc_align[lst[0]:lst[1]], index=None, columns=["acc_x", "acc_y", "acc_z"])
        imu_gyr = pd.DataFrame(imu_gyr_align[lst[0]:lst[1]], index=None, columns=["gyr_x", "gyr_y", "gyr_z"])
        imu_mag = pd.DataFrame(imu_mag_align[lst[0]:lst[1]], index=None, columns=["mag_x", "mag_y", "mag_z"])
        time_imu = pd.DataFrame(time_imu_align[lst[0]:lst[1]], index=None, columns=["time"])
        data = pd.concat([time_imu, imu_acc, imu_mag, imu_gyr], axis=1)
        data.to_csv(path + f'/data_deep/imu/{i}.csv', index=False)

        tango_ori = pd.DataFrame(tango_ori_align[lst[0]:lst[1]], index=None, columns=["q", "x", "y", "z"])
        tango_position = pd.DataFrame(tango_position_align[lst[0]:lst[1]], index=None, columns=["x", "y", "z"])
        data = pd.concat([time_imu, tango_position, tango_ori], axis=1)
        data.to_csv(path + f'/data_deep/gt/{i}.csv', index=False)

    