import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from scipy.ndimage.interpolation import shift

def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = interp.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

def data_transform(data, sensibility):
    assert data.shape[1] % 2 == 0
    
    data_list = []
    for i in range(int(data.shape[1]/2)):
        data_trans = data.iloc[:, i*2+1] * 2**8 + data.iloc[:, i*2]
        data_trans[data_trans > 32767] -= 65536
        data_trans /= sensibility
        data_list.append(data_trans)
    return pd.concat(data_list, axis=1)


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean()
    B_mB = B - B.mean()

    # Sum of squares across rows
    ssA = (A_mA**2).sum()
    ssB = (B_mB**2).sum()

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA,ssB))

path = "H:\\data\\9"
gt = np.loadtxt(path + '\\pose.txt')
gt[:, 0] /= 1e9

start_imu = pd.read_csv(path + '\\0.csv', nrows=1, header=None).iloc[0, 1]
df = pd.read_csv(path + '\\0.csv', header=1)
step1_time = df.iloc[0, 0] - start_imu

f = np.loadtxt(path + '\\gyro.txt')
f[:, 0] /= 1e9
f = interpolate_3dvector_linear(f, f[:, 0], np.arange(f[0, 0], f[-1, 0], 0.01))
start_tango = np.where(np.abs(f[int(100*step1_time)-200:int(100*step1_time)+500, 3]) > 1)[0][0] + int(100*step1_time)-200


time = df.iloc[:, 0]
acc_raw = df.iloc[:, 1:7]
mag_raw = df.iloc[:, 7:13]
gyr_raw = df.iloc[:, 13:19]
ori_raw = df.iloc[:, 19:25]
quat_raw = df.iloc[:, 25:33]

start_measurement = np.where((time.diff(1) > 0.5).to_numpy())[0][0]
moment_start_measurement = time[start_measurement]

acc = data_transform(acc_raw, 100)
acc.columns = ['acc_x', 'acc_y', 'acc_z']
mag = data_transform(mag_raw, 900)
mag.columns = ['mag_x', 'mag_y', 'mag_z']
gyr = data_transform(gyr_raw, 16) / 180.0 * np.pi
gyr.columns = ['gyr_x', 'gyr_y', 'gyr_z']
ori = data_transform(ori_raw, 16) / 180.0 * np.pi
ori.columns = ['ori_z', 'ori_y', 'ori_x']
quat = data_transform(quat_raw, 2**14)
quat.columns = ['q', 'p1', 'p2', 'p3']

# data = pd.concat([time, acc, mag, gyr, ori, quat], axis=1)
# data.to_csv(path + '\\m0.csv', index=False)

time = time.to_numpy()
gyro_np = gyr.to_numpy()

ori_np = interpolate_3dvector_linear(ori.to_numpy(), time, np.arange(time[0], time[-1], 0.01))
gyro_np = interpolate_3dvector_linear(gyro_np, time, np.arange(time[0], time[-1], 0.01))
time = interpolate_3dvector_linear(time, time, np.arange(time[0], time[-1], 0.01))
start_measurement = np.where(time > moment_start_measurement)[0][0] + 1000 # +1000 to take off 1sec after start the measurement


period = 200
stop_tango = start_tango + period

cor = -100
idx_align_imu = 0
for i in range(gyro_np.shape[0] - period):
    window = gyro_np[i:i+period, 2]
    if cor < corr2_coeff(f[start_tango:stop_tango, 3], window):
        cor = corr2_coeff(f[start_tango:stop_tango, 3], window)
        idx_align_imu = i
n_pts = min(len(time) - start_measurement, len(f) - (start_tango + start_measurement - idx_align_imu))


plt.figure()
plt.plot(f[start_tango:start_tango+period, 0] - f[start_tango, 0], f[start_tango:start_tango+period, 3], label='tango')
plt.plot(time[idx_align_imu:idx_align_imu+period] - time[idx_align_imu], gyro_np[idx_align_imu:idx_align_imu+period, 2], label='imu')
plt.legend()

gt = interpolate_3dvector_linear(gt, gt[:, 0], f[start_tango + start_measurement - idx_align_imu: start_tango + start_measurement - idx_align_imu + n_pts, 0])
gt = gt - np.array([gt[0]])


plt.figure()
plt.plot(time, ori_np[:, 0]/np.pi*180)


plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=[14.4, 10.8])
ax = fig.gca(projection='3d')
ax.plot(gt[:, 1], gt[:, 2], gt[:, 3])
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.1, 0.1)
ax.set_zlim(0, 0.1)
plt.show()
