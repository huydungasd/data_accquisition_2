import pandas as pd
import numpy as np
import scipy.interpolate as interp


def data_transform(data, sensibility):
    assert data.shape[1] % 2 == 0
    
    data_list = []
    for i in range(int(data.shape[1]/2)):
        data_trans = data.iloc[:, i*2+1] * 2**8 + data.iloc[:, i*2]
        data_trans[data_trans > 32767] -= 65536
        data_trans /= sensibility
        data_list.append(data_trans)
    return pd.concat(data_list, axis=1)

def irread_to_m(data):
	# Source: https://www.upgradeindustries.com/product/58/Sharp-10-80cm-Infrared-Distance-Sensor-(GP2Y0A21YK0F)
    return (data * 5) ** -1.15 * 27.86 / 100

def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = interp.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

def create_imu_data_deep(filepath, frequency=100):
    df = pd.read_csv(filepath)
    df_interp = interpolate_3dvector_linear(df, df.iloc[:, 0], np.arange(df.iloc[0, 0], df.iloc[-1, 0], 0.01))
    df_interp = pd.DataFrame(df_interp, columns=list(df.columns))
    return df_interp

def coeff_determination(h, l, a0, b0, t0, t1, t2):
    AB = np.sqrt((a0 - b0)**2 + l**2)
    dt1 = t1 - t0
    dt2 = t2 - t0
    a = AB * (0.5 * dt2 - dt1) / (dt1 * dt2 * (dt1 - dt2))
    b = AB * (0.5 * dt2**2 - dt1**2) / (dt1 * dt2 * (dt2 - dt1))
    if a < 0:
        print("Warning!! a < 0")
    return a, b, AB

def position_calulation(time, i_depart, i_final, h, l, a0, b0, t0, t1, t2):
    a, b, AB = coeff_determination(h, l, a0, b0, t0, t1, t2)
    distance = time.copy()
    x = time.copy()
    y = time.copy()
    z = time.copy()
    for i in range(i_depart, i_final + 1):
        distance.iloc[i] = a * (time.iloc[i] - time.iloc[i_depart])**2 + b * (time.iloc[i] - time.iloc[i_depart])
    distance.iloc[:i_depart] = 0
    distance.iloc[i_final+1:] = distance.iloc[i_final]
    for i in range(time.size):
        z.iloc[i] = h - h * distance.iloc[i] / AB
        x.iloc[i] = -np.sqrt(l**2 - h**2) * distance.iloc[i] / AB
        y.iloc[i] = b0 + (a0 - b0) * distance.iloc[i] / AB
    x.rename('pos_x', inplace=True)
    y.rename('pos_y', inplace=True)
    z.rename('pos_z', inplace=True)
    return x, y, z

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

def SHOE(imudata, g=9.8, W=5, G=4.1e8, sigma_a=0.00098**2, sigma_w=(8.7266463e-5)**2):
    T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    inv_a = 1/sigma_a
    inv_w = 1/sigma_w
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - g * smean_a/np.linalg.norm(smean_a)).dot(( a - g * smean_a/np.linalg.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    return zupt < G


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean()
    B_mB = B - B.mean()

    # Sum of squares across rows
    ssA = (A_mA**2).sum()
    ssB = (B_mB**2).sum()

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA,ssB))

def skew(omega):
    assert omega.shape == (3,)
    return np.array([[  0,          -omega[2],  omega[1]    ],
                     [  omega[2],   0,          -omega[0]   ],
                     [  -omega[1],  omega[0],   0           ]])

# def cal_A(gyro_np, idx):
#     omega = gyro_np[idx]
#     domega = (gyro_np[idx + 1] - gyro_np[idx])/0.01
#     return skew(omega) @ skew(omega) + skew(domega)

# def cal_A2(quat, idx):
#     omega_m1 = R.from_quat(quat[idx - 1]).as_euler(seq='xyz')
#     omega = R.from_quat(quat[idx]).as_euler(seq='xyz')
#     omega_p1 = R.from_quat(quat[idx + 1]).as_euler(seq='xyz')

#     velo = (omega_p1 - omega_m1) / (2 * 0.01)
#     acc = (omega_p1 - 2 * omega + omega_m1) / (0.01**2)
#     return skew(velo) @ skew(velo) + skew(acc)