import pandas as pd
import numpy as np
import scipy.interpolate


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
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
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