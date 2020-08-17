import os
import csv
import shutil

from scipy.spatial.transform import Rotation as R
from utils.data_processing import *


for data_num in range(10, 11):
    if data_num != 6:
        print(f'Making data{data_num}')
        data_dir = f'./transformed_data/data{data_num}/'
        data_files = [name for name in os.listdir(data_dir) if os.path.isfile(data_dir + name)]
        try:
            shutil.rmtree(f'./data_deep/data{data_num}')
            os.mkdir(f'./data_deep/data{data_num}')
            os.mkdir(f'./data_deep/data{data_num}/gt')
            os.mkdir(f'./data_deep/data{data_num}/imu')
        except FileNotFoundError:
            pass
        for name in data_files:
            # print(name)
            filepath = data_dir + f'{name}'
            df = pd.read_csv(filepath)
            # df = create_imu_data_deep(filepath)

            time = df.iloc[:, 0]
            quat_data = df.iloc[:, 13:17]
            sensor_data = df.iloc[:, :10]
            acc = sensor_data.to_numpy()[:, 1:4]
            ll = []
            for i in range(acc.shape[0]):
                ll.append(R.from_quat(quat_data.to_numpy()[i, [1, 2, 3, 0]]).as_matrix() @ acc[i])
            
    print(np.array(ll).mean(axis = 0))

            