import os
import csv
import shutil

from utils.data_processing import *


for data_num in range(1, 9):
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
            # df = pd.read_csv(filepath)
            df = create_imu_data_deep(filepath)

            time = df.iloc[:, 0]
            quat_data = df.iloc[:, 13:17]
            sensor_data = df.iloc[:, :10]
            ir_A0 = df.iloc[:, 17]
            ir_A1 = df.iloc[:, 18]
            ir_A2 = df.iloc[:, 19]

            thressholdA2 = 0.21 # We says the sensor see something if the returned value is less than 20cm
            thressholdA1 = 0.22
            thressholdA0 = 0.26
            # Make sure the initial position of Rasp is captured by the IR connected to Arduino's A2 port
            for i in range(10):
                if ir_A2.iloc[i] >= thressholdA2:
                    print(f"Error in file {name} - Initial position error")
                    assert ir_A2.iloc[i] < thressholdA2
                if ir_A0.iloc[time.size - 1 - i] >= thressholdA0:
                    print(f"Error in file {name} - Final position error")
                    assert ir_A0.iloc[time.size - 1 - i] < thressholdA0
            
            # Departure instance
            i_depart = 10
            while ir_A2.iloc[i_depart] < thressholdA2:
                i_depart += 1
            i_depart -= 1
            # Initial distance
            initial_distance = ir_A2.iloc[:i_depart].mean()
            for i in range(1, 6):
                if ir_A2.iloc[i_depart + i] < thressholdA2 and abs(initial_distance - ir_A2.iloc[i_depart + i]) < 0.02:
                    print(f"Error: Review the file {name} - Data of IR A2 line {i_depart + i + 2}")
            
            # Arriving instance
            i_final = time.size - 1
            while ir_A0.iloc[i_final] < thressholdA0:
                i_final -= 1
            i_final += 1
            # Final distance
            final_distance = ir_A0.iloc[i_final:].mean()
            for i in range(1, 6):
                if ir_A0.iloc[i_final - i] < thressholdA0 and abs(final_distance - ir_A0.iloc[i_final - i]) < 0.02:
                    print(f"Error: Review the file {name} - Data of IR A0 line {i_final - i + 2}")
            
            # Middle instance
            i_mid_1 = i_depart
            while ir_A1.iloc[i_mid_1] > thressholdA1:
                i_mid_1 += 1
            i_mid_2 = i_final
            while ir_A1.iloc[i_mid_2] > thressholdA1:
                i_mid_2 -= 1
            list_idx_begin = []
            list_idx_end = []
            for i in range(i_mid_1, i_mid_2 + 1):
                if (i == i_mid_1) or ((ir_A1.iloc[i] < thressholdA1) and (ir_A1.iloc[i-1] > thressholdA1)):
                    list_idx_begin.append(i)
                if (i == i_mid_2) or ((ir_A1.iloc[i] < thressholdA1) and (ir_A1.iloc[i+1] > thressholdA1)):
                    list_idx_end.append(i)
            diff = [list_idx_end[i] - list_idx_begin[i] for i in range(len(list_idx_begin))]
            n = sum(diff[i] > 4 for i in range(len(diff)))
            if n > 1:
                print(name)
            assert n == 1
            indices = [i for i, x in enumerate(diff) if x == max(diff)] 
            print(f'{name} Beginning Midle Ending range: 1-{i_depart + 2}; {list_idx_begin[indices[0]] + 2}-{list_idx_end[indices[0]] + 2}; {i_final + 2}-f')
            
            i_mid = int((list_idx_end[indices[0]] + list_idx_begin[indices[0]])/2)

            x, y, z = position_calulation( time, i_depart, i_final, h=0.56, l=1.40, a0=initial_distance, b0=final_distance, \
                                            t0=time.iloc[i_depart], t1=time.iloc[i_mid], t2=time.iloc[i_final])

            data = pd.concat([time, x, y, z, quat_data], axis=1)
            data.to_csv(f'./data_deep/data{data_num}/gt/{name}', index=False)
            sensor_data.to_csv(f'./data_deep/data{data_num}/imu/{name}', index=False)
