'''
	Link to library BNO055: https://github.com/adafruit/Adafruit_Python_BNO055.git
'''
import csv
import numpy as np
import argparse
import time

from utils.data_acq import *
from Adafruit_BNO055 import BNO055


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_num', type=int, help='Data folder')
	parser.add_argument('--f_output', type=int, help='First output name')
	args = parser.parse_args()

	bno = BNO055.BNO055(serial_port='/dev/ttyAMA0')
	bno.begin()#mode=BNO055_MODE_ACCONLY)

	com_all_data = create_command(BNO055_ACCEL_DATA_X_LSB_ADDR, 32)
	
	#write_to_register(bno, BNO055_PAGE_ID_ADDR, 0x01, 'Page ID')
	#write_to_register(bno, BNO055_ACC_CONFIG, 0b00011101, 'Acc Config')
	#write_to_register(bno, BNO055_MAG_CONFIG, 0b00000000, 'Mag Config')
	#write_to_register(bno, BNO055_PAGE_ID_ADDR, 0x00, 'Page ID')

	input("Step 1: Calibrate IMU sensor. Press any key to continue...")
	calibration = False
	while not calibration:
		sys, gyro, accel, mag = bno.get_calibration_status()
		print('Sys_cal={0} Gyro_cal={1} Accel_cal={2} Mag_cal={3}'.format(sys, gyro, accel, mag))
		if sys == 3 & gyro == 3 & accel == 3 & mag == 3:
			os.system('clear')
			print('Calibration completed !')
			calibration = True

	os.system('clear')
	input("Step 2: Put the IMU in 6 different stable positions. Press any key to continue...")
	filename = str(args.f_output) + '_6_fix_positions'
	data_num = args.data_num
	f, writer = init_data_file(filename, data_num)
	time.sleep(0.5)
	for i in range(6):
		input(f"Put your IMU in the position number {i+1} then press any key to continue... You have 4s")
		time_start = time.time()
		print("Writing data...")
		while time.time() - time_start < 4:
			all_data = read_raw_data(bno, com_all_data, length=32)
			writer.writerow([time.time(), *all_data])
	f.close()
	
	os.system('clear')
	input("Step 3: Calibrate your Tango phone. Press any key to continue...")

	filename = args.f_output
	data_num = args.data_num
	os.system('clear')
	input("Step 4: Press Enter and start the tango phone at the same time to continue...")
	f, writer = init_data_file(filename, data_num)
	time.sleep(0.5)
	
	os.system('clear')
	print("Step 5: You need to put the tango phone on the IMU and make a rotation around Z axis. You will have 5sec")
	input("Ready? Press Enter to continue...")
	s2_time = time.time()
	print("Writing data...")
	while time.time() - s2_time < 5:
		all_data = read_raw_data(bno, com_all_data, length=32)
		writer.writerow([time.time(), *all_data])

	time.sleep(0.5)
	os.system('clear')
	print("Step 6: Now you can make your measurement. Make sure the relative positions of your 2 devices doesn't change during the measurement")
	input("Ready? Press Enter to continue...")
	
	all_data = read_raw_data(bno, com_all_data, length=32)
	print(all_data)
	while True:
		try:
			# Read all data at once and write to scv file
			all_data = read_raw_data(bno, com_all_data, length=32)
			writer.writerow([time.time(), *all_data])
		except KeyboardInterrupt:
			while True:
				os.system('clear')
				q = input("\nCTRL-C was pressed what would you like to do:\n\
							\t(1) Remake the measurement\n\
							\t(2) Start a new measurement\n\
							\t(3) Check calibration\n\
							\t(4) Quit\n")
				if q == "3":
					for _ in range(10):
						sys, gyro, accel, mag = bno.get_calibration_status()
						print('Sys_cal={0} Gyro_cal={1} Accel_cal={2} Mag_cal={3}'.format(sys, gyro, accel, mag))
					time.sleep(2)
				if q == "1" or q == "2" or q == "4":
					break
			if q == "1":
				f.close()
				print(f'Rewriting file {filename}...')
				f, writer = init_data_file(filename, data_num)
			elif q == "2":
				f.close()
				filename += 1
				print(f'Start writing file {filename}...')
				f, writer = init_data_file(filename, data_num)
			elif q == "4":
				f.close()
				exit()


if __name__ == '__main__':
    main()
