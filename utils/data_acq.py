import logging
import sys
import time
import os
import csv


BNO055_PAGE_ID_ADDR = 0x07

# Adresses of utile registers on Page 0
BNO055_OPR_MODE_ADDR = 0x3D               
BNO055_ACCEL_DATA_X_LSB_ADDR = 0x08       
BNO055_GYRO_DATA_X_LSB_ADDR = 0x14        
BNO055_MAG_DATA_X_LSB_ADDR = 0x0E         
BNO055_EULER_H_LSB_ADDR = 0x1A            
BNO055_QUATERNION_DATA_W_LSB_ADDR = 0x20  

# Adresses of utile registers on Page 1
BNO055_ACC_CONFIG_ADDR = 0x08             
BNO055_GYR_CONFIG_ADDR = 0x0A             
BNO055_MAG_CONFIG_ADDR = 0x09             

# Value of different modes of operation
BNO055_MODE_ACCONLY = 0x01
BNO066_MODE_AMG = 0x07

def write_to_register(bno, address, value, id_text):
	bno._config_mode()
	val = bno._read_bytes(address, length=1)
	bno._write_byte(address, value, ack=True)
	bno._operation_mode()
	print('{0}: Address {1} changed from {2:08b} to {3:08b}.'.format(id_text, hex(address), val[0], value))
	time.sleep(0.3)

def create_command(address, length):
	com = bytearray(4)
	com[0] = 0xAA  # Start byte
	com[1] = 0x01  # Read
	com[2] = address & 0xFF
	com[3] = length & 0xFF
	return com

def read_raw_data(bno, command, length=6):
	resp = bno._serial_send(command)
	resp = bytearray(bno._serial.read(length))
	return resp

def init_data_file(filename, data_num):
	f = open(f'./raw_data/data{data_num}/{filename}.csv', 'w', newline='')
	writer = csv.writer(f)
	writer.writerow(['time', 'acc raw', '', '', '', '', '', 'mag raw', '', '',  '', '', '', 'gyr raw', '', '', \
			'', '', '', 'orientation fusion', '', '', '', '', '',  'quaternion fusion', '', '', '', '', '', \
			'', '', 'IR on A0', 'IR on A1', 'IR on A2'])
	return f, writer