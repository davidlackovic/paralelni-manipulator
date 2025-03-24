import serial
import time
import numpy as np
from finished.funkcije_v3 import izracun_kotov

serial_port = 'COM8'

#Dejanski parametri
b = 0.071014 # m
p = 0.223723 # m
l_1 = 0.16198 # m
l_2 = 0.23481 # m

ser = serial.Serial(serial_port, 115200, timeout=1)

ser.write(bytes("", 'utf-8'))
time.sleep(2)

ser.write(bytes('G90\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('M203 X2500 Y2500 Z2500\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('M201 X200 Y200 Z200\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('M92 X100 Y100 Z100\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('G92 X0 Y0 Z0\n', 'utf-8'))
time.sleep(0.01)
ser.write(bytes('G1 X0 Y0 Z0\n', 'utf-8'))
time.sleep(0.01)

X = Y = h = None

while True:
    line = input("Format: X[v stopinjah] Y[v stopinjah] Z[v metrih] F[mm/s]:   ")
    if line.startswith('S'):
        break
    parts = line.split()
    f = 700
    for part in parts:
        if part.startswith('X'):
            X = float(part[1:])
        elif part.startswith('Y'):
            Y = float(part[1:])
        elif part.startswith('Z'):
            h = float(part[1:])
        elif part.startswith('F'):
            f = int(part[1:])
        


    angles = np.array(izracun_kotov(b, p, l_1, l_2, h, X, Y))

    angles = -angles*4/1.8*16/100
    
    output = f'G1 X{angles[0]:.5f} Y{angles[1]:.5f} Z{angles[2]:.5f} F{f}\n'
    print(output)
    ser.write(bytes(output, 'utf-8'))

ser.close()
    





