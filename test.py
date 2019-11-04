import serial
#port="COM4"
#bluetooth=serial.Serial(port,9600)
port="COM3"
bluetooth=serial.Serial(port,9600)
while(1):
	
	print("connected")
	pred_classes=1234567
	bluetooth.flushInput()
	bluetooth.write(str(pred_classes).encode())
	#input_data=bluetooth.readline()
	#print(input_data.decode())
