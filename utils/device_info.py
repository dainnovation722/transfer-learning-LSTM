from tensorflow.python.client import device_lib
with open('utils/device_info.txt','w') as f:
    f.write('\n')
    f.write(str(device_lib.list_local_devices()))
    f.write('\n')