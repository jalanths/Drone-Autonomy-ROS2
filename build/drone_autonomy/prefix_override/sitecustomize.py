import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jalanth/Downloads/ROBO/drone_ws/install/drone_autonomy'
