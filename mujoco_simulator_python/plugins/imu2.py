"""第二套 IMU，复用本地采样计算。"""

from .imu import Imu


class Imu2(Imu):
    default_prefix = "imu2"
