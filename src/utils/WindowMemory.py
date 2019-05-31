
import numpy as np

class WindowMemory():
    def __init__(self):
        self.num_windows = 1
        self.last_target = None
        self.last_start = None
        self.last_end = None
        self.avg_conf = 0.0
        self.divisor = 1
        self.sensorConfiguration = None

    def update_num_windows(self):
        self.num_windows += 1

    def update_last_target(self, target):
        self.last_target = target

    def update_last_start(self, time):
        self.last_start = time

    def update_last_end(self, time):
        self.last_end = time

    def update_avg_conf_nominator(self, conf):
        self.avg_conf += conf

    def update_avg_conf_divisor(self):
        self.divisor += 1

    def update_sensor_configuration(self, sensorConfiguration):
        self.sensorConfiguration = sensorConfiguration

    def reset_num_windows(self):
        self.num_windows = 0

    def reset_last_target(self):
        self.last_target = None

    def reset_last_end(self):
        self.last_end = None

    def reset_avg_conf(self):
        self.avg_conf = 0.0

    def reset_divisor(self):
        self.divisor = 1

    def reset_sensor_configuration(self):
        self.sensorConfiguration = None

    def get_num_windows(self):
        return self.num_windows

    def get_last_target(self):
        return self.last_target

    def get_last_start(self):
        return self.last_start

    def get_last_end(self):
        return self.last_end

    def get_avg_conf(self):
        return np.divide(self.avg_conf, self.divisor)

    def get_sensor_configuration(self):
        return self.sensorConfiguration

    def check_targets(self, target):
        return self.last_target == target
