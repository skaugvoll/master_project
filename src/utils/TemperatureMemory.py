import numpy as np

class TemperatureMemory:
    def __init__(self, memory_length_seconds, window_length_in_seconds):
        self.memory_length = memory_length_seconds  # seconds
        self.memory_cost = window_length_in_seconds
        self.memory_used = 0  # seconds
        self.memory = []

    def add_to_memory(self, feature):
        if not np.ceil(self.memory_used + self.memory_cost) > self.memory_length:
            self.memory.append(feature)
            self.memory_used += self.memory_cost
        else:  # if we try to remember more than allowed, removed first memory
            temp = self.memory[1:]
            temp.append(feature)
            self.memory = temp

    def get_memory(self):
        return np.array(self.memory)


    def get_num_memories(self):
        return self.get_memory().shape

    def get_memory_length(self):
        return self.memory_used
