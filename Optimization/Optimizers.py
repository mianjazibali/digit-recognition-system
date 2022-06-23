import numpy as np

class Sgd:
    def __init__(self, learing_rate):
        self.learing_rate = learing_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - (self.learing_rate * gradient_tensor)

class SgdWithMomentum:
    def __init__(self, learing_rate, momentum_rate):
        self.learing_rate = learing_rate
        self.momentum_rate = momentum_rate
        self.value = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.value = (self.momentum_rate * self.value) - (self.learing_rate * gradient_tensor)       
        return weight_tensor + self.value   

class Adam:
    def __init__(self, learing_rate, mu, rho):
        self.learing_rate = learing_rate
        self.mu = mu
        self.rho = rho

        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1

        self.v = (self.mu * self.v) + ((1 - self.mu) * gradient_tensor)
        self.r = (self.rho * self.r) + ((1 - self.rho) * gradient_tensor * gradient_tensor)

        # Bias Corrections
        _v = self.v / (1 - np.power(self.mu, self.k))
        _r = self.r / (1 - np.power(self.rho, self.k))

        return weight_tensor - self.learing_rate * _v / (np.sqrt(_r) + np.finfo(float).eps)
