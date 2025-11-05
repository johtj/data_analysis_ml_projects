import autograd.numpy as np

class scheduler:

    def __init__(self,eta: float):
        self.eta = eta

    def calculate_update(self,gradients):
        raise NotImplementedError
    
    # overwritten if needed
    def reset(self):
        pass

class constant(scheduler):

    def __init__(self, eta: float):
        super().__init__(eta)

    def calculate_update(self, gradients):
        return self.eta * gradients
    
    def reset(self):
        pass

class momentum(scheduler):

    def __init__(self, eta: float,momentum: float):
        super().__init__(eta)
        self.change = 0
        self.momentum = momentum
    
    def calculate_update(self, gradients):
        self.change = self.momentum * self.change + self.eta * gradients
        return self.change
    
    def reset(self):
        pass

class RMSprop(scheduler):

    def __init__(self, eta: float, rho: float):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def calculate_update(self, gradients):
        delta = 1e-8

        self.second = self.rho * self.second + (1-self.rho) * gradients * gradients
        return self.eta * gradients / (np.sqrt(self.second + delta)) 

    def reset(self):
        pass

class ADAM(scheduler):

    def __init__(self, eta: float,rho: float, rho2: float):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.time_step = 1  

    def calculate_update(self, gradients):
        delta = 1e-8

        self.moment = self.rho * self.moment + (1-self.rho) * gradients
        self.second = self.rho2 * self.second + (1-self.rho2) * gradients

        moment_debiased = self.moment / (1-self.rho**self.time_step)
        second_debiased = self.second / (1-self.rho2**self.time_step)

        return self.eta * moment_debiased / (np.sqrt(second_debiased + delta))
    
    def reset(self):
        self.time_step += 1
        self.moment = 0
        self.second = 0