import copy
import torch
from operator import *


class ExponentialMovingAverage():
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()

    def __call__(self, x):
        return self.model(x)
    
    def update(self, model):
        # assure the same dictionary of parameters
        assert eq(self.model.state_dict().keys(), model.state_dict().keys()) == 1
        
        param_student = model.state_dict()
        param_old = self.model.state_dict()
        param_teacher = {}

        with torch.no_grad():
            for key in param_old:
                param_teacher[key] = \
                self.decay*param_old[key] + (1-self.decay)*param_student[key]
            self.model.load_state_dict(param_teacher)

    def save(self, path):
        torch.save(self.model, path)

