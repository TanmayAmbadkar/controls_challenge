from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):

      
      steer = -7.80690249e-01*target_lataccel + -7.41714685e-05*current_lataccel+  3.79966677e-02*state.v_ego+  5.49765231e+00*state.a_ego + 2.04737239e-01*state.roll_lataccel + 0.0018295017607205602
      
      return steer
