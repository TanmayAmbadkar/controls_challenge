from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    
    return np.random.uniform(low = -2, high = 2, size = (1, ))[0]
