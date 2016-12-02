import numpy as np
from pinocchio import SE3, log3, exp3, Motion


''' Base class for a trajectory '''
class RefTrajectory (object):

  def __init__ (self, name):
    self._name = name
    self._dim = 0

  @property
  def dim(self):
    return self._dim

  def __call__ (self, t):
    return (np.matrix ([]).reshape (0, 0),)*3
    
''' An se3 trajectory with constant state and zero velocity/acceleration. '''
class ConstantSE3Trajectory (object):

  def __init__ (self, name, Mref):
    self._name = name
    self._dim = 6
    self._Mref = Mref;
    self._v_ref = Motion.Zero()
    self._a_ref = Motion.Zero()

  @property
  def dim(self):
    return self._dim
    
  def setReference(self, Mref):
    self._Mref = Mref;

  def __call__ (self, t):
    return (self._Mref, self._v_ref, self._a_ref);
