import pinocchio as se3
import numpy as np
from pinocchio import SE3

#_ Reference behaviours
def exponentialDecay(kp):
    ''' 
    Returns a kv value given a kp value
    This control law imposes an exponential decay on the task
    i.e. when tracking a moving target
    '''
    kv = 2*np.sqrt(kp)
    return kv

def adaptativeGain(p_error, kmin, kmax, beta):
    '''
    Returns a kp value given:
    kmin is the minimum gain when the error approaches zero
    kmax is the maximum gain when the task is far
    beta regulates the velocity of the transition of the behaviour
    i.e. to reach a fixed target
    '''
    kp = ( (kmin - kmax)*np.exp(-beta*np.linalg.norm(p_error)) ) + kmax
    return kp

def errorInSE3(M, Mdes):        
    '''                                                                                                     
    Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference                          
    between M and Mdes, both element of SE3.    
    '''
    error = se3.log(Mdes.inverse()*M)
    return error


class Task:
    def __init__(self, robot, name = "Task"):
        self.robot = robot
        self.name = name
        self.kp = 1
        self.kv = 1
        # reference behaviours
        self.expDecay = False
        self.adaptGain = False
        self.kmin = 1
        self.kmax = 10
        self.beta = 5

class SE3Task(Task):
    def __init__(self, robot, frame_id, ref_trajectory, name = "Task"):
        Task.__init__ (self, robot, name)
        self._frame_id = frame_id
        self._ref_trajectory = ref_trajectory
        # set default value to M_ref
        self._M_ref = SE3.Identity
        # mask over the desired euclidian axis
        self._mask = (np.ones(6)).astype(bool)
        # for local to global
        self._gMl = SE3.Identity()

    def mask(self, mask):
        assert len(mask) == 6, "The mask must have 6 elemets"
        self._mask = mask.astype(bool)

    @property
    def refConfiguration (self, M_ref):
        assert isinstance(M_ref, SE3), "M_ref is not an element of class SE3"
        self._M_ref = M_ref

    @property
    def refTrajectory(self):
        return self._ref_trajectory

    def jointPostition(self):
        return robot.position(robot.q, joint_id)
    
    def framePosition(self):
        return self.robot.framePosition(self._frame_id)

    def positionError(self, t):
        oMi = self.robot.framePosition(self._frame_id)
        M_ref, v_ref, a_ref = self._ref_trajectory(t)
        p_error = errorInSE3(oMi, M_ref)
        return p_error.vector[self._mask]
    
    def velocityError(self, t):
        oMi = self.robot.framePosition(self._frame_id);
        self._gMl.rotation = oMi.rotation
        v_frame = self.robot.frameVelocity(self._frame_id);
        M_ref, v_ref, a_ref  = self._ref_trajectory(t);
        v_error = v_frame - self._gMl.actInv(v_ref)
        return v_error.vector[self._mask];
    
    def kin_value(self, t, q, local_frame = True):
        oMi = self.robot.framePosition(self._frame_id)
        v_frame = self.robot.frameVelocity(self._frame_id)

        # Get the reference trajectory   
        M_des, v_ref, a_ref  = self._ref_trajectory(t)
        
        # Transformation from local to world    
        self._gMl.rotation = oMi.rotation 
        
        #_ Task functions:
        # Compute desired velocity
        p_error = errorInSE3(oMi, M_des)
        v_error = v_frame - self._gMl.actInv(v_ref)
        
        # porportional derivative task
        if self.expDecay is True:
            self.kv = exponentialDecay(self.kp)
        if self.adaptGain is True:
            self.kp = adaptativeGain(p_error.vector, self.kmin, self.kmax, self.beta)
        v_des = - self.kp * p_error.vector  - self.kv * v_error.vector
        J= self.robot.frameJacobian(q, self._frame_id, False)

        if(local_frame==False):
            v_des[:3] = self._gMl.rotation * v_des[:3];
            v_des[3:] = self._gMl.rotation * v_des[3:];
            J[:3,:] = self._gMl.rotation * J[:3,:];
            J[3:,:] = self._gMl.rotation * J[3:,:];
        return J[self._mask,:], v_des[self._mask]


    def dyn_value(self, t, q, v, local_frame = True):
        # Get the current configuration of the link
        oMi = self.robot.framePosition(self._frame_id);
        v_frame = self.robot.frameVelocity(self._frame_id)
        
        # Get the reference trajectory
        M_ref, v_ref, a_ref  = self._ref_trajectory(t)
        
        # Transformation from local to world    
        self._gMl.rotation = oMi.rotation 
        
        #_ Taks functions:
        # Compute error acceleration desired
        p_error= errorInSE3(oMi, M_ref);
        v_error = v_frame - self._gMl.actInv(v_ref)
        drift = self.robot.frameAcceleration(self._frame_id)
        drift.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T    
        
        # porportional derivative task
        if self.expDecay is True:
            self.kv = self.exponentialDecay(self.kp)
        if self.adaptGain is True:
            self.kp = self.adaptativeGain(p_error.vector, self.kmin, self.kmax, self.beta)
        a_des = - self.kp * p_error.vector - self.kv * v_error.vector + self._gMl.actInv(a_ref).vector #sign +-
        J = self.robot.frameJacobian(q, self._frame_id, False)
    
        if(local_frame==False):
            drift = self._gMl.act(drift);
            a_des[:3] = self._gMl.rotation * a_des[:3];
            a_des[3:] = self._gMl.rotation * a_des[3:];
            J[:3,:] = self._gMl.rotation * J[:3,:];
            J[3:,:] = self._gMl.rotation * J[3:,:];
        
        return J[self._mask,:], drift.vector[self._mask], a_des[self._mask]


    def jacobian(self, q, update_geometry = False):
        self.__jacobian_value = self.robot.frameJacobian(q, self._frame_id, update_geometry)
        return self.__jacobian_value[self._mask,:] 
    

# Define CoM Task
class CoMTask(Task):

    def __init__ (self, robot, ref_trajectory, name = "CoM Task"):
        assert ref_trajectory.dim == 3
        Task.__init__ (self, robot, name)
        self._ref_trajectory = ref_trajectory
        # mask over the desired euclidian axis
        self._mask = (np.ones(3)).astype(bool)
    
    @property
    def dim(self):
        return self._mask.sum()
    @property
    def RefTrajectory(self):
        return self._ref_trajectory

    def mask(self, mask):
        assert len(mask) == 3, "The mask must have 3 elements"
        self._mask = mask.astype(bool)

    def dyn_value(self, t, q, v, update_geometry = False):
        # Get the current CoM position, velocity and acceleration
        p_com, v_com, a_com = self.robot.com(q,v,0*v)
        # Get reference CoM trajectory
        p_ref, v_ref, a_ref = self._ref_trajectory(t)
        
        # Compute errors
        p_error = p_com - p_ref
        v_error = v_com - v_ref 
        
        drift = a_com # Coriolis acceleration
        a_des = -(self.kp * p_error + self.kv * v_error) + a_ref

        # Compute jacobian
        J = self.robot.Jcom(q)

        return J[self._mask,:], drift[self._mask], a_des[self._mask]

    def jacobian(self, q, update_geometry = True):    
        self.__jacobian_value = self.robot.Jcom(q) # TODO - add update geometry option
        return self.__jacobian_value[self._mask,:] 

''' Define Postural Task considering only the joints (and not the floating base). '''
class JointPostureTask(Task):

    def __init__ (self, robot, ref_trajectory, name = "Joint Posture Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(robot.nv-6)).astype(bool)
        # desired postural configuration
        self._ref_traj = ref_trajectory;
        # Init
        #self._jacobian = np.zeros((robot.nv-6, robot.nv))
        #self._jacobian[:,6:] = np.identity((robot.nv-6))
        self._jacobian = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])
    
    @property
    def dim(self):
        return self._mask.sum ()

    def mask(self, mask):
        assert len(mask) == self.robot.nv-6, "The mask must have {} elements".format(self.robot.nv-6)
        self._mask = mask.astype(bool)       

    def dyn_value(self, t, q, v, update_geometry = False):
        # Compute error
        (q_ref, v_ref, a_ref) = self._ref_traj(t)
        err = se3.differentiate(self.robot.model, q_ref, q)[6:]
        derr = v[6:, 0] - v_ref[self._mask]
        a_des = a_ref[self._mask] - (self.kp * err + self.kv * derr) 
        drift = 0*a_des
        return self._jacobian[self._mask,:], drift[self._mask], a_des[self._mask]




# Define Angular Momentum Task
class AngularMomentumTask(Task):
    def __init__ (self, robot, name = "Angular Momentum Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(robot.nv)).astype(bool)

    @property
    def dim(self):
        return self._mask.sum ()

    def mask(self, mask):
        assert len(mask) == 3, "The mask must have {} elements".format(3)
        self._mask = mask.astype(bool)

    def setTrajectory(self, traj):
        self._ref_traj = traj

    def error_dyn(self, t, q, v):
        g = self.robot.biais(q,0*v)
        b = self.robot.biais(q,v)
        b -= g;
        M = self.robot.mass(q)
        
        com_p = self.robot.com(q)
        cXi = SE3.Identity()
        oXi = self.robot.data.oMi[1]
        cXi.rotation = oXi.rotation
        cXi.translation = oXi.translation - com_p
        b_com = cXi.inverse().np.T * b[:6,0]
        b_angular = -b_com[3:,:]
        
        M_com = cXi.inverse().np.T * M[:6,:]
        L = M_com[3:,:] * v

        L_des, Ldot_des = self._ref_traj(t)
        L_error = L - L_des

        acc = Ldot_des - b_com[3:,:]
    
        # Compute error
        #error_value = self.__error_value
        #error_value[:6,0] = error_ff
        #error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]
    
        #print error_value
        #diag = np.matrix(self.robot.data.M.diagonal()) 
        #print diag
    
        #M = self.robot.data.M
        #P = np.diag(np.diag(M.A)) 
        #print P.shape 
        #print error_value.shape 
        #error_value_pond = np.matrix(P * error_value)
        #print b_angular[self._mask,0]
        #print L
        #L -= 10.
        #wXc  = SE3(eye(3),self.robot.position(q,1).inverse()*self.robot.com(q))
        #Jang = wXc.action.T[3:,:]*self.robot.mass(q)[:6,:]
        #b_com = wXc.action.T[3:,:]*b[:6]
        #b_angular = -0*b_com
        #bang = Jang*v
        #return L[self._mask], 0., b_angular[self._mask,0]
        return self._coeff * L_error[self._mask], 0., self._coeff * acc[self._mask,0]
        #return bang[self._mask], 0., b_angular[self._mask,0]


    def jacobian(self, q):
        self.robot.mass(q)
        com_p = self.robot.com(q)
        cXi= SE3.Identity()
        oXi = self.robot.data.oMi[1]
        cXi.rotation = oXi.rotation
        cXi.translation = oXi.translation - com_p
        M_ff = self.robot.data.M[:6,:]
        M_com = cXi.inverse().np.T * M_ff
        L_dot = M_com[3:,:]
        wXc  = SE3(eye(3),self.robot.position(q,1).inverse()*self.robot.com(q))
        Jang = wXc.action.T[3:,:]*self.robot.mass(q)[:6,:]
        return self._coeff * L_dot[self._mask,:] 
        #return Jang[self._mask,:] 