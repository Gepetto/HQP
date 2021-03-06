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

def exponentialGrowth(kp):
    kv = 1./2*np.sqrt(kp)
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
        self.kp = 80
        self.kv = 2*np.sqrt(self.kp)
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
        self.__gain_matrix = np.matrix(np.eye(robot.nv))

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
        
    def setGain(self, gain_vector):
        assert gain_vector.shape == (self.robot.nv,)         
        self.__gain_matrix = np.matrix(np.diag(gain_vector))

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
        self.p_error= errorInSE3(oMi, M_ref);
        self.v_error = v_frame - self._gMl.actInv(v_ref)
        drift = self.robot.frameAcceleration(self._frame_id)
        drift.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T    
        
        # porportional derivative task
        #if self.expDecay is True:
        #    self.kv = exponentialDecay(self.kp)
        #if self.adaptGain is True:
        #    self.kp = adaptativeGain(p_error.vector, self.kmin, self.kmax, self.beta)
        a_des = self._gMl.actInv(a_ref).vector - (self.kp*self.p_error.vector + self.kv*self.v_error.vector) 
        J = self.robot.frameJacobian(q, self._frame_id, False)
        
        if(local_frame==False):
            drift = self._gMl.act(drift);
            a_des[:3] = self._gMl.rotation * a_des[:3];
            a_des[3:] = self._gMl.rotation * a_des[3:];
            J[:3,:] = self._gMl.rotation * J[:3,:];
            J[3:,:] = self._gMl.rotation * J[3:,:];

        return J[self._mask,:]*self.__gain_matrix, drift.vector[self._mask], a_des[self._mask]


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
        self.__gain_matrix = np.matrix(np.eye(robot.nv))
    
    @property
    def dim(self):
        return self._mask.sum()
    @property
    def RefTrajectory(self):
        return self._ref_trajectory

    def mask(self, mask):
        assert len(mask) == 3, "The mask must have 3 elements"
        self._mask = mask.astype(bool)

    def setGain(self, gain_vector):
        assert gain_vector.shape == (self.robot.nv,)         
        self.__gain_matrix = np.matrix(np.diag(gain_vector))

    def dyn_value(self, t, q, v, update_geometry = False):
        # Get the current CoM position, velocity and acceleration
        p_com, v_com, a_com = self.robot.com(q,v,0*v)
        # Get reference CoM trajectory
        p_ref, v_ref, a_ref = self._ref_trajectory(t)
        
        # Compute errors
        self.p_error = p_com - p_ref
        self.v_error = v_com - v_ref 
        drift = a_com # Coriolis acceleration
        # porportional derivative task
        if self.expDecay is True:
            self.kv = exponentialDecay(self.kp)
        if self.adaptGain is True:
            self.kp = adaptativeGain(self.p_error, self.kmin, self.kmax, self.beta)
        a_des = -(self.kp * self.p_error + self.kv * self.v_error) + a_ref

        # Compute jacobian
        J = self.robot.Jcom(q)*self.__gain_matrix
        
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
        self.p_error = se3.differentiate(self.robot.model, q_ref, q)[6:]
        self.v_error = v[6:, t] - v_ref[self._mask]
        a_des = a_ref[self._mask] - (self.kp * self.p_error + self.kv * self.v_error) 
        drift = 0*a_des
        return self._jacobian[self._mask,:], drift[self._mask], a_des[self._mask]


''' Define Postural Task ''' 
class PosturalTask(Task):

  def __init__ (self, robot, name = "Postural Task"):
    Task.__init__ (self, robot, name)

    # mask over the desired euclidian axis
    self._mask = (np.ones(robot.nv)).astype(bool)

    # desired postural configuration
    self.q_posture_des = zero(robot.nq)

    # Init
    self.__error_value = np.matrix(np.empty([robot.nv,1]))
    self.__jacobian_value = np.matrix(np.identity(robot.nv))
    self.__gain_vector = np.matrix(np.ones([1.,robot.nv]))
    
  @property
  def dim(self):
    return self._mask.sum ()

  def setPosture(self, q_posture):
    self.q_posture_des = np.matrix.copy(q_posture)

  def setGain(self, gain_vector):
    assert gain_vector.shape == (1, self.robot.nv) 
    self.__gain_vector = np.matrix(gain_vector)

  def getGain(self):
    return self.__gain_vector

  def mask(self, mask):
    assert len(mask) == self.robot.nv, "The mask must have {} elements".format(self.robot.nq)
    self._mask = mask.astype(bool)

  def error_dyn(self, t, q, v):
    M_ff = XYZQUATToSe3(q[:7])
    M_ff_des = XYZQUATToSe3(self.q_posture_des[:7])
    error_ff = errorInSE3(M_ff, M_ff_des).vector() 
    
    # Compute error
    error_value = self.__error_value
    error_value[:6,0] = error_ff
    error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]
 
    return error_value[self._mask], v[self._mask], 0.

  def dyn_value(self, t, q, v, update_geometry = False):
    M_ff = XYZQUATToSe3(q[:7])
    M_ff_des = XYZQUATToSe3(self.q_posture_des[:7])
    error_ff = errorInSE3(M_ff, M_ff_des).vector
    
    # Compute error
    error_value = self.__error_value
    error_value[:6,0] = error_ff
    error_value[6:,0] = q[7:,0] - self.q_posture_des[7:,0]
    
    self.J = np.diag(self.__gain_vector.A.squeeze())
    self.a_des = -(self.kp * error_value + self.kv * v)
    self.drift = 0*self.a_des
    return self.J[self._mask,:], self.drift[self._mask], self.a_des[self._mask]

  def jacobian(self, q):
    self.__jacobian_value = np.diag(self.__gain_vector.A.squeeze())
    return self.__jacobian_value[self._mask,:] 



''' Defidata.oMi[f.parent].act(ne Free Flyer Rotation Task '''
class FreeFlyerTask(Task):

    def __init__ (self, robot, ref_trajectory, name = "Joint Posture Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(6)).astype(bool)
        #(np.hstack([np.zeros([3]),np.ones([3]),np.zeros([robot.nv-6])])).astype(bool)
        # desired postural configuration
        self._ref_traj = ref_trajectory;
        # Init
        self._jacobian = np.matrix(np.eye(robot.nv))
        
    
    @property
    def dim(self):
        return self._mask.sum ()

    def mask(self, mask):
        assert len(mask) == 6, "The mask must have 6 elemets"
        self._mask = mask.astype(bool)
               

    def dyn_value(self, t, q, v, update_geometry = False):
        # Compute error
        (q_ref, v_ref, a_ref) = self._ref_traj(t)
        M_ff = se3.utils.XYZQUATToSe3(q[:7])
        M_ff_des = se3.utils.XYZQUATToSe3(q_ref[:7,t])
        self.p_error = errorInSE3(M_ff, M_ff_des).vector
        self.v_error = v[self._mask] - v_ref[self._mask]
        self.a_des = a_ref[self._mask] - (self.kp * self.p_error[self._mask] + self.kv * self.v_error)
        self.drift = 0*self.a_des
        return self._jacobian[self._mask,:], self.drift, self.a_des



''' Define Momentum Task  '''
class MomentumTask(Task):
    
    def __init__ (self, robot, ref_trajectory, name = "Momentum Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(6)).astype(bool)
        self._ref_traj = ref_trajectory
        #self.__gain_matrix = np.matrix(np.ones([robot.nv,robot.nv]))
        self.__gain_matrix = np.matrix(np.eye(robot.nv))

    @property
    def dim(self):
        return self._mask.sum ()

    def mask(self, mask):
        assert len(mask) == 6, "The mask must have {} elements".format(6)
        self._mask = mask.astype(bool)

    def setTrajectory(self, traj):
        self._ref_traj = traj
    
    def setGain(self, gain_vector):
        assert gain_vector.shape == (self.robot.nv,)         
        #self.__gain_matrix = np.matrix(np.matlib.repmat(gain,1,self.dim))
        self.__gain_matrix = np.matrix(np.diag(gain_vector))
        #self.__gain_matrix = np.matrix(np.matlib.repmat(gain,1,42))

    def _getBiais(self,q,v):
        model = self.robot.model
        data = self.robot.data
        p_com = self.robot.com(q)

        oXi = data.oMi[1]
        cXi = se3.SE3.Identity()
        cXi.rotation = oXi.rotation
        cXi.translation = oXi.translation - p_com
        #cXi.translation = p_com

        m_gravity = model.gravity.copy()
        model.gravity.setZero()
        b = se3.nle(model,data,q,v)
        model.gravity = m_gravity
        f_ff = se3.Force(b[:6])
        f_com = cXi.act(f_ff)
        return f_com.np

    def _getContribution(self,com,s):
        ''' 
        Get segment contribution to centroidal angular
        momenta and its rate of change 
        Inputs:                                                                           
        - s : segment index  
        - com : position of the CoM in world reference frame 
        '''
        # get spatial inertia and velocity
        Y = self.robot.model.inertias[s]
        V = self.robot.data.v[s]
        # centroidal momenta in body frame 
        # ihi = I Vi
        iHi = se3.Force(Y.matrix()*V.vector)
        # rate of change of centroidal momenta in body frame 
        # ih_doti = Iai + Vi x* hi  
        # TO VERIFY  
        iHDi = (self.robot.model.inertias[s]*self.robot.data.a[s]) + se3.Inertia.vxiv(Y,V)
        #iHDi.linear  
        # express at the center of mass 
        oMc = se3.SE3.Identity()
        oMc.translation = self.robot.data.oMi[1].translation - com
        # uncomment in case need to change the rotation of the reference frame 
        oMc.rotation = self.robot.data.oMi[1].rotation 
        cMi = oMc.actInv( self.robot.data.oMi[s] )
        cHi = cMi.act(iHi).np.A1
        cHDi = cMi.act(iHDi).np.A1
        return cHi, cHDi

    def _calculateHdot(self, q):
        # Individual contributions to centroidal momenta (and its rate of change)                            
        segH = []; segF=[]
        p_com = self.robot.com(q)
        for s in range(1,26):
            hc, hcd = self._getContribution(p_com,s)
            segH.append(hc)
            segF.append(hcd)
        return np.matrix(np.sum(np.array(segF).squeeze(),0)).T


    def dyn_value_h(self, t, q, v):
        (self.hg_ref, self.dhg_ref, aahg_ref) = self._ref_traj(t)
        JMom = se3.ccrba(self.robot.model, self.robot.data, q, v)
        b = self._getBiais(q,v)
        self.hg_act =  self.robot.data.hg.np.A.copy()
        self.dhg_act = self._calculateHdot(q)
        
        self.drift=b[self._mask,:]
        
        self.p_error = self.hg_act[self._mask,:] - self.hg_ref[self._mask,:]        
        self.v_error = self.dhg_act[self._mask,:] - self.dhg_ref[self._mask,:]
        #self.dh_des =  -( (self.kp * self.p_error) + (self.kv*self.v_error) )
        self.dh_des = - (self.kp * self.p_error)
        self._jacobian = JMom.copy()[self._mask,:] * self.__gain_matrix
        #self.drift = 0*self.dh_des
        return self._jacobian, self.drift, self.dh_des

    def dyn_value(self, t, q, v):
        (self.hg_ref, self.dhg_ref, aahg_ref) = self._ref_traj(t)
        JMom = se3.ccrba(self.robot.model, self.robot.data, q, v)
        b = self._getBiais(q,v)
        self.hg_act =  self.robot.data.hg.np.A.copy()
        self.dhg_act = self._calculateHdot(q)
        
        self.drift=b[self._mask,:]
        
        self.p_error = self.hg_act[self._mask,:] - self.hg_ref[self._mask,:]        
        self.v_error = self.dhg_act[self._mask,:] - self.dhg_ref[self._mask,:]
        self.dh_des =  - ((self.kp * self.p_error) + (self.kv*self.v_error) )
        self.h_des = - (self.kp * self.p_error)
        self._jacobian = JMom.copy()[self._mask,:] * self.__gain_matrix
        #self.drift = 0*self.dh_des
        return self._jacobian, self.drift, self.h_des


''' Define Fly Momentum Task '''
class FlyMomentumTask(Task):
    def __init__ (self, robot, ref_trajectory, name = "Momentum Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(6)).astype(bool)
        self._ref_traj = ref_trajectory
        #self.__gain_matrix = np.matrix(np.ones([robot.nv,robot.nv]))
        self.__gain_matrix = np.matrix(np.eye(robot.nv))
        #self.Hg = MomentumTask()
        
    @property
    def dim(self):
        return self._mask.sum ()

    def mask(self, mask):
        assert len(mask) == 6, "The mask must have {} elements".format(6)
        self._mask = mask.astype(bool)

    def setTrajectory(self, traj):
        self._ref_traj = traj
        
    def setGain(self, gain_vector):
        assert gain_vector.shape == (self.robot.nv,)         
        #self.__gain_matrix = np.matrix(np.matlib.repmat(gain,1,self.dim))
        self.__gain_matrix = np.matrix(np.diag(gain_vector))
        #self.__gain_matrix = np.matrix(np.matlib.repmat(gain,1,42))
        
    def dyn_value(self, t, q, v):
        #(hg_ref, vhg_ref, ahg_ref) = self._ref_traj(t)
        vhg_ref = np.matrix([0., 0., 0., 0., 0., 0.]).T
        model =   self.robot.model
        data =    self.robot.data 
        JMom =    se3.ccrba(model, data, q, v)
        hg_prv = data.hg.vector.copy()[self._mask,:]
        #self.p_error = data.hg.vector.copy()[self._mask,:] - vhg_ref[self._mask,:]
        #self.v_error = self.robot.fext[self._mask,:] - vhg_ref[self._mask,:]
        #self.v_error = self.robot.fext[self._mask,:] 
        #self.v_error = data.hg.vector.copy()[self._mask,:] - vhg_ref[self._mask,:]
        #self.a_des   = -self.kv*self.v_error #+model.gravity.vector[self._mask,:]

        self.a_des   = self.kv*self.robot.fext[self._mask,:]#vhg_ref #+model.gravity.vector[self._mask,:]
        #self.drift = 0 * self.a_des
        #self.a_des   = 
        #***********************
        p_com = data.com[0]
        cXi = SE3.Identity()
        oXi = self.robot.data.oMi[1]
        cXi.rotation = oXi.rotation
        cXi.translation = oXi.translation - p_com
        m_gravity = model.gravity.copy()
        model.gravity.setZero()
        b = se3.nle(model,data,q,v)
        model.gravity = m_gravity
        f_ff = se3.Force(b[:6])
        f_com = cXi.act(f_ff)
        hg_drift = f_com.angular 
        self.drift=f_com.np[self._mask,:]
        #************************
        #print JMom.copy()[self._mask,:].shape
        #print self.__gain_matrix.shape
        self._jacobian = JMom.copy()[self._mask,:] * self.__gain_matrix
        return self._jacobian, self.drift, self.a_des

''' Define Gaze Task '''
class GazeSE3Task(Task):
    def __init__(self, robot, frame_id, op_point, target, ref_trajectory, name = "Task"):
        Task.__init__ (self, robot, name)
        self._frame_id = frame_id
        self._ref_trajectory = ref_trajectory
        self._op_point = op_point
        # set default value to M_ref
        self._M_ref = SE3.Identity
        # mask over the desired euclidian axis
        self._mask = (np.ones(6)).astype(bool)
        self._target = target
        # for local to global
        self._gMl = SE3.Identity()

    def mask(self, mask):
        assert len(mask) == 6, "The mask must have 6 elemets"
        self._mask = mask.astype(bool)

    @property
    def refTrajectory(self):
        return self._ref_trajectory

    def __split(self, V):
        return (V[0], V[1], V[2])

    def get3DPointOnEyeOnHand(self,q):
        # joint in global frame
        oMi = self.robot.framePosition(self._frame_id,q);
        #vision in global frame 
        oMp = se3.SE3.Identity()
        self.oMp.translation= oMi.translation+oMi.rotation*self._op_point
        self.oMp.rotation = oMi.rotation.copy()
        pMt =oMp.inverse()*self._target#oMp.inverse().act(self._target)
        self.vision = oMp.act(pMt)
        return self.__split(pMt.translation.A1)

    def getProjection(self, X, Y, Z):
        depth = Z#np.linalg.norm(np.array([X,Y,Z]))
        x = X/depth #u0 + f * px * X / Z
        y = Y/depth #v0 + f * py * Y / Z
        return (x,y)
    
    def get3DPointVelocity(self, X,Y,Z):
        oMi = self.robot.framePosition(self._frame_id);
        self._gMl.rotation = oMi.rotation
        v_frame = self.robot.frameVelocity(self._frame_id);
        (Xdot,Ydot,Zdot) = -v_frame.linear.A1 - np.cross(v_frame.angular.A1,
                                                         np.array([X,Y,Z],dtype=np.float64))
        return (Xdot,Ydot,Zdot)
    
    def visualServoMethod(self, t, q, v):
        x_ref, v_ref, a_ref  = self._ref_trajectory(t)
        (X,Y,Z) =self.get3DPointOnEyeOnHand(q)
        (x,y) = self.getProjection(X,Z,Y)
        D = Y
        Lx = np.matrix([ [-1./D,     0.,  x/D,         x*y,   -(1+(x*x)),    y] ,
                         [    0., -1/D,   y/D,    1.+(y*y),         -x*y,   -x] ],
                       dtype=np.float)
        self.p_error = np.matrix([x,y], dtype=np.float).T - x_ref
        a_des = -self.kp * self.p_error
        drift = 0*a_des
        J = Lx*self.robot.frameJacobian(q, self._frame_id, False)
        self.v_error = self.p_error
        return J, drift, a_des
        
    def axisAngleFromRotation(self, R):
        pass
    def vectorsMethod(self, t, q, v):                
        from biomechanics.maths import rotation_matrix, rotation_from_matrix
        x_ref, v_ref, a_ref  = self._ref_trajectory(t)
        # joint in global frame
        oMi = self.robot.framePosition(self._frame_id,q);
        #vision in global frame 
        oMp = se3.SE3.Identity()
        oMp.translation= oMi.translation+oMi.rotation*self._op_point
        oMp.rotation = oMi.rotation.copy()
        pMt =oMp.inverse()*self._target
        self.vision = oMp.act(pMt)
        # ----
        (x,y,z) = pMt.translation
        r= np.linalg.norm([x,y,z])
        t = np.trace(pMt.rotation)
        x = 0#oMp.rotation[2,1] - oMp.rotation[1,2]
        y = oMp.rotation[0,2] - oMp.rotation[2,0]
        z = 0#oMp.rotation[1,0] - oMp.rotation[0,1]
        theta = np.arctan2(r,t-1)
        axis_dir = (x,y,z)
        #axis_dir = np.array( (x/d_norm, y/d_norm, z/d_norm))
        #theta = np.arccos(dnorm,)
        M_des=se3.SE3.Identity()
        M_des.rotation = np.asmatrix(rotation_matrix(axis_dir,theta))
        a_des =-self.kp * errorInSE3(self.robot.framePosition(self._frame_id,q), M_des).vector[5]
        print errorInSE3(self.robot.framePosition(self._frame_id,q), M_des).vector[5]
        J = self.robot.frameJacobian(q, self._frame_id, False)[5,:]
        v_frame = self.robot.frameVelocity(self._frame_id)
        drift = self.robot.frameAcceleration(self._frame_id)
        drift.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T    
        drift = drift.vector[5]
        return J, drift, a_des

    def dyn_value(self, t, q, v, local_frame = True):
        # Get the reference trajectory
        #self.J, self.drift, self.a_des = self.visualServoMethod(t, q, v)
        self.J, self.drift, self.a_des = self.vectorsMethod(t, q, v)
        return self.J, self.drift, self.a_des

        #pMtVec = self._target-pMt.translation
        #Vision = pMtVec.translation/np.linalg.norm(pMtVec.translation)
        
        #Eyes= se3.utils.matrixToRpy(oMp.rotation)
        #p_error = np.cross(pMt_vector.A1,vec.A1)
        #p_error = np.cross(oMop.rotation[:,0].A1,opMv_vector.A1)
        
        #self._gMl.rotation *opMv_vector.A
        #self.vec= vec.A1
        #
        #a_des = self.kp*np.matrix(p_error).T
        
        # Interaction matrix
        

        
        #v_frame = self.robot.frameVelocity(self._frame_id)
        #drift = self.robot.frameAcceleration(self._frame_id)
        #drift.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T    
        #self.drift =drift.vector[3:6]
        
        #J = self.robot.frameJacobian(q, self._frame_id, False)[3:6,:]
        
        #self.p_error = p_error
        #print im2d
        
        
        




''' Define Kinetic Energy Task '''
class KineticEnergyTask(Task):
    def __init__ (self, robot, name = "Kinetic Energy Task"):
        Task.__init__ (self, robot, name)
        # mask over the desired euclidian axis
        self._mask = (np.ones(robot.nv)).astype(bool)

    def dyn_value(self, t, q, v):
        #(ke_ref, vke_ref, ake_ref) = self._ref_traj(t)
        ke_ref = self.robot.hg.vector
        v_ref = np.matrix([0., 0., 0., 0., 0., 0.]).T
        Jf = self.robot.frameJacobian(q, self._frame_id, False)
        J = 0.5*(self.robot.data.M*np.transpose(Jf)*Jf)
        ke_act = 0.5*self.robot.data.mass[0]*self.robot.vcom^2
        self.err = ke_act[self._mask,:] - ke_ref[self._mask,:]
        self.a_des = -self.kp * self.err 
        self.drift = 0*self.a_des
        self._jacobian = J.copy()[self._mask,:] #* self.__gain_matrix
        return self._jacobian, self.drift, self.a_des


''' Define Angular Momentum Task  '''
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

    def dyn_value(self, t, q, v):
        g = self.robot.biais(q,0*v)
        b = self.robot.biais(q,v)
        b -= g;
        M = self.robot.data.mass[0]#(q)
        
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
        self.a_des = acc
        self.drift = 0*self.a_des
        self._jacobian = self.jacobian(q)

        return self._jacobian[self._mask,:], self.drift[self._mask], self.a_des[self._mask]

        #return self._coeff * L_error[self._mask], 0., self._coeff * acc[self._mask,0]
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
        #print L000000000
        #L -= 10.
        #wXc  = SE3(eye(3),self.robot.position(q,1).inverse()*self.robot.com(q))
        #Jang = wXc.action.T[3:,:]*self.robot.mass(q)[:6,:]
        #b_com = wXc.action.T[3:,:]*b[:6]
        #b_angular = -0*b_com
        #bang = Jang*v
        #return L[self._mask], 0., b_angular[self._mask,0]
        #return self._coeff * L_error[self._mask], 0., self._coeff * acc[self._mask,0]
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
