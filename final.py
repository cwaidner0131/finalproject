import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import cartopy.crs as ccrs
from scipy.linalg import expm
import time
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#Helper functions
def rotate(val,angle): #rotation matrix
    # value 0 corresponds to first column or x and so on
    if val==0:
        R=np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
        ])
    elif(val==1):
        R=np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif(val==2):
        R=np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
        ])
    return R

def convert_to_rv(oe,mu): #converts orbital elements to r and v
    #intaializations
    r=[0,0,0]
    v=[0,0,0]
    
    a=oe[0]
    e=oe[1]
    i=oe[2]*math.pi/180
    omega=oe[3]*math.pi/180
    OMEGA=oe[4]*math.pi/180
    nu=oe[5]*math.pi/180
    p=a*(1-e**2)
    ri=p/(1+e*math.cos(nu))
    h=math.sqrt(mu*a*p)
    # equations
################
    #PQW frame
    r=np.array([[ri*math.cos((nu))],
                [ri*math.sin(nu)],
                [0]])
    v=math.sqrt(mu/p)*np.array([[-math.sin(nu)],
                                [(e+math.cos(nu))],
                                [0]])
    #313 rotation
    Rw=rotate(2,omega)
    Ri=rotate(0,i)
    RO=rotate(2,OMEGA)

    
    Rtot=np.transpose(Rw@Ri@RO)
    r_vec=Rtot@r
    v_vec=Rtot@v
    return r_vec,v_vec

def plot_orbit(y,title,xaxis=None,yaxis=None,zaxis=None,color=None): #plots orbit, simple 3d plot after propogation
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot3D(y.y[0],y.y[1],y.y[2],color)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)  
    ax.set_zlabel(zaxis)
    plt.show()

def propogate(r,v,tspan): #propogates orbit  using scipy
    y0=np.concatenate([r.flatten(),v.flatten()])
    t=np.linspace(0,tspan,1000)
    y=scipy.integrate.solve_ivp(fun_def1,(0,tspan),y0,method='DOP853',t_eval=t)
    return y

def fun_def1(t,y): #numerical integration function
    #function format passed to scipy
    #initializations
    mu=3.986004418*math.pow(10,5)
    r=y[0:3]
    v=y[3:6]
    r_norm=np.linalg.norm(r)
    mu=int(mu)
    a1=-mu*r[0]/r_norm**3
    a2=-mu*r[1]/r_norm**3
    a3=-mu*r[2]/r_norm**3
    a=([a1,a2,a3])
    A=np.concatenate([v,a])
    return A

def calculate_A(x,y,z,mu): #calculates A matrix
    r=np.array([x,y,z])
    r_norm=np.linalg.norm(r)
    mu=3.986004418*math.pow(10,5)
    a1=mu*(2*(x**2)-y**2-z**2)/r_norm**5
    a2=mu*(2*(y**2)-z**2-x**2)/r_norm**5
    a3=mu*(2*(z**2)-x**2-y**2)/r_norm**5

    b1=3*mu*x*y/r_norm**5
    b2=3*mu*y*x/r_norm**5
    b3=3*mu*x*z/r_norm**5
    c1=3*mu*x*z/r_norm**5
    c2=3*mu*y*z/r_norm**5
    c3=3*mu*(z*y)/r_norm**5

    A=np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[a1,b1,c1,0,0,0],[b2,a2,c2,0,0,0],[b3,c3,a3,0,0,0]])
    
    return A

def calculate_H(x_vec, index, RSite0, RSite1, RSite2):  
    """
    Calculates the Jacobian (H matrix) for a measurement model.

    Parameters:
    - x_vec: The current state vector, a 6x1 array containing position and velocity.
    - index: Integer, the index of the ground station (0, 1, or 2).
    - RSite0, RSite1, RSite2: 3D position vectors of the ground stations.

    Returns:
    - H: The Jacobian matrix (2x6) representing partial derivatives of rho and rho_dot w.r.t. state variables.
    """
    r = x_vec[:3]  # Extract position vector
    v = x_vec[3:6]  # Extract velocity vector

    # Earth's rotation rate (rad/s)
    w = 7.2921159e-5  

    # Select observation site based on the index
    if index == 0:
        r_obs = np.array(RSite0)
    elif index == 1:
        r_obs = np.array(RSite1)
    elif index == 2:
        r_obs = np.array(RSite2)
    else:
        raise ValueError(f"Invalid ground station index: {index}")

    # Compute observer velocity assuming Earth's rotation
    r_dot_obs = np.cross([0, 0, w], r_obs)  # Velocity due to Earth's rotation

    # Relative position and distance
    r_obv_to_spacecraft = r - r_obs
    rho = np.linalg.norm(r_obv_to_spacecraft)

    # Partial derivatives
    H_pos = r_obv_to_spacecraft / rho  # Partial of rho w.r.t. position

    # Intermediate term for rho_dot
    relative_velocity = v - r_dot_obs
    g = np.dot(r_obv_to_spacecraft, relative_velocity) / rho
    H_vel = relative_velocity / rho - r_obv_to_spacecraft * g / (rho**2)

    # Construct the H matrix
    H = np.vstack([
        np.hstack([H_pos, np.zeros(3)]),  # Row for rho
        np.hstack([H_vel, H_pos])        # Row for rho_dot
    ])

    return H

def calculate_hmeas(x_vec,dt,index, RSite0, RSite1, RSite2):  
    """
    Calculates the measurement vector h for a given state vector and ground station index.

    Parameters:
    - x_vec: The current state vector, a 6x1 array containing position and velocity.
    - index: Integer, the index of the ground station (0, 1, or 2).
    - RSite0, RSite1, RSite2: 3D position vectors of the ground stations.

    Returns:
    - h: The measurement vector (2x1) containing rho and rho_dot.
    """
    r = x_vec[:3]  # Extract position vector
    v = x_vec[3:6]  # Extract velocity vector

    # Earth's rotation rate (rad/s)
    w = 7.2921159e-5  

    
    # Select observation site based on the index
    if index == 0:
        r_obs = np.array(RSite0)
        lat = 35.297
        lon = -116.914
        RSitedot=np.cross([0,0,w],r_obs)
    elif index == 1:
        r_obs = np.array(RSite1)
        lat = 40.4311
        lon = -4.248
    elif index == 2:
        r_obs = np.array(RSite2)
        lat = -35.4023
        lon = 148.9813
    else:
        raise ValueError(f"Invalid ground station index: {index}")
    # Compute observer velocity 
    RSitedot=np.cross([0,0,w],r_obs)
    # Convert ECI to ECEF
    RSiteECEF,RSitedot_ECEF = convert_frame_ECI_to_ECEF(r_obs, RSitedot, dt, we_earth)
    r_ecef,rdot_ecef=convert_frame_ECI_to_ECEF(r, v, dt, we_earth)
    # Get long,lat
    phi, lambda1 = convert_frame_ECEF_to_AnglesOnly(r_ecef)
    # Convert ECEF to Topocentric
    range_vec,range_rate_vec=ECEF_to_Topo(phi*math.pi/180,lambda1*math.pi/180,RSiteECEF,RSitedot_ECEF,r_ecef,rdot_ecef)
    range=np.linalg.norm(range_vec)
    range_rate=np.dot(range_vec,range_rate_vec)/range

    return np.hstack([range, range_rate])

def ECEF_to_Topo(phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF):  
    # Compute relative position and velocity in ECEF
    delta_r = r_ecef - RSiteECEF
    delta_v = rdot_ecef - RSitedot_ECEF
    # Rotation matrices
    R1 = rotate(2, -(90 + lambda1))  # Rotate about Z-axis by longitude
    R2 = rotate(0, -(90 - phi))      # Rotate about X-axis by latitude
    Rtot = np.transpose(R1 @ R2)     # Combined transformation to topocentric frame
    # Transform position and velocity to topocentric frame
    range_vec = Rtot @ delta_r
    range_rate_vec = Rtot @ delta_v
    return range_vec, range_rate_vec

def convert_frame_ECEF_to_AnglesOnly(ECEF_r): #converts ECEF to Topocentric frame
    lambda1 = math.atan2(ECEF_r[1] , ECEF_r[0])  # use atan2 to get the correct quadrant, NOTE DONT USE atan
    norm_r = math.sqrt(ECEF_r[0] ** 2 + ECEF_r[1] ** 2 + ECEF_r[2] ** 2)
    phi = math.asin(ECEF_r[2] / norm_r)  # Corrected from y.y[2] to y[2]
    return phi, lambda1

def convert_frame_ECEF_to_ECI(r,v,tstep,rotation_rate): #converts ECI to ECEF frame
    delta_g=rotation_rate*tstep#+delta_go # initial angle zero
    R3=rotate(2,-delta_g)
    r=R3@r
    v=R3@v
    return r,v

def convert_frame_ECI_to_ECEF(r,v,tstep,rotation_rate): #converts ECI to ECEF frame
    delta_g=rotation_rate*tstep#+delta_go # initial angle zero
    R3=rotate(2,delta_g) 
    r=R3@r
    v=R3@v
    return r,v

def ECEF_vec_from_angles(phi,lambda1): #converts phi and lambda to ECEF unit vector
    """
    Converts latitude (phi) and longitude (lambda1) in degrees to a unit ECEF vector.
    Note: just helps visualize the ground stations on the 3D plot.
    Parameters:
    - phi: Latitude in degrees.
    - lambda1: Longitude in degrees.
    
    Returns:
    - r: ECEF unit vector as a numpy array.
    """
    r=np.zeros(3)
    phi=phi*math.pi/180
    lambda1=lambda1*math.pi/180
    r[0]=math.cos(phi)*math.cos(lambda1) # x
    r[1]=math.cos(phi)*math.sin(lambda1)# y
    r[2]=math.sin(phi) # z
    return r

def _3Dmodel_(data1,data2,GS_0,GS_1,GS_2): #3D model of earth and ground stations
    r_earth=6378.137 #radius of earth
    fig = plt.figure()
    GS_0_vec=r_earth*ECEF_vec_from_angles(GS_0[0],GS_0[1])
    GS_1_vec=r_earth*ECEF_vec_from_angles(GS_1[0],GS_1[1])
    GS_2_vec=r_earth*ECEF_vec_from_angles(GS_2[0],GS_2[1])    
    #
    # Generate a sphere for Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plotting the Earth and ground stations
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    ax.plot_surface(x, y, z, color='b', alpha=0.5, edgecolor='k')

    # Plot ground stations
    ax.scatter(*GS_0_vec, color='r', label="GS 0", s=200)  # Red dot for GS 1
    ax.scatter(*GS_1_vec, color='g', label="GS 1", s=200)  # Green dot for GS 2
    ax.scatter(*GS_2_vec, color='y', label="GS 2", s=200)  # Yellow dot for GS 3

    # Labels and legend
    ax.set_title("3D Earth with Ground Stations")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()

    # Equal aspect ratio for 3D plot
    max_radius = r_earth
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])
    ax.set_box_aspect([1, 1, 1])

    plt.show()
    return None

def XLSXwriter(data1): #writes to xlsx file so I can see the data
    # Create a DataFrame
    columns = ["Time (t)", "Observer Index", "Range (rho)", "Range Rate (rho_dot)"]
    df = pd.DataFrame(data1, columns=columns)
    # Save to Excel file
    output_file = r"C:\Users\cwaid\output.xlsx"
    df.to_excel(output_file, index=False)

def state_derivative(t, X):
    A = calculate_A(X[0], X[1], X[2], mu)  # Update A based on current state
    return (A @ X).flatten()  # Ensure it returns a 1D array

def stm_derivative(t, stm_flattened, A):
    # Reshape the flattened STM back to its original shape
    stm = stm_flattened.reshape((6, 6))
    # Compute the derivative of the STM
    stm_dot = A @ stm
    # Return the derivative as a flattened array
    return stm_dot.flatten()

def convert_frame_ECI_to_Topo(r,v,RSite,RSite_dot): #converts ECI to Topocentric frame# 
    #r,v in ECI
    #RSite in ECI
    #Convert ECI to ECEF
    r_ecef,rdot_ecef=convert_frame_ECI_to_ECEF(r,v)
    RSiteECEF,RSitedot_ECEF=convert_frame_ECI_to_ECEF(RSite,RSite_dot)  
    #Convert ECEF to Topocentric
    phi,lambda1=convert_frame_ECEF_to_AnglesOnly(r_ecef)
    range_vec,range_rate_vec=ECEF_to_Topo(phi,lambda1,RSiteECEF,RSitedot_ECEF,r_ecef,rdot_ecef)
    return range_vec,range_rate_vec

def matrix_frame_ECI_to_Topo(X,size,RSite,RSite_dot): #converts ECI to Topocentric frame#
    
    range=[]
    range_rate=[]
    for i in range(0,size):
        r=X[i][:3]
        v=X[i][3:]
        rho,rho_dot=convert_frame_ECI_to_Topo(r,v,RSite,RSite_dot)
        range.append(rho)
        range_rate.append(rho_dot)
    

    return rho,rho_dot
    
def kalman_easy(data, Xo, Po, mu): #EKF implementation for easy measurements
    
    extract_data = data
    data2=np.load("C:/Users/cwaid/Downloads/Project-Measurements-Easy.npy")
    # finding covariance matrix- based on the noise of rho,rho_dot
    
    dt=data2[1][0]-data2[0][0]

  

    #Initialize ground station positions
    RSite_0 = np.array([35.297, -116.914])  # Ground station 1 latitude and longitude
    RSite_1 = np.array([40.4311, -4.248])  # Ground station 2 latitude and longitude
    RSite_2 = np.array([-35.4023, 148.9813])  # Ground station 3 latitude and longitude
    we_earth = 7.2921159e-5  # Earth's rotation rate (rad/s)
    #Convert to ECEF
    RSite_0 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_0[0], RSite_0[1]))
    RSite_1 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_1[0], RSite_1[1]))
    RSite_2 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_2[0], RSite_2[1]))
    # Convert to ECI
    RSite_0 = convert_frame_ECEF_to_ECI(RSite_0, [0, 0, 0], dt, we_earth)
    RSite_1 = convert_frame_ECEF_to_ECI(RSite_1, [0, 0, 0], dt, we_earth)
    RSite_2 = convert_frame_ECEF_to_ECI(RSite_2, [0, 0, 0], dt, we_earth)
    # Extract the position vectors
    RSite_0 = RSite_0[0]
    RSite_1 = RSite_1[0]
    RSite_2 = RSite_2[0]
    #Givens , initializations
    W=(1e-1)*np.eye(3)
    aeta=np.array([[0, 0, 0]
          ,[0, 0, 0]
          ,[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]])
    upperright=aeta@W@aeta.T
    dt=data2[1][0]-data2[0][0]
    RSite_0=convert_frame_ECEF_to_ECI(RSite_0,[0,0,0],dt,we_earth)
    RSite_1=convert_frame_ECEF_to_ECI(RSite_1,[0,0,0],dt,we_earth)
    RSite_2=convert_frame_ECEF_to_ECI(RSite_2,[0,0,0],dt,we_earth)
    
    RSite_0=RSite_0[0]
    RSite_1=RSite_1[0]
    RSite_2=RSite_2[0]
    X=Xo
    X=np.array([X[0][0],X[0][1],X[0][2],X[1][0],X[1][1],X[1][2]])
    X = Xo.flatten()  
    x_arr_stochastic=[]
    x_arr_stochastic.append(X)
    P_arr_stochastic=[]
    xk=X
    Pk_1=Po
    P_arr_stochastic.append(Pk_1)
    H=calculate_H(xk,0,RSite_0,RSite_1,RSite_2)
    dt_prev=data2[1][0]-data2[0][0]
    # EKF implementation of Easy measurements
    surprise=[]
    for i in range(0, 2287-1):
        #print("Iteration: ",i)
        # Extract data for the current iteration
        dt=data2[i+1][0]-data2[i][0] #current time step
        index_num = int(data2[i][1])  # Extract ground station index
        Y = data2[i][:][2:4].reshape(2, 1) # Extract rho and rho_dot
        Y = Y.flatten() 
        #Finding R matrix(assuming not constant)
        V = np.array([[1e-1, 0], [0, 1e-8]])
        R=V
        if dt>dt_prev:
            Q_dt=dt**2
        else:
            Q_dt=1
        A=calculate_A(xk[0],xk[1],xk[2],mu) #dynamics matrix

        #Finding Q and F (no gap in data)
        upperright=aeta@W@aeta.T
        lowerleft=np.zeros((6,6))
        upperleft=-1*A
        lowerright=A.T
        upper=np.hstack((upperleft,upperright))
        lower=np.hstack((lowerleft,lowerright))
        z=np.vstack((upper,lower))
        ez=expm(z)
        upperright=ez[0:6,6:12]
        lowerright=ez[6:,6:]
        F=lowerright.T
        Q=F@upperright
        Q=Q*Q_dt*np.eye(6)
    
        #Prediction
        xk_1=F@xk
        Pk_1=F@Pk_1@F.T+Q
        I=np.eye(Pk_1.shape[0])
        #Correction
        H=calculate_H(xk_1,index_num,RSite_0,RSite_1,RSite_2) # should 2x6 [rho,rho_dot]->jacobian
        h=calculate_hmeas(xk_1,dt,index_num,RSite_0,RSite_1,RSite_2)
        

        K_k=Pk_1@H.T@np.linalg.pinv(H@Pk_1@H.T+R)  
        xk=xk_1+K_k@(Y-h)  
        Pk_1=(I-K_k@H)@Pk_1@((I-K_k@H).T)+K_k@R@K_k.T 

        diff=Y-h
        surprise.append(diff)

        #print('h: ', h)
        #print("Surprise: ",surprise)
        
        # Store the results in lists
        x_arr_stochastic.append(xk)
        P_arr_stochastic.append(Pk_1)
      
        # Update the ground station positions
        '''RSite_0=np.cross([0,0,we_earth],RSite_0)+RSite_0
        RSite_1=np.cross([0,0,we_earth],RSite_1)+RSite_1
        RSite_2=np.cross([0,0,we_earth],RSite_2)+RSite_2'''

        #deugging step
        '''if i>=10:
            break'''
    print("Final estimates x,y,z,x_dot,y_dot,z_dot")
    print(x_arr_stochastic[-1])
    print("Final Covariance Matrix")
    print(Pk_1)
    rho_arr_stochastic = []
    y = np.arange(0, 2287)  # Use np.arange instead of range to create a numpy array
    x_arr_stochastic = np.array(x_arr_stochastic)
    #P_arr_stochastic = np.array(P_arr_stochastic)
    rho_arr_stochastic = np.array(rho_arr_stochastic)

    return x_arr_stochastic, P_arr_stochastic,surprise

def kalman_hard(data, Xo, Po, mu): #EKF implementation for hard measurements
    extract_data = data
    data2=np.load("C:/Users/cwaid/Downloads/Project-Measurements-Hard.npy")
    # finding covariance matrix- based on the noise of rho,rho_dot
    
    dt=data2[1][0]-data2[0][0]

  

    #Initialize ground station positions
    RSite_0 = np.array([35.297, -116.914])  # Ground station 1 latitude and longitude
    RSite_1 = np.array([40.4311, -4.248])  # Ground station 2 latitude and longitude
    RSite_2 = np.array([-35.4023, 148.9813])  # Ground station 3 latitude and longitude
    we_earth = 7.2921159e-5  # Earth's rotation rate (rad/s)
    #Convert to ECEF
    RSite_0 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_0[0], RSite_0[1]))
    RSite_1 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_1[0], RSite_1[1]))
    RSite_2 = 6378.137 * np.array(ECEF_vec_from_angles(RSite_2[0], RSite_2[1]))
    # Convert to ECI
    RSite_0 = convert_frame_ECEF_to_ECI(RSite_0, [0, 0, 0], dt, we_earth)
    RSite_1 = convert_frame_ECEF_to_ECI(RSite_1, [0, 0, 0], dt, we_earth)
    RSite_2 = convert_frame_ECEF_to_ECI(RSite_2, [0, 0, 0], dt, we_earth)
    # Extract the position vectors
    RSite_0 = RSite_0[0]
    RSite_1 = RSite_1[0]
    RSite_2 = RSite_2[0]
    #Givens , initializations
    W=(1*10**-1)*np.eye(3)
    aeta=np.array([[0, 0, 0]
          ,[0, 0, 0]
          ,[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]])
    upperright=aeta@W@aeta.T
    dt=data2[1][0]-data2[0][0]
    RSite_0=convert_frame_ECEF_to_ECI(RSite_0,[0,0,0],dt,we_earth)
    RSite_1=convert_frame_ECEF_to_ECI(RSite_1,[0,0,0],dt,we_earth)
    RSite_2=convert_frame_ECEF_to_ECI(RSite_2,[0,0,0],dt,we_earth)
    
    RSite_0=RSite_0[0]
    RSite_1=RSite_1[0]
    RSite_2=RSite_2[0]
    X=Xo
    X=np.array([X[0][0],X[0][1],X[0][2],X[1][0],X[1][1],X[1][2]])
    X = Xo.flatten()  
    x_arr_stochastic=[]
    x_arr_stochastic.append(X)
    P_arr_stochastic=[]
    xk=X
    Pk_1=Po
    P_arr_stochastic.append(Pk_1)
    H=calculate_H(xk,0,RSite_0,RSite_1,RSite_2)
    dt_prev=data2[1][0]-data2[0][0]
    # EKF implementation of Easy measurements
    Q_dt=1
    surprise=[]
    for i in range(0, len(data2)-1):
        #print("Iteration: ",i)
        # Extract data for the current iteration
        
        dt=data2[i+1][0]-data2[i][0] #current time step
        if dt>dt_prev:
            Q_dt=dt**2
        else:
            Q_dt=1
        index_num = int(data2[i][1])  # Extract ground station index
        Y = data2[i][:][2:4].reshape(2, 1) # Extract rho and rho_dot
        Y = Y.flatten() 
        #Finding R matrix(assuming not constant)
        V = np.array([[1e-1, 0], [0, 1e-8]])
        R=V

        A=calculate_A(xk[0],xk[1],xk[2],mu) #dynamics matrix

        #Finding Q and F (no gap in data)
        upperright=aeta@W@aeta.T
        lowerleft=np.zeros((6,6))
        upperleft=-1*A
        lowerright=A.T
        upper=np.hstack((upperleft,upperright))
        lower=np.hstack((lowerleft,lowerright))
        z=np.vstack((upper,lower))
        ez=expm(z)
        upperright=ez[0:6,6:12]
        lowerright=ez[6:,6:]
        F=lowerright.T
        Q=F@upperright
        Q=Q*np.eye(6)*Q_dt
    
        #Prediction
        xk_1=F@xk
        Pk_1=F@Pk_1@F.T+Q
        I=np.eye(Pk_1.shape[0])
        #Correction
        H=calculate_H(xk_1,index_num,RSite_0,RSite_1,RSite_2) # should 2x6 [rho,rho_dot]->jacobian
        h=calculate_hmeas(xk_1,dt,index_num,RSite_0,RSite_1,RSite_2)
        

        K_k=Pk_1@H.T@np.linalg.pinv(H@Pk_1@H.T+R)  
        xk=xk_1+K_k@(Y-h)  
        Pk_1=(I-K_k@H)@Pk_1@((I-K_k@H).T)+K_k@R@K_k.T 

        diff=Y-h
        surprise.append(diff)

        #print('h: ', h)
        #print("Surprise: ",surprise)
        
        # Store the results in lists
        x_arr_stochastic.append(xk)
        P_arr_stochastic.append(Pk_1)
      
        # Update the ground station positions
        RSite_0=np.cross([0,0,we_earth],RSite_0)+RSite_0
        RSite_1=np.cross([0,0,we_earth],RSite_1)+RSite_1
        RSite_2=np.cross([0,0,we_earth],RSite_2)+RSite_2

        #deugging step
        '''if i>=10:
            break'''
    print("Final estimates x,y,z,x_dot,y_dot,z_dot")
    print(x_arr_stochastic[-1])
    print("Final Covariance Matrix")
    print(Pk_1)
    rho_arr_stochastic = []
    y = np.arange(0, len(data2)-1)  # Use np.arange instead of range to create a numpy array
    x_arr_stochastic = np.array(x_arr_stochastic)
    #P_arr_stochastic = np.array(P_arr_stochastic)
    rho_arr_stochastic = np.array(rho_arr_stochastic)

    return x_arr_stochastic, P_arr_stochastic,surprise
if __name__ == "__main__":
    # Character orbit
    oe=[7000,0.2,45*math.pi/180,0,270*math.pi/180,78.75*math.pi/180]
    [r,v]=convert_to_rv(oe,3.986004418*(10**5))
    y=propogate(r,v,54000)
    #plot_orbit(y,'Character Orbit','x','y','z','r')

    # Data laoding
    data1 = np.load("C:/Users/cwaid/Downloads/Project-Measurements-Easy.npy")
    data2 = np.load("C:/Users/cwaid/Downloads/Project-Measurements-Hard.npy")
    tspan=54000
    
    #plot data
    rho1 = data1[:, 2]
    rho_dot1 = data1[:, 3]

    rho2 = data2[:, 2]
    rho_dot2 = data2[:, 3]

    timevar = data1[:, 0]
    timevar2 = data2[:, 0]

    GS_0=np.array([35.297, -116.914]) #ground station 1 latitude and longitude
    GS_1=np.array([40.4311, -4.248]) #ground station 2 latitude and longitude
    GS_2=np.array([-35.4023, 148.9813]) #ground station 3 latitude and longitude
    #3D model of earth and ground stations (this is for me to verify the ground stations are in the right place)
    #_3Dmodel_(data1,data2,GS_0,GS_1,GS_2)

    # Extract measurements for each observer
    GS_measurements = {0: [], 1: [], 2: []}
    for row in data2:
        observer_index = int(row[1]) 
        GS_measurements[observer_index].append(row)
    GS_measurements = {key: np.array(value) for key, value in GS_measurements.items()}
    #print(GS_measurements[0])
    #outputfile=XLSXwriter(data2)
    re=6378.137#radius of earth
    w=7.2921159e-5 #rad/s
    dt=data1[1][0]-data1[0][0]
    RSite_0=re*np.array(ECEF_vec_from_angles(35.297, -116.914))
    RSite_1=re*np.array(ECEF_vec_from_angles(40.4311, -4.248))
    RSite_2=re*np.array(ECEF_vec_from_angles(-35.4023, 148.9813))
    we_earth=7.2921159e-5 #rad/s
    W=(1*10**-3)*np.eye(3)
    RSite_0=convert_frame_ECEF_to_ECI(RSite_0,[0,0,0],dt,we_earth)
    RSite_1=convert_frame_ECEF_to_ECI(RSite_1,[0,0,0],dt,we_earth)
    RSite_2=convert_frame_ECEF_to_ECI(RSite_2,[0,0,0],dt,we_earth)
   

    RSite_0=RSite_0[0]
    RSite_1=RSite_1[0]
    RSite_2=RSite_2[0]
    #given constants
    noise_rho=1 #m^2
    noise_rho_dot=1e-5 #cm^2/s^2
    Xo = np.array([r,v]) # starting position and velocity
    aeta=np.array([[0, 0, 0]
          ,[0, 0, 0]
          ,[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]])
    
    Po = np.block([[50*np.eye(3),np.zeros((3,3))],
                   [np.zeros((3,3)),np.eye(3)]])#Covariance
    mu=3.986004418*(10**5) # km^3/s^2
    X=Xo
    Pk_1=[]
    Pk_1.append(Po)
    print("Initial estimates x,y,z,x_dot,y_dot,z_dot")  
    print(X)
    X=np.array([X[0][0],X[0][1],X[0][2],X[1][0],X[1][1],X[1][2]])
    X = Xo.flatten()  # Flatten the array to 1D if necessary
    X = X.reshape(6, 1)  # Ensure it is a 6x1 column vector
    X_arr=[]
    X_arr.append(X)
    # Discrete system without stochasticity
    range_var=[]
    range_rate_var=[]
    covariance_var=[]
    for i in range(0,2287-1): #

        dt=data1[i+1][0]-data1[i][0]
        #A at current time
        A=calculate_A(X[0][0],X[1][0],X[2][0],mu)
        x_dot=A@X #define x_dot
        #STM_at_Current=expm(A*dt) #assumes dSTM  is constant
        STM_at_Current=A@np.eye(6) #Linearized STM 
        STM_dot=STM_at_Current #STM_dot which will be integrated

        #integrate z dot
        x=scipy.integrate.solve_ivp(fun=state_derivative,t_span=(0,dt),y0=X.flatten(),method='DOP853')
        result_stm = scipy.integrate.solve_ivp(
            fun=stm_derivative,                 # Function for STM derivative
            t_span=(0, dt),                     # Time interval
            y0=STM_at_Current.flatten(),        # Flattened initial STM
            method='DOP853',
            args=(A,)                           # Pass A as an additional argument
        )
        # Extract the final STM at the end of the integration

        STM_dot = result_stm.y[:, -1].reshape((6, 6))

        Pk_1=STM_dot@Po@STM_dot.T
        covariance_var.append(Pk_1)
        X = x.y[:, -1].reshape(6, 1)
        F=STM_dot
        dX=F@x_dot #dX
        X=X+dX*dt #integrate propogation and get new state
        X_arr.append(X)
        index_num = int(data1[i][1])  # Extract ground station index
        # Check ground station
        if index_num == 0:
            r_obs = np.array(RSite_0)
            RSitedot=np.cross([0,0,w],r_obs)
        elif index_num == 1:
            r_obs = np.array(RSite_1)
            
        elif index_num == 2:
            r_obs = np.array(RSite_2)
        else:
            raise ValueError(f"Invalid ground station index: {index_num}")
        RSitedot=np.cross([0,0,w],r_obs)
        #Change frames
        r_ecef,rdot_ecef=convert_frame_ECI_to_ECEF(X[:3],X[3:],dt,w)
        phi, lambda1 = convert_frame_ECEF_to_AnglesOnly(r_ecef)
        r_obs_ecef,rdot_obs_ecef=convert_frame_ECI_to_ECEF(r_obs,RSitedot,dt,w)
        range_vec,range_rate_vec=ECEF_to_Topo(phi,lambda1,r_obs_ecef,rdot_ecef,r_ecef,rdot_ecef)

        range_var.append(np.linalg.norm(range_vec))
        range_rate_var.append(range_rate_vec)
    #Plot range values real vs. estimated
    plt.figure()
    plt.scatter(np.arange(0,len(range_var)),range_var,s=.1,c='b')
    plt.scatter(np.arange(0,2287),rho1,s=.1,c='r')
    plt.title("Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.show()

    # Find stds and plot
    std_pos=[]
    var_x=[]
    var_y=[]
    var_z=[]
    var_x_dot=[]
    var_y_dot=[]
    var_z_dot=[]
    for k in range(0,2287-1):
        P=covariance_var[k]
        var_x.append(P[0,0])
        var_y.append(P[1,1])
        var_z.append(P[2,2])
        var_x_dot.append(P[3,3])
        var_y_dot.append(P[4,4])
        var_z_dot.append(P[5,5])

        avg_var = np.mean([covariance_var[k][i,i] for i in range(3)])

        std_pos.append(math.sqrt(avg_var))
    std_pos=np.array(std_pos)

     # Convert to numpy arrays
    svar_x = np.array(var_x)
    svar_y = np.array(var_y)
    svar_z = np.array(var_z)
    svar_vx = np.array(var_x_dot)
    svar_vy = np.array(var_y_dot)
    svar_vz = np.array(var_z_dot)
    s_std_x=np.zeros(len(svar_x))
    s_std_y=np.zeros(len(svar_y))
    s_std_z=np.zeros(len(svar_z))
    s_std_vx=np.zeros(len(svar_vx))
    s_std_vy=np.zeros(len(svar_vy))
    s_std_vz=np.zeros(len(svar_vz))

    for i in range(0,len(var_x)):
        s_std_x[i]=np.sqrt(svar_x[i])
        s_std_y[i]=np.sqrt(svar_y[i])
        s_std_z[i]=np.sqrt(svar_z[i])
        s_std_vx[i]=np.sqrt(svar_vx[i])
        s_std_vy[i]=np.sqrt(svar_vy[i])
        s_std_vz[i]=np.sqrt(svar_vz[i])

    ts=np.arange(0,2287-1)
     ############################ Easy measurements ########################################

    # finding covariance matrix- based on the noise of rho,rho_dot
    Po = np.block([[(1e-1)*np.eye(3),np.zeros((3,3))],
                   [np.zeros((3,3)),(1e-8)*np.eye(3)]])#Covariance matrix
    Xo = np.array([r,v]) # starting position and velocity
    #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF
    
 
    rho_arr_actual = data1[:,2]  # Extract the third column (rho)

    [x_arr_easy, P_arr_easy,surprise] = kalman_easy(data1, Xo, Po, mu)
    rho_arr_easy=[]
    rho_dot_easy=[]
    #Convert x_arr_easy to topocentric frame
    for i in range(0,len(x_arr_easy)):
        dt=data1[i][0]-data1[i-1][0]
        w=7.2921159e-5
        r=x_arr_easy[i][:3]
        v=x_arr_easy[i][3:]
        [r_eci,v_eci]=convert_frame_ECEF_to_ECI(r,v,dt,we_earth)
        [phi, lambda1]=convert_frame_ECEF_to_AnglesOnly(r_eci)
        index_num = int(data2[i][1])  # Extract ground station index
        if index_num == 0:
            r_obs = np.array(RSite_0)
            
            RSitedot=np.cross([0,0,w],r_obs)
        elif index_num == 1:
            r_obs = np.array(RSite_1)
            
        elif index_num == 2:
            r_obs = np.array(RSite_2)
            
        else:
            raise ValueError(f"Invalid ground station index: {index_num}")
        RSitedot=np.cross([0,0,w],r_obs)
        r_obs,RSitedot=convert_frame_ECI_to_ECEF(r_obs,RSitedot,dt,we_earth)
        #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF

        [rho,rho_dot]=ECEF_to_Topo(phi,lambda1,r_eci,v_eci,r_obs,RSitedot)
        rho_arr_easy.append(np.linalg.norm(rho))
        rho_dot_easy.append(np.linalg.norm(rho_dot))

        
    rho_arr_easy=np.array(rho_arr_easy)


    residuals=rho_arr_easy-rho_arr_actual
    print(rho_arr_actual[-1])
    print(rho_arr_easy[-1])
    y = np.arange(len(rho_arr_easy))  # Ensure y matches the time steps
    # Plot the data
    plt.figure()
    plt.scatter(y, residuals,s=.1,label="Range (rho-km)-calculated")
    plt.scatter(y,rho_arr_actual,s=.1,label="Range (rho-km)-actual")
    plt.title("Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.show()


    # Rho dot
    rho_dot_arr_actual = data1[:,3]  # Extract the fourth column (rho_dot)
    rho_dot_arr_easy = np.array(rho_dot_easy)  # Convert to numpy array
    # Plot the data
    plt.figure()

    plt.scatter(y, rho_dot_arr_easy,s=.1,label="Range Rate (rho_dot)-calculated")
    plt.scatter(y, rho_dot_arr_actual,s=.1,label="Range Rate (rho_dot)-actual")
    plt.title("Range Rate (rho_dot) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range Rate (rho_dot)")
    plt.show()


    avg_pos=[]
    avg_std=[]
    for k in range(0,len(x_arr_easy)):
        avg_pos.append(np.mean(x_arr_easy[k][:3]))
        avg_var=np.mean([P_arr_easy[k][i,i]for i in range(3)])
        avg_std.append(np.sqrt(avg_var))
    
    
    avg_std=np.array(avg_std)
    avg_pos=np.array(avg_pos)
    #plot
    lower_bound = avg_pos - 3 * avg_std
    upper_bound = avg_pos + 3 * avg_std

    threshold = 10000  # Example: filter out values above 10000

    # Create a boolean mask where values are below the threshold
    mask = rho_arr_easy <= threshold

    # Apply the mask to filter rho_arr_hard and corresponding time steps
    filtered_rho_arr_easy= rho_arr_easy[mask]
    filtered_time_steps = np.arange(len(rho_arr_easy))[mask]
    filtered_rho_arr_actual = rho_arr_actual[mask]

    rho_arr_actual = data1[:,2]  # Extract the third column (rho)
    # Plot the filtered data
    plt.figure()
    y=np.arange(0,len(filtered_rho_arr_easy))
    plt.scatter(filtered_time_steps, filtered_rho_arr_easy, s=0.5, label="Filtered Range (rho)")
    plt.scatter(filtered_time_steps,filtered_rho_arr_actual,s=.1,label="Range (rho)-actual")
    plt.title("Filtered Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.legend()
    plt.grid(True)
    plt.show()


    threshold = 10000  # Threshold to filter values

    # Create a boolean mask where values are below the threshold
    mask = rho_arr_easy <= threshold

    # Find indices for unfiltered values (those above the threshold)
    unfiltered_indices = np.where(~mask)[0]  # Indices where mask is False

    # Initialize lists to store the variances
    unfiltered_avg_pos = []
    unfiltered_avg_var = []

    # Loop through the unfiltered indices and calculate variances
    for idx in unfiltered_indices:
        # Calculate mean position (x, y, z) for the unfiltered step
        unfiltered_avg_pos.append(np.mean(x_arr_easy[idx][:3]))
        # Calculate average variance for position (x, y, z)
        avg_var = np.mean([P_arr_easy[idx][i, i] for i in range(3)])
        unfiltered_avg_var.append(avg_var)

    # Convert to numpy arrays for further analysis or plotting
    unfiltered_avg_pos = np.array(unfiltered_avg_pos)
    unfiltered_avg_var = np.array(unfiltered_avg_var)

    # Plot the variances
    mask_std=avg_std>75
    print('avg_var',avg_std)
    print('mask_std',mask_std.shape)
    unfiltered_avg_pos=avg_pos[mask_std]
    unfiltered_avg_var=avg_std[mask_std]
    unfiltered_indices=np.where(~mask_std)[0]
    print("Unfiltered indices: ", unfiltered_indices)

    # Initialize arrays to store variances
    var_x = []
    var_y = []
    var_z = []
    var_vx = []
    var_vy = []
    var_vz = []

    # Extract variances from the diagonal of each covariance matrix
    for idx in unfiltered_indices:
        P = P_arr_easy[idx]  # Extract the covariance matrix
        var_x.append(P[0, 0])
        var_y.append(P[1, 1])
        var_z.append(P[2, 2])
        var_vx.append(P[3, 3])
        var_vy.append(P[4, 4])
        var_vz.append(P[5, 5])
        

    # Convert to numpy arrays
    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_z = np.array(var_z)
    var_vx = np.array(var_vx)
    var_vy = np.array(var_vy)
    var_vz = np.array(var_vz)
    std_x=np.zeros(len(var_x))
    std_y=np.zeros(len(var_y))
    std_z=np.zeros(len(var_z))
    std_vx=np.zeros(len(var_vx))
    std_vy=np.zeros(len(var_vy))
    std_vz=np.zeros(len(var_vz))

    for i in range(0,len(var_x)):
        std_x[i]=np.sqrt(var_x[i])*1000
        std_y[i]=np.sqrt(var_y[i])*1000
        std_z[i]=np.sqrt(var_z[i])*1000
        std_vx[i]=np.sqrt(var_vx[i])*1000
        std_vy[i]=np.sqrt(var_vy[i])*1000
        std_vz[i]=np.sqrt(var_vz[i])*1000


    ############################ Plots measurements ########################################
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Standard Deviation of Position and Velocity")
    time_steps = np.arange(0, len(std_x))
    
    # Plot the variances in subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("±3σ Uncertainty, and Position Estimates")

    axs[0, 0].fill_between(time_steps, 3*std_x, -3*std_x, color="blue", alpha=0.2, label="±3σ (Uncertainty) of Stoachstic")
    axs[0,0].fill_between(ts, 3*s_std_x*1000, -3*s_std_x*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[0, 0].plot(time_steps,var_x,label="x")
    axs[0, 0].set_title("Variance of x")
    axs[0, 0].set_xlabel("Time Steps")
    axs[0, 0].set_ylabel("Variance")
    axs[0, 0].grid(True)

    axs[0, 1].fill_between(time_steps, 3*std_y, -3*std_y, color="blue", alpha=0.2, label="±3σ (Uncertainty) of Stoachstic")
    axs[0,1].fill_between(ts, 3*s_std_y*1000, -3*s_std_y*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[0, 1].plot(time_steps,var_y,label="y")
    axs[0, 1].set_title("Variance of y")
    axs[0, 1].set_xlabel("Time Steps")
    axs[0, 1].set_ylabel("Variance")
    axs[0, 1].grid(True)

    axs[0, 2].fill_between(time_steps, 3*std_z, -3*std_z, color="blue", alpha=0.2, label="±3σ (Uncertainty) of stoachstic")
    axs[0,2].fill_between(ts, 3*s_std_z*1000, -3*s_std_z*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[0, 2].plot(time_steps,var_z,label="z")
    axs[0, 2].set_title("Variance of z")
    axs[0, 2].set_xlabel("Time Steps")
    axs[0, 2].set_ylabel("Variance")
    axs[0, 2].grid(True)

    axs[1, 0].fill_between(time_steps, 3*std_vx, -3*std_vx, color="blue", alpha=0.2, label="±3σ (Uncertainty) of Sotchastic")
    axs[1,0].fill_between(ts, 3*s_std_vx*1000, -3*s_std_vx*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[1, 0].plot(time_steps,var_vx,label="vx")
    axs[1, 0].set_title("Variance of v_x")
    axs[1, 0].set_xlabel("Time Steps")
    axs[1, 0].set_ylabel("Variance")
    axs[1, 0].grid(True)

    axs[1, 1].fill_between(time_steps, 3*std_vy, -3*std_vy, color="blue", alpha=0.2, label="±3σ (Uncertainty) of Purely Stochastic")
    axs[1, 1].fill_between(ts, 3*s_std_vy*1000, -3*s_std_vy*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[1, 1].plot(time_steps,var_vy,label="vy")
    axs[1, 1].set_title("Variance of v_y")
    axs[1, 1].set_xlabel("Time Steps")
    axs[1, 1].set_ylabel("Variance")
    axs[1, 1].grid(True)

    axs[1, 2].fill_between(time_steps, 3*std_vz, -3*std_vz, color="blue", alpha=0.2, label="±3σ (Uncertainty) of Stochastic")
    axs[1,2].fill_between(ts, 3*s_std_vz*1000, -3*s_std_vz*1000, color="red", alpha=0.2, label="±3σ (Uncertainty) of Non Stochastic")
    axs[1, 2].plot(time_steps,var_vz,label="vz")
    axs[1, 2].set_title("Variance of v_z")
    axs[1, 2].set_xlabel("Time Steps")
    axs[1, 2].set_ylabel("Variance")
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


    ############################ Easy measurements ########################################

    # finding covariance matrix- based on the noise of rho,rho_dot
    Po = np.block([[(1e-1)*np.eye(3),np.zeros((3,3))],
                   [np.zeros((3,3)),(1e-8)*np.eye(3)]])#Covariance matrix
    Xo = np.array([r,v]) # starting position and velocity
    #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF
    
 
    rho_arr_actual = data1[:,2]  # Extract the third column (rho)

    [x_arr_easy, P_arr_easy,surprise] = kalman_easy(data1, Xo, Po, mu)
    rho_arr_easy=[]
    rho_dot_easy=[]
    #Convert x_arr_easy to topocentric frame
    for i in range(0,len(x_arr_easy)):
        dt=data1[i][0]-data1[i-1][0]
        w=7.2921159e-5
        r=x_arr_easy[i][:3]
        v=x_arr_easy[i][3:]
        [r_eci,v_eci]=convert_frame_ECEF_to_ECI(r,v,dt,we_earth)
        [phi, lambda1]=convert_frame_ECEF_to_AnglesOnly(r_eci)
        index_num = int(data2[i][1])  # Extract ground station index
        if index_num == 0:
            r_obs = np.array(RSite_0)
            
            RSitedot=np.cross([0,0,w],r_obs)
        elif index_num == 1:
            r_obs = np.array(RSite_1)
            
        elif index_num == 2:
            r_obs = np.array(RSite_2)
            
        else:
            raise ValueError(f"Invalid ground station index: {index_num}")
        RSitedot=np.cross([0,0,w],r_obs)
        r_obs,RSitedot=convert_frame_ECI_to_ECEF(r_obs,RSitedot,dt,we_earth)
        #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF

        [rho,rho_dot]=ECEF_to_Topo(phi,lambda1,r_eci,v_eci,r_obs,RSitedot)
        rho_arr_easy.append(np.linalg.norm(rho))
        rho_dot_easy.append(np.linalg.norm(rho_dot))

        
    rho_arr_easy=np.array(rho_arr_easy)


    residuals=rho_arr_easy-rho_arr_actual
    print(rho_arr_actual[-1])
    print(rho_arr_easy[-1])
    y = np.arange(len(rho_arr_easy))  # Ensure y matches the time steps
    # Plot the data
    plt.figure()
    plt.scatter(y, residuals,s=.1,label="Range (rho-km)-calculated")
    plt.scatter(y,rho_arr_actual,s=.1,label="Range (rho-km)-actual")
    plt.title("Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.show()


    # Rho dot
    rho_dot_arr_actual = data1[:,3]  # Extract the fourth column (rho_dot)
    rho_dot_arr_easy = np.array(rho_dot_easy)  # Convert to numpy array
    # Plot the data
    plt.figure()

    plt.scatter(y, rho_dot_arr_easy,s=.1,label="Range Rate (rho_dot)-calculated")
    plt.scatter(y, rho_dot_arr_actual,s=.1,label="Range Rate (rho_dot)-actual")
    plt.title("Range Rate (rho_dot) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range Rate (rho_dot)")
    plt.show()


    avg_pos=[]
    avg_std=[]
    for k in range(0,len(x_arr_easy)):
        avg_pos.append(np.mean(x_arr_easy[k][:3]))
        avg_var=np.mean([P_arr_easy[k][i,i]for i in range(3)])
        avg_std.append(np.sqrt(avg_var))
    
    
    avg_std=np.array(avg_std)
    avg_pos=np.array(avg_pos)
    #plot
    lower_bound = avg_pos - 3 * avg_std
    upper_bound = avg_pos + 3 * avg_std

    ############################ Hard measurements ########################################

    # finding covariance matrix- based on the noise of rho,rho_dot
    Po = np.block([[(1e-1)*np.eye(3),np.zeros((3,3))],
                   [np.zeros((3,3)),(1e-5)*np.eye(3)]])#Covariance matrix
    Xo = np.array([r,v]) # starting position and velocity
    #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF
    
 
    rho_arr_actual = data2[:,2]  # Extract the third column (rho)

    [x_arr_hard, P_arr_hard,surprise] = kalman_hard(data2, Xo, Po, mu)
    rho_arr_hard=[]
    rho_dot_hard=[]
    #Convert x_arr_easy to topocentric frame
    for i in range(0,len(x_arr_hard)):
        dt=data2[i][0]-data2[i-1][0]
        w=7.2921159e-5
        r=x_arr_hard[i][:3]
        v=x_arr_hard[i][3:]
        [r_eci,v_eci]=convert_frame_ECEF_to_ECI(r,v,dt,we_earth)
        [phi, lambda1]=convert_frame_ECEF_to_AnglesOnly(r_eci)
        index_num = int(data2[i][1])  # Extract ground station index
        if index_num == 0:
            r_obs = np.array(RSite_0)
            
            RSitedot=np.cross([0,0,w],r_obs)
        elif index_num == 1:
            r_obs = np.array(RSite_1)
            
        elif index_num == 2:
            r_obs = np.array(RSite_2)
            
        else:
            raise ValueError(f"Invalid ground station index: {index_num}")
        RSitedot=np.cross([0,0,w],r_obs)
        r_obs,RSitedot=convert_frame_ECI_to_ECEF(r_obs,RSitedot,dt,we_earth)
        #phi, lambda1, r_ecef, rdot_ecef, RSiteECEF, RSitedot_ECEF

        [rho,rho_dot]=ECEF_to_Topo(phi,lambda1,r_eci,v_eci,r_obs,RSitedot)
        rho_arr_hard.append(np.linalg.norm(rho))
        rho_dot_hard.append(np.linalg.norm(rho_dot))

        
    rho_arr_hard=np.array(rho_arr_hard)


    residuals=rho_arr_actual-rho_arr_hard
    print(rho_arr_actual[-1])
    print(rho_arr_hard[-1])
    y = np.arange(len(rho_arr_hard))  # Ensure y matches the time steps
    # Plot the data
    plt.figure()
    plt.scatter(y,rho_arr_hard,s=.1,label="Range (rho-km)-calculated")
    plt.scatter(y,rho_arr_actual,s=.1,label="Range (rho-km)-actual")
    plt.title("Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.show()


    # Rho dot
    rho_dot_arr_actual = data2[:,3]  # Extract the fourth column (rho_dot)
    rho_dot_arr_hard = np.array(rho_dot_hard)  # Convert to numpy array
    # Plot the data
    plt.figure()

    plt.scatter(y, rho_dot_arr_hard,s=.1,label="Range Rate (rho_dot)-calculated")
    plt.scatter(y, rho_dot_arr_actual,s=.1,label="Range Rate (rho_dot)-actual")
    plt.title("Range Rate (rho_dot) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range Rate (rho_dot)")
    plt.show()
    

    avg_pos=[]
    avg_std=[]
    x_var=[]
    y_var=[]
    z_var=[]
    vx_var=[]
    vy_var=[]
    vz_var=[]

    for k in range(0,len(x_arr_hard)):
        avg_pos.append(np.mean(x_arr_hard[k][:3]))
        x_var.append(x_arr_hard[k][0])
        y_var.append(x_arr_hard[k][1])
        z_var.append(x_arr_hard[k][2])
        vx_var.append(x_arr_hard[k][3])
        vy_var.append(x_arr_hard[k][4])
        vz_var.append(x_arr_hard[k][5])

        avg_var=np.mean([P_arr_hard[k][i,i]for i in range(3)])
        avg_std.append(np.sqrt(avg_var))
    
    
    avg_std=np.array(avg_std)
    avg_pos=np.array(avg_pos)
    #plot
    lower_bound = avg_pos - 3 * avg_std
    upper_bound = avg_pos + 3 * avg_std

    t=np.arange(0,len(x_arr_hard))
    plt.figure()
    plt.plot(t,avg_pos/1000,label="x")
    plt.fill_between(t,avg_pos/1000 - 3 * avg_std,avg_pos/1000+ 3 * avg_std,color="blue",alpha=0.2,label="±3σ (Uncertainty)",)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position Estimates and Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Variance of the position estimates by itself
    plt.figure()
    plt.fill_between(t,avg_pos/1000 - 3 * avg_std,avg_pos/1000+ 3 * avg_std,color="blue",alpha=0.2,label="±3σ (Uncertainty)",)
    plt.xlabel("Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Uncertainty")
    plt.show()

    
    '''rho_arr_actual = data1[:,2]  # Extract the third column (rho)
    print('rho_arr_actual',rho_arr_actual[-1])
    print('rho_arr_easy',rho_arr_easy[-1])
    print("rho_arr_actual shape:", rho_arr_actual.shape)
    rho_arr_easy = np.array(rho_arr_easy) 

    print("rho_arr_easy shape:", rho_arr_easy.shape)
    print("rho_arr_actual shape:", rho_arr_actual.shape)
    error=rho_arr_actual-rho_arr_easy'''
    # Plotting
    error=surprise[0]*(1/1000)
    y = np.arange(len(surprise))  # Ensure y matches the time steps
    plt.figure()
    plt.plot(y, surprise, label="Residuals - Mean")
    plt.axhline(0, color='red', linestyle='--', label="Mean Line (y=0)")
    plt.title("Difference of Residuals from Their Mean")
    plt.xlabel("Time")
    plt.ylabel("Residuals - Mean")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure()
    t=np.arange(0,len(x_arr_hard))
    plt.fill_between(t,-3*avg_std,3*avg_std,color="blue",alpha=0.2,label="±3σ (Uncertainty)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (km)")
    plt.title("Position Uncertainty")
    plt.show()
    

    threshold = 10000  # Example: filter out values above 10000

    # Create a boolean mask where values are below the threshold
    mask = rho_arr_hard <= threshold

    # Apply the mask to filter rho_arr_hard and corresponding time steps
    filtered_rho_arr_hard = rho_arr_hard[mask]
    filtered_time_steps = np.arange(len(rho_arr_hard))[mask]
    filtered_rho_arr_actual = rho_arr_actual[mask]

    rho_arr_actual = data2[:,2]  # Extract the third column (rho)
    # Plot the filtered data
    plt.figure()
    y=np.arange(0,len(filtered_rho_arr_hard))
    plt.scatter(filtered_time_steps, filtered_rho_arr_hard, s=0.5, label="Filtered Range (rho)")
    plt.scatter(filtered_time_steps,filtered_rho_arr_actual,s=.1,label="Range (rho)-actual")
    plt.title("Filtered Range (rho) vs. Time")
    plt.xlabel("Time")
    plt.ylabel("Range (rho)")
    plt.legend()
    plt.grid(True)
    plt.show()

    avg_std=[]
    avg_pos=[]
    for k in range(0,len(x_arr_hard)):
        avg_pos.append(np.mean(x_arr_hard[k][:3]))
        avg_var=np.mean([P_arr_hard[k][i,i]for i in range(3)])
        avg_std.append(np.sqrt(avg_var))
    
    
    avg_std=np.array(avg_std)
    avg_pos=np.array(avg_pos)
    #plot
    lower_bound = avg_pos - 3 * avg_std
    upper_bound = avg_pos + 3 * avg_std

    t=np.arange(0,len(x_arr_hard))
    plt.figure()
    plt.plot(t,avg_pos/1000,label="x")
    plt.fill_between(t,avg_pos/1000 - 3 * avg_std,avg_pos/1000+ 3 * avg_std,color="blue",alpha=0.2,label="±3σ (Uncertainty)",)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position Estimates and Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Variance of the position estimates by itself
    plt.figure()
    plt.fill_between(t,avg_pos/1000 - 3 * avg_std,avg_pos/1000+ 3 * avg_std,color="blue",alpha=0.2,label="±3σ (Uncertainty)",)
    plt.xlabel("Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Uncertainty")
    plt.show()

    threshold = 10000  # Threshold to filter values

    # Create a boolean mask where values are below the threshold
    mask = rho_arr_hard <= threshold

    # Find indices for unfiltered values (those above the threshold)
    unfiltered_indices = np.where(~mask)[0]  # Indices where mask is False

    # Initialize lists to store the variances
    unfiltered_avg_pos = []
    unfiltered_avg_var = []

    # Loop through the unfiltered indices and calculate variances
    for idx in unfiltered_indices:
        # Calculate mean position (x, y, z) for the unfiltered step
        unfiltered_avg_pos.append(np.mean(x_arr_hard[idx][:3]))
        # Calculate average variance for position (x, y, z)
        avg_var = np.mean([P_arr_hard[idx][i, i] for i in range(3)])
        unfiltered_avg_var.append(avg_var)

    # Convert to numpy arrays for further analysis or plotting
    unfiltered_avg_pos = np.array(unfiltered_avg_pos)
    print('unfiltered_avg_pos',unfiltered_avg_pos.shape)
    unfiltered_avg_var = np.array(unfiltered_avg_var)

    # Plot the variances
    mask_std=avg_std>75
    print('avg_var',avg_std)
    print('mask_std',mask_std.shape)
    unfiltered_avg_pos=avg_pos[mask_std]
    unfiltered_avg_var=avg_std[mask_std]
    unfiltered_indices=np.where(~mask_std)[0]
    print("Unfiltered indices: ", unfiltered_indices)

    avg_pos = np.array(avg_pos)
    # Initialize arrays to store variances
    var_x = []
    var_y = []
    var_z = []
    var_vx = []
    var_vy = []
    var_vz = []

    # Extract variances from the diagonal of each covariance matrix
    for idx in unfiltered_indices:
        P = P_arr_hard[idx]  # Extract the covariance matrix
        var_x.append(P[0, 0])
        var_y.append(P[1, 1])
        var_z.append(P[2, 2])
        var_vx.append(P[3, 3])
        var_vy.append(P[4, 4])
        var_vz.append(P[5, 5])
        

    # Convert to numpy arrays
    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_z = np.array(var_z)
    var_vx = np.array(var_vx)
    var_vy = np.array(var_vy)
    var_vz = np.array(var_vz)
    std_x=np.zeros(len(var_x))
    std_y=np.zeros(len(var_y))
    std_z=np.zeros(len(var_z))
    std_vx=np.zeros(len(var_vx))
    std_vy=np.zeros(len(var_vy))
    std_vz=np.zeros(len(var_vz))

    for i in range(0,len(var_x)):
        std_x[i]=np.sqrt(var_x[i])*1000
        std_y[i]=np.sqrt(var_y[i])*1000
        std_z[i]=np.sqrt(var_z[i])*1000
        std_vx[i]=np.sqrt(var_vx[i])*1000
        std_vy[i]=np.sqrt(var_vy[i])*1000
        std_vz[i]=np.sqrt(var_vz[i])*1000

    time_steps = np.arange(len(unfiltered_indices))  # Time steps for x-axis
    t=np.arange(0,len(avg_pos))
    # Plot the variances in subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("±3σ Uncertainty, and Position Estimates")

    axs[0, 0].fill_between(time_steps, 3*std_x, -3*std_x, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[0, 0].plot(t,x_var,label="x")
    axs[0, 0].set_title("Variance of x")
    axs[0, 0].set_xlabel("Time Steps")
    axs[0, 0].set_ylabel("Variance")
    axs[0, 0].grid(True)

    axs[0, 1].fill_between(time_steps, 3*std_y, -3*std_y, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[0, 1].plot(t,y_var,label="y")
    axs[0, 1].set_title("Variance of y")
    axs[0, 1].set_xlabel("Time Steps")
    axs[0, 1].set_ylabel("Variance")
    axs[0, 1].grid(True)

    axs[0, 2].fill_between(time_steps, 3*std_z, -3*std_z, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[0, 2].plot(t,z_var,label="z")
    axs[0, 2].set_title("Variance of z")
    axs[0, 2].set_xlabel("Time Steps")
    axs[0, 2].set_ylabel("Variance")
    axs[0, 2].grid(True)

    axs[1, 0].fill_between(time_steps, 3*std_vx, -3*std_vx, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[1, 0].plot(t,vx_var,label="vx")
    axs[1, 0].set_title("Variance of v_x")
    axs[1, 0].set_xlabel("Time Steps")
    axs[1, 0].set_ylabel("Variance")
    axs[1, 0].grid(True)

    axs[1, 1].fill_between(time_steps, 3*std_vy, -3*std_vy, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[1, 1].plot(t,vy_var,label="vy")
    axs[1, 1].set_title("Variance of v_y")
    axs[1, 1].set_xlabel("Time Steps")
    axs[1, 1].set_ylabel("Variance")
    axs[1, 1].grid(True)

    axs[1, 2].fill_between(time_steps, 3*std_vz, -3*std_vz, color="blue", alpha=0.2, label="±3σ (Uncertainty)")
    axs[1, 2].plot(t,vz_var,label="vz")
    axs[1, 2].set_title("Variance of v_z")
    axs[1, 2].set_xlabel("Time Steps")
    axs[1, 2].set_ylabel("Variance")
    axs[1, 2].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
