import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from typing import List

def polartorect(data:np.ndarray) -> np.ndarray:
    r = data[:,0]
    theta = data[:,1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x,y))

def create_mesh(data:np.ndarray,res:int) -> List[np.ndarray]:
    '''
    data: np.ndarray, an array of shape(n,2) of x and y coordinates of measured 
    data
    '''
    maxr = np.max(np.hypot(data[:,0],data[:,1]))
    x = np.linspace(-maxr,maxr,res)
    y = np.linspace(-maxr,maxr,res)
    xx,yy = np.meshgrid(x,y)
    return [xx,yy]

def createplot_polar(data:np.ndarray,res:int,ax) -> None:
    '''
    data: np.ndarray, an array of shape(n,3) of x, y and z coordinates of measured data in
    cylindrical coordinates
    '''
    #check shape of data
    if data.shape[1] != 3:
        raise ValueError("Data should be of shape (n,3)")
    r = data[:,0]
    theta =  data[:,1]
    z= data[:,2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    interp = LinearNDInterpolator(list(zip(x,y)),z)
    xx,yy = create_mesh(data,res)
    zz = interp(xx,yy)
    ax.pcolormesh(xx,yy,zz,shading='auto')
    return None

def createplot_rect(data:np.ndarray,res:int,ax) -> None:
    '''
    data: np.ndarray, an array of shape(n,3) of x, y and z coordinates of measured data in
    rectangular coordinates
    '''
    r = np.hypot(data[:,0],data[:,1])
    theta = np.arctan2(data[:,1],data[:,0])
    data = np.column_stack((r,theta,data[:,2]))
    createplot_polar(data,res,ax)
    return None