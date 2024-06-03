import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal
import os

class GMM():
    def __init__(self,K_numOfGauss=5,BG_thresh=0.6, alpha=0.01, height=240, width=320, colorChannels=3):
        self.K_numOfGauss=K_numOfGauss
        self.BG_thresh=BG_thresh
        self.alpha=alpha
        self.height=height
        self.width=width
        self.colorChannels=colorChannels
   
        self.mus = np.full((self.height, self.width, self.K_numOfGauss, self.colorChannels), 122)
        self.sigmaSQs = np.full((self.height, self.width, self.K_numOfGauss), 36.0)
        self.omegas = np.full((self.height, self.width, self.K_numOfGauss), 1.0 / self.K_numOfGauss)
    
    def reorder(self):
        ratios = self.omegas / np.sqrt(self.sigmaSQs)
        indices = np.argsort(ratios, axis=2)[:, :, ::-1]
        self.mus = np.take_along_axis(self.mus, indices[..., None], axis=2)
        self.sigmaSQs = np.take_along_axis(self.sigmaSQs, indices, axis=2)
        self.omegas = np.take_along_axis(self.omegas, indices, axis=2)

        cum_probs = np.cumsum(self.omegas, axis=2)
        BG_pivot = np.argmax(cum_probs >= self.BG_thresh, axis=2)
        BG_pivot[BG_pivot == 0] = self.K_numOfGauss - 2  # Adjust for cases with no pivot found
        return BG_pivot

    
    def updateParam(self, curFrame, BG_pivot):
        assert curFrame.shape == (self.height, self.width, self.colorChannels)
        
        labels=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                x=curFrame[i,j]
                match=-1
                for k in range(self.K_numOfGauss):
                    CoVarInv=np.linalg.inv(self.sigmaSQs[i,j,k]*np.eye(self.colorChannels)) 
                    x_mu=x-self.mus[i,j,k]
                    dist=np.dot(x_mu.T, np.dot(CoVarInv, x_mu))
                    if dist<6.25*self.sigmaSQs[i,j,k]:
                        match=k
                        break
                if match!=-1:  ## a match found
                    ##update parameters
                    self.omegas[i,j]=(1.0-self.alpha)*self.omegas[i,j]
                    self.omegas[i,j,match]+=self.alpha
                    rho=self.alpha * multivariate_normal.pdf(x,self.mus[i,j,match],np.linalg.inv(CoVarInv))
                    self.sigmaSQs[match]=(1.0-rho)*self.sigmaSQs[i,j,match]+rho*np.dot((x-self.mus[i,j,match]).T, (x-self.mus[i,j,match]))
                    self.mus[i,j,match]=(1.0-rho)*self.mus[i,j,match]+rho*x
                    ##label the pixel
                    if match>BG_pivot[i,j]:
                        labels[i,j]=250
                else:
                    self.mus[i,j,-1]=x
                    labels[i,j]=250
        return labels