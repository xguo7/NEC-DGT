# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:10:36 2018

@author: gxjco
"""
import numpy as np
    
def read_data(self):    
      a=np.ones((self.No,self.No))-np.eye(self.No)
      x=np.load('C:/Users/gxjco/Desktop/mutli-attributed/dataset/degree/degree20_x.npy')
      y=np.load('C:/Users/gxjco/Desktop/mutli-attributed/dataset/degree/degree20_y.npy')

      node_x=np.zeros((x.shape[0],x.shape[1]))
      node_y=np.zeros((x.shape[0],x.shape[1]))
      for i in range(x.shape[0]):
         node_x[i]=sum(x[i]*np.eye(self.No))
         node_y[i]=sum(y[i]*np.eye(self.No))
      node_x=node_x.reshape(x.shape[0],1,self.No)
      node_y=node_y.reshape(x.shape[0],1,self.No) 
       
      data=np.zeros((x.shape[0],x.shape[1],x.shape[1],2))
      for i in range(x.shape[0]):
         data[i,:,:,0]=x[i,:,:]*a  #diagnal is removed
         data[i,:,:,1]=y[i,:,:]*a
     
      Rr_data=np.zeros((5000,self.No,self.Nr),dtype=float);
      Rs_data=np.zeros((5000,self.No,self.Nr),dtype=float);
      Ra_data=np.zeros((5000,self.Dr,self.Nr),dtype=float); 
      Ra_label=np.zeros((5000,self.Dr,self.Nr),dtype=float);  
      
      cnt=0
      for i in range(self.No):
       for j in range(self.No):
         if(i!=j):
           Rr_data[:,i,cnt]=1.0;
           Rs_data[:,j,cnt]=1.0;
           for k in range(data.shape[0]):
             Ra_data[k,int(data[k,i,j,0]),cnt]=1
             Ra_label[k,int(data[k,i,j,1]),cnt]=1
           cnt+=1;   
        
      node_data_train=node_x[0:int(x.shape[0]/2)]
      node_label_train=node_y[0:int(x.shape[0]/2)]
      node_data_test=node_x[int(x.shape[0]/2):x.shape[0]]
      node_label_test=node_y[int(x.shape[0]/2):x.shape[0]] 
      Ra_data_train=Ra_data[0:int(x.shape[0]/2)]
      Ra_data_test=Ra_data[int(x.shape[0]/2):x.shape[0]]  
      Ra_label_train=Ra_label[0:int(x.shape[0]/2)]
      Ra_label_test=Ra_label[int(x.shape[0]/2):x.shape[0]] 

      return node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data

