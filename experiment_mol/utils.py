# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:10:36 2018

@author: gxjco
"""
import numpy as np
    
def read_data():    
      node_feature=np.transpose(np.load('C:/Users/gxjco/Desktop/mole/f_atom.npy'),[0,2,1])[:5000]
      adj=np.load('C:/Users/gxjco/Desktop/mole/adj.npy')[:5000]
      n_node_list=np.load('C:/Users/gxjco/Desktop/mole/n_node_list.npy')[:5000]
      data_input=np.zeros((adj.shape[0],adj.shape[1],adj.shape[2],5))
      data_target=np.zeros((adj.shape[0],adj.shape[1],adj.shape[2],5))
      for i in range(adj.shape[0]):
          for a in range(int(n_node_list[i])):
              for b in range(int(n_node_list[i])):
                  data_input[i,a,b,int(adj[i,a,b,0])]=1
                  data_target[i,a,b,int(adj[i,a,b,1])]=1
                  
      Rr_data=np.zeros((5000,20,380),dtype=float);
      Rs_data=np.zeros((5000,20,380),dtype=float);
      Ra_data=np.zeros((5000,5,380),dtype=float); 
      Ra_label=np.zeros((5000,5,380),dtype=float); 
      
      cnt=0
      for i in range(20):
       for j in range(20):
         if(i!=j):
           Rr_data[:,i,cnt]=1.0;
           Rs_data[:,j,cnt]=1.0;
           Ra_data[:,:,cnt]=data_input[:5000,i,j]
           Ra_label[:,:,cnt]=data_target[:5000,i,j]
           cnt+=1;   
        
      node_data_train=node_feature[:2500]
      node_data_test=node_feature[2500:]
      Ra_data_train=Ra_data[:2500]
      Ra_data_test=Ra_data[2500:]  
      Ra_label_train=Ra_label[0:2500]
      Ra_label_test=Ra_label[2500:] 
      return node_data_train,node_data_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,n_node_list

