# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:14:01 2018

@author: gxjco
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
import numpy as np
import time
import os
#from utlis import read_data, phi_E_O, phi_E_R, phi_U_O, phi_U_R,m,a_O,a_R,phi_map


def variable_summaries(var,idx):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+str(idx)):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def read_data():    
   a=np.ones((FLAGS.No,FLAGS.No))-np.eye(FLAGS.No)
   x=np.load('data_new_x.npy')[:343]
   y=np.load('data_new_y.npy')[:343]
   data=np.zeros((x.shape[3],x.shape[0],x.shape[1],5))
   for i in range(x.shape[3]):
       data[i,:,:,0:4]=x[:,:,:,i]
       data[i,:,:,4]=y[:,:,i]
   node_x=np.zeros((x.shape[3],x.shape[0]))
   node_y=np.zeros((x.shape[3],x.shape[0]))
   for i in range(x.shape[3]):
       node_x[i]=sum(data[i,:,:,0]*np.eye(FLAGS.No))
       node_y[i]=sum(data[i,:,:,4]*np.eye(FLAGS.No))
   for i in range(x.shape[3]):
       data[i,:,:,0]=data[i,:,:,0]*a  #diagnal is removed
       data[i,:,:,4]=data[i,:,:,4]*a
   node_x=node_x.reshape(x.shape[3],1,FLAGS.No)
   node_y=node_y.reshape(x.shape[3],1,FLAGS.No)   
   
   Rr_data=np.zeros((343,FLAGS.No,FLAGS.Nr),dtype=float);
   Rs_data=np.zeros((343,FLAGS.No,FLAGS.Nr),dtype=float);
   Ra_data=np.zeros((343,FLAGS.Dr,FLAGS.Nr),dtype=float); 
   Ra_label=np.zeros((343,FLAGS.Dr,FLAGS.Nr),dtype=float); 
   X_data=np.zeros((343,FLAGS.Dx,FLAGS.No),dtype=float); 
   
   
   cnt=0
   for i in range(FLAGS.No):
    for j in range(FLAGS.No):
      if(i!=j):
        Rr_data[:,i,cnt]=1.0;
        Rs_data[:,j,cnt]=1.0;
        Ra_data[:,0,cnt]=data[:,i,j,0]
        Ra_label[:,0,cnt]=data[:,i,j,4]
        cnt+=1;
   
   for i in range(343):
       for j in range(FLAGS.No):
          X_data[i,:,j]=np.array([data[i,0,0,1],data[i,0,0,2],data[i,0,0,3]])
        
   node_data_train=node_x[0:200]
   node_label_train=node_y[0:200]
   node_data_test=node_x[200:343]
   node_label_test=node_y[200:343] 
   Ra_data_train=Ra_data[0:200]
   Ra_data_test=Ra_data[200:343]  
   Ra_label_train=Ra_label[0:200]
   Ra_label_test=Ra_label[200:343] 
   X_data_train=X_data[:200]
   X_data_test=X_data[200:343]
   return node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,X_data_train,X_data_test



def m(O,Rr,Rs,Ra):
  #return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)),Ra],1);
  return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],1);

def phi_E_O(B):
  h_size=50;
  B_trans=tf.transpose(B,[0,2,1]);
  B_trans=tf.reshape(B_trans,[-1,(2*FLAGS.Ds+FLAGS.Dr)]);
  
  w1 = tf.Variable(tf.truncated_normal([(2*FLAGS.Ds+FLAGS.Dr), h_size], stddev=0.1), name="r_w1o", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="r_b1o", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
  w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w2o", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([h_size]), name="r_b2o", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  
  w3 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w3o", dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([h_size]), name="r_b3o", dtype=tf.float32);
  h3 = tf.nn.relu(tf.matmul(h2, w3) + b3);
  
  w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4o", dtype=tf.float32);
  b4 = tf.Variable(tf.zeros([h_size]), name="r_b4o", dtype=tf.float32);
  h4 = tf.nn.relu(tf.matmul(h3, w4) + b4);
  
  w5 = tf.Variable(tf.truncated_normal([h_size, FLAGS.De_o], stddev=0.1), name="r_w5o", dtype=tf.float32);
  b5 = tf.Variable(tf.zeros([FLAGS.De_o]), name="r_b5o", dtype=tf.float32);
  h5 = tf.matmul(h4, w5) + b5;
  
  h5_trans=tf.reshape(h5,[-1,FLAGS.Nr,FLAGS.De_o]);
  h5_trans=tf.transpose(h5_trans,[0,2,1]);
  return(h5_trans);
  
def a_O(O,Rr,X,E):
  E_bar=tf.matmul(E,tf.transpose(Rr,[0,2,1]));
  #O_2=tf.stack(tf.unstack(O,FLAGS.Ds,1)[3:5],1);
  #return (tf.concat([O_2,X,E_bar],1));
  return (tf.concat([O,X,E_bar],1));

def phi_U_O(C):
  h_size=50;
  C_trans=tf.transpose(C,[0,2,1]);
  C_trans=tf.reshape(C_trans,[-1,(FLAGS.Ds+FLAGS.Dx+FLAGS.De_o)]);
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.Ds+FLAGS.Dx+FLAGS.De_o), h_size], stddev=0.1), name="o_w1o", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="o_b1o", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Ds], stddev=0.1), name="o_w2o", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Ds]), name="o_b2o", dtype=tf.float32);
  h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
  h2_trans=tf.reshape(h2,[-1,FLAGS.No,FLAGS.Ds]);
  h2_trans=tf.transpose(h2_trans,[0,2,1]);
  return(h2_trans);
  
def phi_E_R(B):
  h_size=50;
  B_trans=tf.transpose(B,[0,2,1]);
  B_trans=tf.reshape(B_trans,[-1,(2*FLAGS.Ds+FLAGS.Dr)]);
  
  w1 = tf.Variable(tf.truncated_normal([(2*FLAGS.Ds+FLAGS.Dr), h_size], stddev=0.1), name="r_w1r", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="r_b1r", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
  w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w2r", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([h_size]), name="r_b2r", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  
  w3 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w3r", dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([h_size]), name="r_b3", dtype=tf.float32);
  h3 = tf.nn.relu(tf.matmul(h2, w3) + b3);
  
  w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4r", dtype=tf.float32);
  b4 = tf.Variable(tf.zeros([h_size]), name="r_b4r", dtype=tf.float32);
  h4 = tf.nn.relu(tf.matmul(h3, w4) + b4);
  
  w5 = tf.Variable(tf.truncated_normal([h_size, FLAGS.De_r], stddev=0.1), name="r_w5r", dtype=tf.float32);
  b5 = tf.Variable(tf.zeros([FLAGS.De_r]), name="r_b5r", dtype=tf.float32);
  h5 = tf.matmul(h4, w5) + b5;
  
  h5_trans=tf.reshape(h5,[-1,FLAGS.Nr,FLAGS.De_r]);
  h5_trans=tf.transpose(h5_trans,[0,2,1]);
  return(h5_trans);  
  
def a_R(Ra,E):
  C_R=tf.concat([Ra,E],1)
  return (C_R); 
 
def phi_U_R(C_R):
  h_size=50;
  C_trans=tf.transpose(C_R,[0,2,1]);
  C_trans=tf.reshape(C_trans,[-1,(FLAGS.De_r+FLAGS.Dr)]);
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.De_r+FLAGS.Dr), h_size], stddev=0.1), name="o_w1r", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="o_b1r", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Dr], stddev=0.1), name="o_w2r", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Dr]), name="o_b2r", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  h2_trans=tf.reshape(h2,[-1,FLAGS.Nr,FLAGS.Dr]);
  h2_trans=tf.transpose(h2_trans,[0,2,1]);
  return(h2_trans);  
  
  


def phi_map(Ra):
  h_size=50;
  B_trans=tf.transpose(Ra,[0,2,1]);
  B_trans=tf.reshape(B_trans,[-1,(FLAGS.Dr)]);
  
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.Dr), h_size], stddev=0.1), name="r_w1m", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="r_b1m", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  
  w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w2m", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([h_size]), name="r_b2m", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  
  w3 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w3m", dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([h_size]), name="r_b3m", dtype=tf.float32);
  h3 = tf.nn.relu(tf.matmul(h2, w3) + b3);
  
  w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4m", dtype=tf.float32);
  b4 = tf.Variable(tf.zeros([h_size]), name="r_b4m", dtype=tf.float32);
  h4 = tf.nn.relu(tf.matmul(h3, w4) + b4);
  
  w5 = tf.Variable(tf.truncated_normal([h_size, FLAGS.De_o], stddev=0.1), name="r_w5m", dtype=tf.float32);
  b5 = tf.Variable(tf.zeros([FLAGS.Dmap]), name="r_b5m", dtype=tf.float32);
  h5 = tf.nn.sigmoid(tf.matmul(h4, w5) + b5)
  
  h5_trans=tf.reshape(h5,[-1,FLAGS.Nr,FLAGS.Dmap]);
  h5_trans=tf.transpose(h5_trans,[0,2,1]);
  return(h5_trans);
  
def save(saver, sess,checkpoint_dir, step):
        model_name = "g2g.model"
       # model_dir = "%s" % ('flu')
        #checkpoint_dir = os.path.join(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step) 
        
        
def load(sess, saver,checkpoint_dir):
        print(" [*] Reading checkpoint...")

        #model_dir = "%s" % ('flu')
        #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False 
        
def train():
  # Object Matrix
  mini_batch_num=FLAGS.Mini_batch
  O = tf.placeholder(tf.float32, [None,FLAGS.Ds,FLAGS.No], name="O")
  O_target=tf.placeholder(tf.float32, [None,FLAGS.Ds,FLAGS.No], name="O_target")
  # Relation Matrics R=<Rr,Rs,Ra>
  Rr = tf.placeholder(tf.float32, [None,FLAGS.No,FLAGS.Nr], name="Rr")
  Rs = tf.placeholder(tf.float32, [None,FLAGS.No,FLAGS.Nr], name="Rs")
  Ra = tf.placeholder(tf.float32, [None,FLAGS.Dr,FLAGS.Nr], name="Ra")
  Ra_target = tf.placeholder(tf.float32, [None,FLAGS.Dr,FLAGS.Nr], name="Ra_target")
  # External Effects
  X = tf.placeholder(tf.float32, [None,FLAGS.Dx,FLAGS.No], name="X")
  
  # marshalling function, m(G)=B, G=<O,R>  
  B=m(O,Rr,Rs,Ra)
  # updating the node state
  E_O=phi_E_O(B)  
  C_O=a_O(O,Rr,X,E_O)
  O_=phi_U_O(C_O)
  
  #updating the edge 
  E_R=phi_E_R(B)
  C_R=a_R(Ra,E_R)
  Ra_=phi_U_R(C_R)
  
  #mapping node and edge:
  O_map=tf.matmul(phi_map(Ra_),tf.transpose(Rr,[0,2,1]))
  

  # loss
  params_list=tf.global_variables()
  for i in range(len(params_list)):
    variable_summaries(params_list[i],i)
    
  loss_node_mse=tf.reduce_mean(tf.reduce_mean(tf.square(O_-O_target),[1,2]))
  loss_edge_mse=tf.reduce_mean(tf.reduce_mean(tf.square(Ra_-Ra_target),[1,2]))
  loss_map=tf.reduce_mean(tf.reduce_mean(tf.square(O_-O_map),[1,2]))
  
  loss_E_O = 0.001*tf.nn.l2_loss(E_O) #regulization
  loss_E_R = 0.001*tf.nn.l2_loss(E_R)
  loss_para=0
  for i in params_list:
    loss_para+=0.001*tf.nn.l2_loss(i);

    
  #optimizer  
  optimizer = tf.train.AdamOptimizer(0.0005);
  trainer=optimizer.minimize(loss_node_mse+loss_edge_mse+loss_map+loss_E_O+loss_E_R+loss_para);
  
  # tensorboard
  tf.summary.scalar('node_mse',loss_node_mse);
  tf.summary.scalar('edge_mse',loss_edge_mse);
  tf.summary.scalar('map_mse',loss_map);
  merged=tf.summary.merge_all();
  writer=tf.summary.FileWriter(FLAGS.log_dir);

  sess=tf.InteractiveSession();
  tf.global_variables_initializer().run();
  #init_op = tf.global_variables_initializer()
  #sess.run(init_op)
  
  saver=tf.train.Saver()

  #read data
  node_data_train,node_data_test,node_label_train,node_label_test,Ra_data_train,Ra_data_test,Ra_label_train,Ra_label_test,Rr_data,Rs_data,X_data_train,X_data_test=read_data()
  
  # Training
  if FLAGS.Type=='train': 
   max_epoches=FLAGS.epoch
   counter=1
   for i in range(max_epoches):
     tr_loss_node=0
     tr_loss_edge=0
     tr_loss_map=0
     for j in range(int(len(node_data_train)/mini_batch_num)):
       batch_O=node_data_train[j*mini_batch_num:(j+1)*mini_batch_num];
       batch_O_target=node_label_train[j*mini_batch_num:(j+1)*mini_batch_num];
       batch_Ra=Ra_data_train[j*mini_batch_num:(j+1)*mini_batch_num]
       batch_Ra_target=Ra_label_train[j*mini_batch_num:(j+1)*mini_batch_num]
       batch_X_train=X_data_train[j*mini_batch_num:(j+1)*mini_batch_num]
       tr_loss_part_node,tr_loss_part_edge,tr_loss_part_map,_=sess.run([loss_node_mse,loss_edge_mse,loss_map,trainer],feed_dict={O:batch_O,O_target:batch_O_target,Ra:batch_Ra,Ra_target:batch_Ra_target,Rr:Rr_data[:mini_batch_num],Rs:Rs_data[:mini_batch_num],X:batch_X_train});
       tr_loss_node+=tr_loss_part_node
       tr_loss_edge+=tr_loss_part_edge
       tr_loss_map+=tr_loss_part_map
     print("Epoch "+str(i+1)+" node MSE: "+str(tr_loss_node/(int(len(node_data_train)/mini_batch_num)))+" edge MSE: "+str(tr_loss_edge/(int(len(node_data_train)/mini_batch_num)))+" map MSE: "+str(tr_loss_map/(int(len(node_data_train)/mini_batch_num))));
     counter+=1
     save(saver,sess,FLAGS.checkpoint_dir, counter)
     
  #testing   
  if FLAGS.Type=='test':  
        if load(sess,saver,FLAGS.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        te_loss_node=0
        te_loss_edge=0
        te_loss_map=0
        O_=[]
        Ra_=[]
        for j in range(int(len(node_data_test)/mini_batch_num)):
          batch_O=node_data_test[j*mini_batch_num:(j+1)*mini_batch_num];
          batch_O_target=node_label_test[j*mini_batch_num:(j+1)*mini_batch_num];
          batch_Ra=Ra_data_test[j*mini_batch_num:(j+1)*mini_batch_num]
          batch_Ra_target=Ra_label_test[j*mini_batch_num:(j+1)*mini_batch_num]
          batch_X_train=X_data_test[j*mini_batch_num:(j+1)*mini_batch_num]
          te_loss_part_node,te_loss_part_edge,te_loss_part_map,Ra_part,O_part,_=sess.run([loss_node_mse,loss_edge_mse,loss_map,Ra_,O_,trainer],feed_dict={O:batch_O,O_target:batch_O_target,Ra:batch_Ra,Ra_target:batch_Ra_target,Rr:Rr_data[:mini_batch_num],Rs:Rs_data[:mini_batch_num],X:batch_X_train});
          te_loss_node+=te_loss_part_node
          te_loss_edge+=te_loss_part_edge
          te_loss_map+=te_loss_part_map 
          O_.append(O_part)
          Ra_.append(Ra_part)                   
        print(" node MSE: "+str(te_loss_node(int(len(node_data_test)/mini_batch_num)))+" edge MSE: "+str(te_loss_edge(int(len(node_data_test)/mini_batch_num)))+" map MSE: "+str(te_loss_map)(int(len(node_data_test)/mini_batch_num)));
        np.save('O_.npy',np.array(O_).reshape(len(node_data_test),FLAGS.Ds,FLAGS.No))
        np.save('Ra_.npy',np.array(Ra_).reshape(len(node_data_test),FLAGS.Dr,FLAGS.Nr))
        


'''
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20,
                      help='number of training epochs')
parser.add_argument('--log_dir', type=str, default='/tmp/interaction-network/',
                      help='Summaries log directry')
parser.add_argument('--Ds', type=int, default=1,
                      help='The State Dimention')
parser.add_argument('--No', type=int, default=20,
                      help='The Number of Objects')
parser.add_argument('--Nr', type=int, default=380,
                      help='The Number of Relations')
parser.add_argument('--Dr', type=int, default=1,
                      help='The Relationship Dimension')
parser.add_argument('--Dx', type=int, default=3,
                      help='The External Effect Dimension')
parser.add_argument('--De_o', type=int, default=50,
                      help='The Effect Dimension on node')
parser.add_argument('--De_r', type=int, default=50,
                      help='The Effect Dimension on edge')
parser.add_argument('--Dmap', type=int, default=50,
                      help='The dimension on mapping')
parser.add_argument('--Mini_batch', type=int, default=1,
                      help='The training mini_batch')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint',
                      help='models are saved here')
parser.add_argument('--Type', dest='Type', default='test',
                      help='train or test')
FLAGS, unparsed = parser.parse_known_args()

def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  #with tf.Session() as sess:
   #   train(sess)
  train()

if __name__ == '__main__':
  main(_)
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)'''