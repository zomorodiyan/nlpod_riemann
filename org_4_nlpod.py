# -*- coding: utf-8 -*-
"""
Nonlinear POD (NLPOD) framework implementation for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import random
import time as timer
import pickle

import matplotlib.pyplot as plt
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM

from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#%% set seeds for reproducibility
random.seed(0)
np.random.seed(1)
#tf.random.set_seed(2)

#%% Define Keras functions
def coeff_determination(y_pred, y_true): #Order of function inputs is important here
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
              training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

#%% Define functions
def import_fom_data(Re,nx,ny,n):
    folder = '/data_'+ str(nx) + '_' + str(ny)
    filename = './Results/Re_'+str(int(Re))+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

#%% POD routines
def PODproj_svd(u,Phi): #Projection
    a = np.dot(Phi.T,u)  # u = Phi * a if shape of a is [nr,ns]
    return a

def PODrec_svd(a,Phi): #Reconstruction
    u = np.dot(Phi,a)
    return u

#%% AE routines
def AE_trainDNN(input_dim,latent_dim,xtrain,xptrain,ytrain,xvalid,xpvalid,yvalid,configs=dict()):

    seed = configs['seed']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    activation = configs['activation']
    verbose= configs['verbose']
    initializer = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed+1234)
        #tf.random.set_seed(seed+2345)

    #n_samples = xtrain.shape[0]
    if not xtrain.shape[0] == ytrain.shape[0]:
        print("Size mismatch between xtrain and ytrain! \
              The passed xtrain has {0} samples while ytrain have {1} samples.".format(xtrain.shape[0], ytrain.shape[0]))
        raise AssertionError

    if not xvalid.shape[0] == yvalid.shape[0]:
        print("Size mismatch between xvalid and yvalid! \
              The passed xvalid has {0} samples while yvalid have {1} samples.".format(xvalid.shape[0], yvalid.shape[0]))
        raise AssertionError

    #% Encoder
    input_data = Input(shape=(input_dim,))     # define the input to the encoder
    x = input_data
    x = Dense(128,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(64,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(32,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(8,activation=activation,kernel_initializer=initializer)(x)
    encoded = Dense(latent_dim,activation='linear',kernel_initializer=initializer)(x)
    encoder = Model(input_data, encoded, name="encoder")

    #% Decoder
    # start building the decoder model which will accept the output of the encoder
    latent_input = Input(shape=(latent_dim,))
    pg_input = Input(shape=(1,))
    x = concatenate(inputs=[latent_input, pg_input])
    x = Dense(8,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(32,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(64,activation=activation,kernel_initializer=initializer)(x)
    x = Dense(128,activation=activation,kernel_initializer=initializer)(x)
    decoded = Dense(input_dim,activation='linear',kernel_initializer=initializer)(x)
    decoder = Model(inputs=[latent_input,pg_input], outputs=decoded, name="decoder")

    #% Autoencoder
    # autoencoder is the encoder + decoder
    autoencoder = Model(inputs=[input_data,pg_input], outputs=decoder(inputs=[encoder(input_data),pg_input]),name="autoencoder")

    training_time_init = timer.time()

    # compile model
    autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=[coeff_determination])

    # fit the model (optimize weights and biases)
    history = autoencoder.fit(x=[xtrain,xptrain], y=ytrain, epochs=epochs, batch_size=batch_size,
                        validation_data= ([xvalid,xpvalid],yvalid), verbose=verbose)

    training_time = timer.time() - training_time_init

    # evaluate the model
    scores = autoencoder.evaluate(x=[xtrain,xptrain], y=ytrain, verbose=1)
    print("%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    return encoder,decoder,autoencoder,train_loss,valid_loss,training_time

#%% LSTM routines

def lstm_scaling(lookback,xtrain,ytrain,low=-1,high=1,split=True,split_size=0.2,shuffle=True,scaling=True):

    # Scaling xtrain
    p,q,r = xtrain.shape
    if not q == lookback:
        print("Size mismatch for xtrain! \
              The passed features has a lookback of {0} while expected {1}.".format(q, lookback))
        raise AssertionError
    xtrain2d = xtrain.reshape(p*q,r)

    xscaler = MinMaxScaler(feature_range=(low,high))
    xscaler = xscaler.fit(xtrain2d)
    xtrain2d = xscaler.transform(xtrain2d)
    xtrain_sc = xtrain2d.reshape(p,q,r)

    yscaler = MinMaxScaler(feature_range=(low,high))
    yscaler = yscaler.fit(ytrain)
    ytrain_sc = yscaler.transform(ytrain)

    #split into training and testing
    if split:
        #split to train and test
        slice_list = np.arange(0,p,1)
        slice_train, slice_valid = train_test_split(slice_list, test_size=split_size, shuffle=shuffle)
    else:
        slice_train = np.arange(0,p,1)
        slice_valid = []

    if not scaling:
        xtrain_sc = xtrain
        ytrain_sc = ytrain

    return xtrain_sc[slice_train], xtrain_sc[slice_valid], xscaler,\
           ytrain_sc[slice_train], ytrain_sc[slice_valid], yscaler


def lstm_train(xtrain,ytrain,xvalid,yvalid,configs=dict()):

    seed = configs['seed']
    nlayer = configs['nlayer']
    nhidden = configs['nhidden']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    activation = configs['activation']
    verbose= configs['verbose']
    lookback= configs['lookback']

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed+1234)
        #tf.random.set_seed(seed+2345)

    #n_samples = xtrain.shape[0]
    if not xtrain.shape[0] == ytrain.shape[0]:
        print("Size mismatch between xtrain and ytrain! \
              The passed xtrain has {0} samples while ytrain have {1} samples.".format(xtrain.shape[0], ytrain.shape[0]))
        raise AssertionError

    if not xvalid.shape[0] == yvalid.shape[0]:
        print("Size mismatch between xvalid and yvalid! \
              The passed xvalid has {0} samples while yvalid have {1} samples.".format(xvalid.shape[0], yvalid.shape[0]))
        raise AssertionError

    if not xtrain.shape[1] == lookback:
        print("Size mismatch for xtrain! \
              The passed features has a lookback of {0} while expected {1}.".format(xtrain.shape[1], lookback))
        raise AssertionError

    nd_in = xtrain.shape[2]
    nd_out = ytrain.shape[1]

    # functional API
    input_layer= Input(shape=(lookback, nd_in))
    xl=input_layer
    for l in range(nlayer-1):
        xl = LSTM(nhidden, return_sequences=True, activation=activation)(xl)
    xl = LSTM(nhidden, activation=activation)(xl)
    output_layer = Dense(nd_out)(xl)
    model = Model(inputs=input_layer, outputs=output_layer)

    training_time_init = timer.time()
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # fit the model (optimize weights and biases)
    history = model.fit(x=xtrain, y=ytrain, epochs=epochs, batch_size=batch_size,
                        validation_data= (xvalid,yvalid), verbose=verbose)

    training_time = timer.time() - training_time_init

    # evaluate the model
    scores = model.evaluate(x=xtrain, y=ytrain, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    return model, train_loss, valid_loss, training_time


def lstm_predict(model,xscaler,yscaler,xpred,lookback,scaling):

    nd_in = xpred.shape[1]
    xpred_sc = xscaler.transform(xpred)

    if not scaling:
        xpred_sc = xpred

    xpred_sc = xpred_sc.reshape(1,lookback,nd_in)

    ytest_sc = model.predict(xpred_sc)
    ytest = yscaler.inverse_transform(ytest_sc)

    if not scaling:
        ytest = ytest_sc

    return ytest

def create_training_data_lstm(lookback, features, labels):
    """
    create valid training data sets for the LSTM

    Parameters
    ----------
    features: 2D numpy array with the shape of (num_of_samples x dimension_of_input)
    labels: 2D numpy array with the shape of (num_of_samples x dimension_of_output)

    Returns
    -------
    xtrain : 3D numpy array with the shape of (num_of_samples_lstm x lookback x dimension_of_input)
    ytrain : 2D numpy array with the shape of (num_of_samples_lstm x dimension_of_output)
    remark: note that num_of_samples_lstm can be different than num_of_samples as a result of embedding lookback

    """

    if not features.shape[0] == labels.shape[0]:
        print("Size mismatch between features and labels! \
              The passed features has {0} samples while labels have {1} samples.".format(features.shape[0], labels.shape[0]))
        raise AssertionError

    n_samples = features.shape[0]
    nd_in = features.shape[1]
    nd_out = labels.shape[1]

    ytrain = [(labels[i,:]-labels[i-1,:])/dtrom for i in range(lookback,n_samples)]
    ytrain = np.array(ytrain)
    if not (ytrain.shape[0]==n_samples-lookback) and (ytrain.shape[1]==nd_out):
        print("Inconsistent size for ytrain. This should never happen, please check here!")
        raise AssertionError

    xtrain = np.empty([n_samples-lookback,lookback,nd_in], dtype=np.double)
    for i in range(n_samples-lookback):
        for j in range(lookback):
            xtrain[i,j,:] = features[i+j,:]

    return xtrain, ytrain


def create_parametric_training_data_lstm(lookback, parameters, features, labels):
    """
    create valid training data sets for the LSTM from different sets of data corresponding to different parameters

    Parameters
    ----------
    features: 2D numpy array with the shape of (num_of_parameters x num_of_samples x dimension_of_input)
    labels: 2D numpy array with the shape of (num_of_parameters x num_of_samples x dimension_of_output)
    same_time: bool, indicating wheter output and input are at the same time

    Returns
    -------
    xtrain : 3D numpy array with the shape of (num_of_samples_lstm x lookback x dimension_of_input)
    ytrain : 2D numpy array with the shape of (num_of_samples_lstm x dimension_of_output)
    remark: note that num_of_samples_lstm can be different than num_of_samples as a result of embedding lookback

    """

    if not features.shape[0] == labels.shape[0]:
        print("Size mismatch between features and labels! \
              The passed features has {0} parameter values while labels have {1} values.".format(features.shape[0], labels.shape[0]))
        raise AssertionError

    n_param = features.shape[0]
    for p in range(n_param):
        xt, yt = create_training_data_lstm(lookback, features[p,...], labels[p,...])
        xpt = np.ones([xt.shape[0],xt.shape[1],1])*parameters[p]

        if p == 0:
            xtrain = np.concatenate([xpt,xt], axis=2)
            ytrain = yt
        else:
            xtrain = np.vstack((xtrain,np.concatenate([xpt,xt], axis=2)))
            ytrain = np.vstack((ytrain,yt))

    return xtrain , ytrain

def plot_loss(train_loss,valid_loss):
    plt.figure()
    epochs = range(1, len(train_loss) + 1)
    plt.semilogy(epochs, train_loss, 'b', label='Training loss')
    plt.semilogy(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    #plt.show()

    return

#%% Main program
# Inputs
lx = 8
ly = 1
nx = 64
ny = int(nx/8)

ReList = [7e2, 9e2, 10e2, 11e2, 13e2]
ReTrain = [7e2, 9e2, 11e2, 13e2]
ReTest = [10e2]

Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = np.int(np.round(Tm/dt))

ns = 200
freq = np.int(nt/ns)

nr = 74
latent_dim = 2
lookback = 3
scaling = True
dtrom = dt*freq
ens_size = 10
seed_lib = np.arange(10)

#%% grid
dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% reading data

wdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #vorticity
sdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #streamfunction
tdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #temperature

nstart= 0
nend= nt
nstep= freq
for p, Re in enumerate(ReList):
    ii = 0
    for n in range(nstart,nend+1,nstep):
        w,s,t = import_fom_data(Re,nx,ny,n)
        wdata[p,:,ii] = w.reshape([-1,])
        sdata[p,:,ii] = s.reshape([-1,])
        tdata[p,:,ii] = t.reshape([-1,])
        ii = ii + 1


#%% Load POD data

folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './Results/'+folder+'/POD_data.npz'
data = np.load(filename)
wmean = data['wmean']
smean = data['smean']
tmean = data['tmean']
wbasis = data['wbasis']
sbasis = data['sbasis']
tbasis = data['tbasis']
wcoeff = data['wcoeff']
tcoeff = data['tcoeff']


#%% Define training data
inc = 1
Rmin = -1
Rmax = 1
for p, Re in enumerate(ReTrain):
    tmp = tcoeff[ReList.index(Re),:nr,::inc]

    X_std = (Re - ReTrain[0]) / (ReTrain[-1] - ReTrain[0])
    Res = X_std * (Rmax - Rmin) + Rmin
    if p == 0:
        tcoeff_train = tmp.T
        param_train = Res*np.ones([tmp.shape[1],1])
    else:
        tcoeff_train = np.vstack((tcoeff_train,tmp.T))
        param_train = np.vstack((param_train,Res*np.ones([tmp.shape[1],1])))

#%%
random.seed(0)
np.random.seed(1)
#tf.random.set_seed(2)

#% Split into training and validation
slice_list = np.arange(0,tcoeff_train.shape[0],1)
slice_train, slice_valid = train_test_split(slice_list, test_size=0.2, shuffle=True)

x_train, x_valid = tcoeff_train[slice_train], tcoeff_train[slice_valid]
xp_train, xp_valid = param_train[slice_train], param_train[slice_valid]
y_train, y_valid = tcoeff_train[slice_train], tcoeff_train[slice_valid]

#%%
encoders, decoders, autoencoders = [], [], []
ae_train_losses, ae_valid_losses = [], []
ae_training_times = []
for m in range(ens_size):
    print(m)
    # train AE seed_lib[m]
    aeconfigs = dict(seed=seed_lib[m], epochs=250, batch_size=32, activation='tanh', verbose=1)
    encoder,decoder,autoencoder,train_loss,valid_loss,training_time_AE = AE_trainDNN(input_dim=nr,latent_dim=latent_dim,\
                                                                                      xtrain=x_train,xptrain=xp_train,ytrain=y_train,\
                                                                                      xvalid=x_valid,xpvalid=xp_valid,yvalid=y_valid,\
                                                                                      configs=aeconfigs)
    encoders.append(encoder)
    decoders.append(decoder)
    autoencoders.append(autoencoder)
    ae_train_losses.append(train_loss)
    ae_valid_losses.append(valid_loss)
    ae_training_times.append(training_time_AE)

    # Plot loss
    plot_loss(train_loss,valid_loss)

#%% Save autoencoder models
make_keras_picklable()
out_results = {'encoders':encoders, 'decoders':decoders, 'autoencoders':autoencoders,
               'ae_train_losses':ae_train_losses, 'ae_valid_losses':ae_valid_losses,
               'ae_training_times':ae_training_times}

pickle_dir = "__PICKLES"
if not os.path.isdir(pickle_dir): os.makedirs(pickle_dir)
saveto = os.path.join(pickle_dir, "NLPOD_Autencoders_nr="+str(nr)+".pickle")
out_file = open(saveto, 'wb')
pickle.dump(out_results, out_file)
out_file.close()

#%% Prepare LSTM training
inc = 1
lstm_models, xscalers, yscalers = [], [], []
lstm_train_losses, lstm_valid_losses = [], []
lstm_training_times = []
for m in range(ens_size):
    print(m)
    latent_coeff = []
    encoder = encoders[m]
    for p, Re in enumerate(ReTrain):
        dataPOD = tcoeff[ReList.index(Re),:nr,::inc].T
        latent_coeff.append(encoder.predict(dataPOD))
    latent_coeff = np.array(latent_coeff)

    random.seed(0)
    np.random.seed(1)
    #tf.random.set_seed(2)

    xt_lstm, yt_lstm = create_parametric_training_data_lstm(lookback, parameters=ReTrain, features=latent_coeff, labels=latent_coeff)

    xtrain_lstm, xvalid_lstm, xscaler,\
    ytrain_lstm, yvalid_lstm, yscaler = lstm_scaling(lookback,xt_lstm,yt_lstm,low=-1,high=1,split=True,split_size=0.2,shuffle=True,scaling=scaling)

    # Train LSTM
    lstmconfigs = dict(seed=seed_lib[m],lookback=lookback,nlayer=3,nhidden=20,epochs=250,batch_size=32,activation='tanh',verbose=0)
    lstm_model, train_loss, valid_loss, training_time_lstm = lstm_train(xtrain=xtrain_lstm,ytrain=ytrain_lstm,xvalid=xvalid_lstm,yvalid=yvalid_lstm,configs=lstmconfigs)

    lstm_models.append(lstm_model)
    xscalers.append(xscaler)
    yscalers.append(yscaler)
    lstm_train_losses.append(train_loss)
    lstm_valid_losses.append(valid_loss)
    lstm_training_times.append(training_time_lstm)

    # Plot loss
    plot_loss(train_loss,valid_loss)

#%% Save lstm models
make_keras_picklable()
out_results = {'lstm_models':lstm_models, 'xscalers':xscalers, 'yscalers':yscalers,
               'lstm_train_losses':lstm_train_losses, 'lstm_valid_losses':lstm_valid_losses,
               'lstm_training_times':lstm_training_times}

pickle_dir = "__PICKLES"
if not os.path.isdir(pickle_dir): os.makedirs(pickle_dir)
saveto = os.path.join(pickle_dir, "NLPOD_LSTMs_nr="+str(nr)+".pickle")
out_file = open(saveto, 'wb')
pickle.dump(out_results, out_file)
out_file.close()

#%% Test LSTM
for Re in ReList:

    lstm_coeff = np.zeros([ens_size,ns+1,latent_dim])
    decoded_lstm = np.zeros([ens_size,ns+1,nr])
    latent_coeff = np.zeros([ens_size,ns+1,latent_dim])
    decoded_ae = np.zeros([ens_size,ns+1,nr])

    dataPOD = tcoeff[ReList.index(Re),:nr,::inc].T

    for m in range(ens_size):
        xtest = np.zeros((lookback,latent_dim+1))

        time=0
        testing_time = 0

        encoder = encoders[m]
        decoder = decoders[m]

        lstm_model = lstm_models[m]
        xscaler = xscalers[m]
        yscaler = yscalers[m]

        latent_coeff[m,:,:] = encoder.predict(dataPOD)

        for i in range(lookback):
            testing_time_init = timer.time()
            lstm_coeff[m,i,:] = np.copy(latent_coeff[m,i,:])
            testing_time += timer.time() - testing_time_init
            xtest[i,0] = Re
            xtest[i,1:] = np.copy(latent_coeff[m,i,:])

        time = time + (lookback-1)*dt*freq
        for i in range(lookback,ns+1):
            testing_time_init = timer.time()
            lstm_coeff[m,i,:] = lstm_coeff[m,i-1,:] + dtrom*lstm_predict(lstm_model,xscaler,yscaler,xtest,lookback,scaling=scaling)
            testing_time += timer.time() - testing_time_init

            # update xtest
            for ii in range(lookback-1):
                xtest[ii,:] = xtest[ii+1,:]
            xtest[-1,1:] = np.copy(lstm_coeff[m,i,:])

            time = time + dt*freq
            if (i)%50==0:
                print(i, " ", time, " ")

        X_std = (Re - ReTrain[0]) / (ReTrain[-1] - ReTrain[0])
        Res = X_std * (Rmax - Rmin) + Rmin
        datap_test = Res*np.ones([ns+1,1])
        decoded_lstm[m,:,:] = decoder.predict([lstm_coeff[m,:,:],datap_test])
        decoded_ae[m,:,:] = decoder.predict([latent_coeff[m,:,:],datap_test])


    Phit = tbasis[:,:nr]
    t_fom = tdata[ReList.index(Re),:,:]
    t_pod2 = PODrec_svd((dataPOD[:,:latent_dim]).T,Phit[:,:latent_dim]) + tmean.reshape(-1,1)
    t_pod72 = PODrec_svd((dataPOD[:,:nr]).T,Phit[:,:nr]) + tmean.reshape(-1,1)

    t_ae = np.zeros([ens_size,(nx+1)*(ny+1),ns+1])
    t_lstm = np.zeros([ens_size,(nx+1)*(ny+1),ns+1])
    for m in range(ens_size):
        t_ae[m,:,:] = PODrec_svd(decoded_ae[m,:,:].T,Phit) + tmean.reshape(-1,1)
        t_lstm[m,:,:] = PODrec_svd(decoded_lstm[m,:,:].T,Phit) + tmean.reshape(-1,1)


    #% Save data
    folder = 'data_'+ str(nx) + '_' + str(ny)
    if not os.path.exists('./Results/'+folder):
        os.makedirs('./Results/'+folder)

    filename = './Results/'+folder+'/NonPOD_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
    np.savez(filename, latent_ae=latent_coeff, latent_lstm=lstm_coeff,\
                       b_ae=decoded_ae, b_lstm=decoded_lstm,\
                       t_ae=t_ae, t_lstm=t_lstm)
