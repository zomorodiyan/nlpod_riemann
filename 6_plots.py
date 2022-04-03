# -*- coding: utf-8 -*-
"""
Plotting scripts for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        print("---------------------------------------------------------------------")
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {"family": "normal", "weight": "bold", "size": 20}
mpl.rc("font", **font)

#%% Misc functions
def PODproj_svd(u, Phi):  # Projection
    a = np.dot(Phi.T, u)  # u = Phi * a if shape of a is [nr,ns]
    return a


def PODrec_svd(a, Phi):  # Reconstruction
    u = np.dot(Phi, a)
    return u


def import_fom_data(bc, nx, ny, n):
    filename = "./Results/npz/bc_" + str(int(bc)) + "/P" + str(int(n)) + ".npy"
    return np.load(filename)


def zbc(z):
    z[-1, :] = z[-2, :]
    z[:, -1] = z[:, -2]
    return z


#%% Define Keras functions
def coeff_determination(y_pred, y_true):  # Order of function inputs is important here
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


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
                training_config,
                custom_objects={"coeff_determination": coeff_determination},
            )
        )
    restored_model.set_weights(weights)
    return restored_model


#%%
def main():
    print("> main >")
    print(">> variables >>")
    lx = 1
    ly = 1
    nx = 100
    ny = nx

    bcList = [1, 2, 3, 4, 5]
    bcTrain = [1, 2, 3, 5]
    Tm = 0.25
    nt = 50  # 0,1,3,...,50
    ns = nt + 1
    dt = 0.25 / nt
    freq = int(nt / (ns - 1))
    # twick 1
    ens_size = 10

    #%% grid
    dx = lx / nx
    dy = ly / ny

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    print("<< variables <<")

    #%% load FOM data
    print(">> load fom >>")
    zfom = np.zeros([len(bcList), nx, ny, ns])
    for j, bc in enumerate(bcList):
        for i in range(ns):
            n = i * freq
            z = import_fom_data(bc, nx, ny, n)
            zfom[j, :, :, i] = np.copy(z)
    print("<< load fom <<")

    #%% plot FOM contour
    print(">> plot fom contour >>")
    x = np.load("Results/npz/X.npy")
    y = np.load("Results/npz/Y.npy")
    fig_bc = 4  # from 0,1,2,3,4
    snapshot = 50  # from 0,1,..,50
    z = zfom[fig_bc, :, :, 25]  # attention
    fig, ax = plt.subplots(ncols=1)
    plt.rcParams.update({"font.size": 12})
    fig.set_size_inches(6.5, 5)
    cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm")
    cbar = fig.colorbar(cs)
    ax.set_title("Lax-Liu Conf.12, top_right_initial=0.4")
    plt.ylabel("x")
    plt.xlabel("y")
    ax.grid(c="k", ls="-", alpha=0.08)
    plt.savefig("Plots/Euler_bc=" + str(fig_bc) + "_sn=" + str(snapshot) + ".png")
    print("<< plot fom contour <<")

    #%% Load true POD data
    print(">> load pod >>")
    folder = "euler_" + str(nx) + "_" + str(ny)
    filename = "./Results/" + folder + "/P_POD_data.npz"
    poddata = np.load(filename)
    zmean = poddata["zmean"]
    zbasis = poddata["zbasis"]
    print("zbasis.shape", zbasis.shape)
    zeigenvalues = poddata["zeigenvalues"]
    zric = poddata["zric"]
    print("zric.shape", zric.shape)
    print("zric", zric[:6])
    zcoeff = poddata["zcoeff"]
    print("zcoeff.shape", zcoeff.shape)

    print("<< load pod <<")
    #%% Compute true projection fields
    print(">> projection_error >>")
    zPOD2 = np.zeros([len(bcList), nx, ny, ns])
    zPOD34 = np.zeros([len(bcList), nx, ny, ns])
    for p, bc in enumerate(bcList):
        for n in range(ns):
            nr = 2
            tmp = PODrec_svd(zcoeff[p, :nr, n], zbasis[n, :nr]) + zmean
            zPOD2[p, :, :, n] = tmp.reshape([nx, ny])

            nr = 34
            tmp = PODrec_svd(zcoeff[p, :nr, n], zbasis[n, :nr]) + zmean
            zPOD34[p, :, :, n] = tmp.reshape([nx, ny])
    print("<< projection_error <<")
    #%%
    print(">> plot rec. pod contour >>")
    x = np.load("Results/npz/X.npy")
    y = np.load("Results/npz/Y.npy")
    fig_bc = 4  # {0,1,2,3,4}
    snapshot = 50  # {0,1,..,50}
    z = zPOD34[fig_bc, :, :, snapshot]
    fig, ax = plt.subplots(ncols=1)
    plt.rcParams.update({"font.size": 12})
    fig.set_size_inches(6.5, 5)
    cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm")
    cbar = fig.colorbar(cs)
    ax.set_title("Lax-Liu Conf.12, top_right_initial=0.4")
    plt.ylabel("x")
    plt.xlabel("y")
    ax.grid(c="k", ls="-", alpha=0.08)
    plt.savefig("Plots/pod_nr=2_bc=" + str(fig_bc) + "_sn=" + str(snapshot) + ".png")
    print("<< plot reconstructed pod contour <<")
    print("load gp <skiped>")
    """
#%% load GP data
    tGP2 = np.zeros([len(ReList),nx+1,ny+1,ns])
    tGP74 = np.zeros([len(ReList),nx+1,ny+1,ns])

    for p, Re in enumerate(ReList):
        nr = 2
        folder = 'data_'+ str(nx) + '_' + str(ny)
        filename = './Results/'+folder+'/GP_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
        gpdata = np.load(filename)
        tmp1 = gpdata['tGP']

        nr = 74
        folder = 'data_'+ str(nx) + '_' + str(ny)
        filename = './Results/'+folder+'/GP_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
        gpdata = np.load(filename)
        tmp2 = gpdata['tGP']
        for n in range(ns):
            tGP2[p,:,:,n] = tmp1[:,n].reshape([nx+1,ny+1])
            tGP74[p,:,:,n] = tmp2[:,n].reshape([nx+1,ny+1])
    """

    #%% Loading CAE Data
    print(">> load cae data>>")
    latent_dim = 2
    z_ae_cae = np.zeros([len(bcList), ens_size, nx, ny, ns])
    z_lstm_cae = np.zeros([len(bcList), ens_size, nx, ny, ns])
    latent_ae_cae = np.zeros([len(bcList), ens_size, latent_dim, ns])
    latent_lstm_cae = np.zeros([len(bcList), ens_size, latent_dim, ns])

    for j, bc in enumerate(bcList):
        filename = (
            "./Results/"
            + folder
            + "/CNN_AE_bc="
            + str(int(bc))
            + "_z="
            + str(latent_dim)
            + ".npz"
        )
        CNNdata = np.load(filename)
        tmp1 = CNNdata["latent_ae"]
        tmp2 = CNNdata["latent_lstm"]
        tmp3 = CNNdata["z_ae"]
        tmp4 = CNNdata["z_lstm"]
        for n in range(ns):
            latent_ae_cae[j, :, :, n] = tmp1[:, n, :]
            latent_lstm_cae[j, :, :, n] = tmp2[:, n, :]
            # z_ae_cae[j, :, :-1, :-1, n] = tmp3[:, n, :, :, 0]
            # z_lstm_cae[j, :, :-1, :-1, n] = tmp4[:, n, :, :, 0]
            z_ae_cae[j, :, :, :, n] = tmp3[:, n, :, :, 0]
            z_lstm_cae[j, :, :, :, n] = tmp4[:, n, :, :, 0]
            for m in range(ens_size):
                z_ae_cae[j, m, :, :, n] = zbc(z_ae_cae[j, m, :, :, n])
                z_lstm_cae[j, m, :, :, n] = zbc(z_lstm_cae[j, m, :, :, n])
    print("<< load cae data <<")
    #%% load CAE models
    print(">> load cae model >> (skiped)")
    make_keras_picklable()
    pickle_dir = "__PICKLES"
    loadfrom = os.path.join(pickle_dir, "CAE_Autencoders.pickle")

    infile = open(loadfrom, "rb")
    in_results = pickle.load(infile)
    infile.close()

    CAE_encoders = in_results["encoders"]
    CAE_decoders = in_results["decoders"]
    CAE_autoencoders = in_results["autoencoders"]
    CAE_ae_train_losses = in_results["ae_train_losses"]
    CAE_ae_valid_losses = in_results["ae_valid_losses"]
    CAE_ae_training_times = in_results["ae_training_times"]
    print("<< load cae model <<")

    #%% Loading Nonlinear POD Data
    print(">> load nlpod >>")

    nr = 34
    z_ae_nlpod = np.zeros([len(bcList), ens_size, nx, ny, ns])
    z_lstm_nlpod = np.zeros([len(bcList), ens_size, nx, ny, ns])
    latent_ae_nlpod = np.zeros([len(bcList), ens_size, latent_dim, ns])
    latent_lstm_nlpod = np.zeros([len(bcList), ens_size, latent_dim, ns])

    for j, bc in enumerate(bcList):
        filename = (
            "./Results/"
            + folder
            + "/NonPOD_data_bc="
            + str(int(bc))
            + "_nr="
            + str(nr)
            + ".npz"
        )
        nonPODdata = np.load(filename)
        tmp1 = nonPODdata["latent_ae"]
        tmp2 = nonPODdata["latent_lstm"]
        tmp3 = nonPODdata["z_ae"]
        tmp4 = nonPODdata["z_lstm"]
        for n in range(ns):
            latent_ae_nlpod[j, :, :, n] = tmp1[:, n, :]
            latent_lstm_nlpod[j, :, :, n] = tmp2[:, n, :]
            for m in range(ens_size):
                tmp = tmp3[m, :, n]
                z_ae_nlpod[j, m, :, :, n] = tmp.reshape([nx, ny])
                tmp = tmp4[m, :, n]
                z_lstm_nlpod[j, m, :, :, n] = tmp.reshape([nx, ny])
    print("<< load nlpod <<")

    #%% load NLPOD models
    print(">> load nlpod model >> (skiped)")
    make_keras_picklable()
    pickle_dir = "__PICKLES"
    loadfrom = os.path.join(pickle_dir, "NLPOD_Autencoders_nr=" + str(nr) + ".pickle")

    infile = open(loadfrom, "rb")
    in_results = pickle.load(infile)
    infile.close()

    NLPOD_encoders = in_results["encoders"]
    NLPOD_decoders = in_results["decoders"]
    NLPOD_autoencoders = in_results["autoencoders"]
    NLPOD_ae_train_losses = in_results["ae_train_losses"]
    NLPOD_ae_valid_losses = in_results["ae_valid_losses"]
    NLPOD_ae_training_times = in_results["ae_training_times"]
    print("<< load nlpod model <<")

    #%%
    print(">> coputing l2 error >>")
    err_cae_rec = np.zeros([len(bcList), ens_size])
    err_cae_pred = np.zeros([len(bcList), ens_size])
    err_nlpod_rec = np.zeros([len(bcList), ens_size])
    err_nlpod_pred = np.zeros([len(bcList), ens_size])

    for p in range(len(bcList)):
        for m in range(ens_size):
            err_cae_rec[p, m] = np.mean(
                (zfom[p, :, :, :] - z_ae_cae[p, m, :, :, :]) ** 2
            )
            err_cae_pred[p, m] = np.mean(
                (zfom[p, :, :, :] - z_lstm_cae[p, m, :, :, :]) ** 2
            )
            err_nlpod_rec[p, m] = np.mean(
                (zfom[p, :, :, :] - z_ae_nlpod[p, m, :, :, :]) ** 2
            )
            err_nlpod_pred[p, m] = np.mean(
                (zfom[p, :, :, :] - z_lstm_nlpod[p, m, :, :, :]) ** 2
            )

    err_cae_rec_mean = np.mean(err_cae_rec, axis=1)
    err_cae_pred_mean = np.mean(err_cae_pred, axis=1)
    err_nlpod_rec_mean = np.mean(err_nlpod_rec, axis=1)
    err_nlpod_pred_mean = np.mean(err_nlpod_pred, axis=1)

    err_cae_rec_std = np.std(err_cae_rec, axis=1)
    err_cae_pred_std = np.std(err_cae_pred, axis=1)
    err_nlpod_rec_std = np.std(err_nlpod_rec, axis=1)
    err_nlpod_pred_std = np.std(err_nlpod_pred, axis=1)
    print("<< coputing l2 error <<")

    #%%
    print(">> plot RIC (comulative proportional energy by mode) >>")
    nrplot = 50
    ncut1 = 34
    ncut0 = 2
    index = np.arange(1, nrplot + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(index, zric[:nrplot], color="k", linewidth=3)

    plt.plot([ncut0, ncut0], [zric[0], zric[ncut0 - 1]], color="k", linestyle="--")
    plt.plot([0, ncut0], [zric[ncut0 - 1], zric[ncut0 - 1]], color="k", linestyle="--")

    plt.plot([ncut1, ncut1], [zric[0], zric[ncut1 - 1]], color="k", linestyle="--")
    plt.plot([0, ncut1], [zric[ncut1 - 1], zric[ncut1 - 1]], color="k", linestyle="--")

    plt.fill_between(
        index[:ncut0], zric[:ncut0], y2=zric[0], alpha=0.6, color="orange"
    )  # ,
    plt.fill_between(index[:ncut1], zric[:ncut1], y2=zric[0], alpha=0.2, color="blue")

    plt.text(
        1.1,
        78,
        r"$" + str(np.round(zric[ncut0 - 1], decimals=2)) + "\%$",
        va="center",
        fontsize=18,
    )
    plt.text(
        5.1,
        101,
        r"$" + str(np.round(zric[ncut1 - 1], decimals=2)) + "\%$",
        va="center",
        fontsize=18,
    )
    plt.text(2.1, 68, r"$r=" + str(ncut0) + "$", va="center", fontsize=18)
    plt.text(78, 80, r"$r=" + str(ncut1) + "$", va="center", fontsize=18)

    plt.xlim(left=1, right=nrplot)
    plt.ylim(bottom=zric[0], top=104)
    plt.xscale("log")
    #%
    plt.xlabel(r"\bf POD index ($k$)")
    plt.ylabel(r"\bf RIC ($\%$)")

    if not os.path.exists("./Plots"):
        os.makedirs("./Plots")
    plt.savefig("./Plots/RIC.png", dpi=500, bbox_inches="tight")
    plt.savefig("./Plots/RIC.pdf", dpi=500, bbox_inches="tight")
    print("<< plot RIC (comulative proportional energy by mode) <<")

    #%%
    print(">> plot multiple conoutrs >> (skiped)")
    bc = 4
    nlvls = 30
    x_ticks = [0, 1]
    y_ticks = [0, 1]

    colormap = "RdBu_r"
    low = 0
    high = 2
    v = np.linspace(low, high, nlvls, endpoint=True)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for j, ind in enumerate([25, 50]):
        cs = axs[0][j].contourf(
            X,
            Y,
            zfom[bcList.index(bc), :, :, ind],
            v,
            cmap=colormap,
            linewidths=1,
            extend="both",
        )
        cs.set_clim([low, high])

        print(" ---->>>> z_lstm_cae.shape", z_lstm_cae.shape)
        cs = axs[1][j].contourf(
            X,
            Y,
            np.mean(z_lstm_cae[bcList.index(bc), :, :, :, ind], axis=0),
            v,
            cmap=colormap,
            linewidths=1,
            extend="both",
        )
        cs.set_clim([low, high])

        print(" ---->>>> z_lstm_nlpod.shape", z_lstm_nlpod.shape)
        cs = axs[2][j].contourf(
            X,
            Y,
            np.mean(z_lstm_nlpod[bcList.index(bc), :, :, :, ind], axis=0),
            v,
            cmap=colormap,
            linewidths=1,
            extend="both",
        )
        cs.set_clim([low, high])

        axs[2][j].set_xlabel("$x$")
        # for i in range(3): axs[i][j].set_ylabel("$y$")

    axs[0][0].set_title(r"$t=6$")
    axs[0][1].set_title(r"$t=8$")
    """
    plt.text(-11.5, 6.5, r"\bf{FOM}", va="center", fontsize=18)
    plt.text(-11.5, 4.85, r"\bf{($r=2$)}", va="center", fontsize=18)
    plt.text(-11.5, 3.35, r"\bf{($r=74$)}", va="center", fontsize=18)
    plt.text(-11.5, 2.1, r"\bf{CAE}", va="center", fontsize=18)
    plt.text(-11.5, 1.85, r"\bf{($r=2$)}", va="center", fontsize=18)
    plt.text(-11.5, 0.6, r"\bf{NLPOD}", va="center", fontsize=18)
    plt.text(-11.5, 0.35, r"\bf{($r=2$)}", va="center", fontsize=18)
    """

    fig.subplots_adjust(bottom=0.12, wspace=0.15, hspace=0.5)
    if not os.path.exists("./Plots"):
        os.makedirs("./Plots")
    plt.savefig("./Plots/contours.png", dpi=500, bbox_inches="tight")
    plt.savefig("./Plots/contours.pdf", dpi=500, bbox_inches="tight")
    print("<< plot multiple conoutrs <<")


main()
