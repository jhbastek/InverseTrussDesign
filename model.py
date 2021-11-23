import torch
import torch.nn.functional as F
from parameters import *
from voigt_rotation import *

def getActivation(activ):
    if(activ == 'relu'):
        sigma = torch.nn.ReLU()
    elif(activ == 'tanh'):
        sigma = torch.nn.Tanh()
    elif(activ == 'sigmoid'):
        sigma = torch.nn.Sigmoid()
    elif(activ == 'leaky'):
        sigma = torch.nn.LeakyReLU()
    elif(activ == 'softplus'):
        sigma = torch.nn.Softplus()
    elif(activ == 'logsigmoid'):
        sigma = torch.nn.LogSigmoid()
    elif(activ == 'elu'):
        sigma = torch.nn.ELU()
    elif(activ == 'gelu'):
        sigma = torch.nn.GELU()
    elif(activ == 'none'):
        sigma = torch.nn.Identity()
    else:
        raise ValueError('Incorrect activation function')
    return sigma

def createNN(inputDim,arch,outputDim,bias=True):
    model = torch.nn.Sequential()
    currDim = inputDim
    layerCount = 1
    activCount = 1
    for i in range(len(arch)):
        if(type(arch[i]) == int):
            model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,arch[i],bias=bias))
            currDim = arch[i]
            layerCount += 1
        elif(type(arch[i]) == str):
            model.add_module('activ '+str(activCount),getActivation(arch[i]))
            activCount += 1
    model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,outputDim,bias=bias))
    return model

def softmax(input, t):
    return F.log_softmax(input/t, dim=1)

def gumbel(input, t):
    return F.gumbel_softmax(input, tau=t, hard=True, eps=1e-10, dim=1)

def getOrthogonalStiffness(F1,F1_features,C_ort_scaling):
    return C_ort_scaling.unnormalize(F1(F1_features))
    
def assemble_F2_features(C_ort,R1,V,C_ort_scaling,method=None):
    # scale C_ort to its original range to compute R1
    C_ort_unscaled = C_ort_scaling.unnormalize(C_ort)
    # rotate C_ort (directly in Voigt notation)
    if method == '6D':
        C_tilde = direct_rotate_6D(C_ort_unscaled, R1)
        print('FLAG')
    else:
        C_tilde = direct_rotate(C_ort_unscaled, R1)
    return torch.cat((C_tilde,V),dim=1)

# def assemble_F2_features_6D(C_ort,R,V,C_ort_scaling):
#     # scale C_ort to its original range to compute R
#     C_ort_unscaled = C_ort_scaling.unnormalize(C_ort)
#     # rotate C_ort (directly in Voigt notation)
#     C_tilde = direct_rotate_6D(C_ort_unscaled, R)
#     return torch.cat((C_tilde,V),dim=1)

# def assemble_F2_features_old(F1,features,R,shear_features,C_ort_scaling):
#     # obtain 9 orthogonal stiffness parameters from first model
#     orthogonal_stiffness = getOrthogonalStiffness(F1,features,C_ort_scaling)
#     # rotate orthogonal stiffness matrix using direct Voigt matrix multiplication
#     rotated_stiffness = direct_rotate(orthogonal_stiffness, R)
#     return torch.cat((rotated_stiffness,shear_features),dim=1)

# def assemble_F2_features_6D_old(F1,features,R,shear_features,C_ort_scaling):
#     # obtain 9 orthogonal stiffness parameters from first model
#     orthogonal_stiffness = getOrthogonalStiffness(F1,features,C_ort_scaling)
#     # rotate orthogonal stiffness matrix using direct Voigt matrix multiplication
#     rotated_stiffness = direct_rotate_6D(orthogonal_stiffness, R)
#     return torch.cat((rotated_stiffness,shear_features),dim=1)

def invModel_output(G1,G2,input,t,activation):
    # continuous params: [stretch1, stretch2, stretch3, rot_stretch1, rot_stretch2, rot_stretch3, theta, rot_ax1, rot_ax2]
    topology1,topology2,topology3,rep1,rep2,rep3 = torch.split(G1(input), [7,7,7,2,2,2], dim=1)
    m = getActivation('sigmoid')
    if(activation == 'one-hot'):
        t = 1.e-6
    if(activation == 'softmax' or activation == 'one-hot'):
        topology = torch.cat((softmax(topology1,t),softmax(topology2,t),softmax(topology3,t),softmax(rep1,t),softmax(rep2,t),softmax(rep3,t)), dim=1)
    elif(activation == 'gumbel'):
        topology1,topology2,topology3,rep1,rep2,rep3 = softmax(topology1,t),softmax(topology2,t),softmax(topology3,t),softmax(rep1,t),softmax(rep2,t),softmax(rep3,t)
        topology = torch.cat((gumbel(topology1,t),gumbel(topology2,t),gumbel(topology3,t),gumbel(rep1,t),gumbel(rep2,t),gumbel(rep3,t)), dim=1)
    else:
        raise ValueError('Incorrect activation function')

    features = torch.cat((topology, input), dim=1)
    rho_U, V, rot1, rot2 = torch.split(G2(features), [4,3,6,6], dim=1)
    # scale to range using sigmoid
    rho_U, V = m(rho_U), m(V)

    return rho_U, V, rot1, rot2, topology
    
# input: normalized rotated C, output: normalized un-rotated C
def rotate_C(C_hat,R,C_scaling,C_hat_scaling,method=None):
    temp = C_hat_scaling.unnormalize(C_hat)
    if method == '6D':
        temp = direct_rotate_6D_full(temp,R)
    else:
        temp = direct_rotate_full(temp,R)
    C = C_scaling.normalize(temp)
    return C

# # input: normalized rotated C, output: normalized un-rotated C
# def backrotate_C(C,R,C_scaling,C_hat_scaling):
#     temp = C_scaling.unnormalize(C)
#     a,b,c = torch.split(R,[1,1,1],dim=1)
#     R = torch.cat((-a,b,c),dim=1)
#     temp = direct_rotate_full(temp,R)
#     C_hat = C_hat_scaling.normalize(temp)
#     return C_hat

# # input: normalized rotated C, output: normalized un-rotated C
# def rotate_C_6D(C,R,C_scaling,C_hat_scaling):
#     temp = C_hat_scaling.unnormalize(C)
#     temp = direct_rotate_6D_full(temp,R)
#     rotated_C = C_scaling.normalize(temp)
#     return rotated_C