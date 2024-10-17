import pathlib
import torch
from torch.utils.data import DataLoader
from train_parameters import *
from src.loadDataset import *
from src.normalization import decodeOneHot
from src.model_utils import *
from src.errorAnalysis import compute_NMSE

if __name__ == '__main__':
    
    ## define parameters
    # set Young's modulus of base material
    E = 114.
    # set softmax temperature (higher value enforces larger exploration)
    t = 200.
    # set number of resamplings (higher number might find better predictions)
    passes = 10000
    # set number of stored predictions (with the lowest NMSE)
    stored_pred = 5

    # prediction data path
    # consider, e.g., bone samples from Colabella et al., 2017. 'Mimetization of the elastic properties of cancellous bone via a parameterized cellular material'
    dataPath_pred = 'data/prediction.csv'
    # create directory to store predicted lattices
    pathlib.Path('prediction').mkdir(exist_ok=True)

    ## load and preprocess data
    # load normalization (based on training dataset)
    F1_features_scaling, C_ort_scaling, C_scaling, V_scaling, C_hat_scaling = getSavedNormalization()
    pred_set = getDataset_pred(C_scaling,E,dataPath_pred)
    pred_set_loader = DataLoader(dataset=pred_set, num_workers=numWorkers, batch_size=len(pred_set))
    # Note: for test, batch_size=len(test_set) so that we load the entire test set at once
    C_target = next(iter(pred_set_loader))

    ## load models
    # load F1
    F1 = torch.load("models/F1.pt",map_location=device)
    F1.eval()
    # load F2
    F2 = torch.load("models/F2.pt",map_location=device)
    F2.eval()
    # load inverse model (G1 & G2)
    G1 = torch.load("models/G1.pt",map_location=device)
    G2 = torch.load("models/G2.pt",map_location=device)
    G1.eval(), G2.eval()

    # initialize nested lists to collect best predictions
    num_samples = len(C_target)
    top_C_target_pred_pred = [[] for i in range(num_samples)]
    top_full_target_pred = [[] for i in range(num_samples)]

    print('\n-------------------------------------\nPredict inverse designs:')

    with torch.no_grad():
        C_target = C_target.to(device)
        # repeat target lables to generate large variety of predictions
        C_target = C_target.repeat(1,passes).view(-1,21)
        # inverse prediction
        rho_U_target_pred, V_target_pred, R1_target_pred, R2_target_pred, topology_target_pred = invModel_output(G1,G2,C_target,t,'gumbel')
        # assemble F1 features based on output of inverse model
        F1_features_target_pred = torch.cat((rho_U_target_pred, topology_target_pred), dim=1)
        # forward prediction based on inversely designed lattice
        C_ort_target_pred_pred = F1(F1_features_target_pred)
        F2_features_target_pred_pred = assemble_F2_features(C_ort_target_pred_pred,R1_target_pred,V_target_pred,C_ort_scaling,method='6D')
        C_hat_target_pred_pred = F2(F2_features_target_pred_pred)
        C_target_pred_pred = rotate_C(C_hat_target_pred_pred, R2_target_pred, C_hat_scaling, C_scaling,method='6D')
        # assemble full descriptor
        full_target_pred = torch.cat((F1_features_target_pred,R1_target_pred,R2_target_pred,V_target_pred),dim=1)
        # scale stiffness to original range and compute NMSE
        C_target = C_scaling.unnormalize(C_target)
        C_target_pred_pred = C_scaling.unnormalize(C_target_pred_pred)
        rel_error = compute_NMSE(C_target,C_target_pred_pred)
        # collect lattices with lowest NMSE
        lowest_error = torch.zeros(num_samples,device=device)+1.e20
        for j in range(num_samples):
            for i in range(passes):
                cur_iter = j*passes + i
                if (rel_error[cur_iter] < lowest_error[j]):
                    print('Identified lattice with lower NMSE.')
                    top_full_target_pred[j].append(full_target_pred[cur_iter])
                    top_C_target_pred_pred[j].append(C_target_pred_pred[cur_iter])
                    lowest_error[j] = rel_error[cur_iter]
        print('Finished.')

        # select the n best lattices with lowest NMSE and sort
        selected_full_target_pred = torch.zeros((num_samples,stored_pred,46+1),device=device)
        for i, list in enumerate(top_full_target_pred):
            temp = torch.stack(list[:-stored_pred-1:-1])
            num_predictions = temp.shape[0]
            temp = torch.cat((torch.zeros(num_predictions,1,device=device)+i+1,temp),dim=1)
            selected_full_target_pred[i,0:temp.shape[0]] = temp

        # select the n best lattice stiffnesses corresponding to the best designs
        selected_C_target_pred_pred = torch.zeros((num_samples,stored_pred,21+1),device=device)
        for i, list in enumerate(top_C_target_pred_pred):
            temp = torch.stack(list[:-stored_pred-1:-1])
            num_predictions = temp.shape[0]
            temp = torch.cat((torch.zeros(num_predictions,1,device=device)+i+1,temp),dim=1)
            selected_C_target_pred_pred[i,0:num_predictions] = temp

        # flatten and delete zero-rows
        selected_full_target_pred = torch.flatten(selected_full_target_pred,end_dim=1)
        selected_full_target_pred = selected_full_target_pred[selected_full_target_pred[:,0]!=0]
        selected_C_target_pred_pred = torch.flatten(selected_C_target_pred_pred,end_dim=1)
        selected_C_target_pred_pred = selected_C_target_pred_pred[selected_C_target_pred_pred[:,0]!=0]

        ## export for post-processing
        print('\n-------------------------------------\nExport predictions to:')

        # split prediction into subparts to rescale them to original range
        sample, rho_U_target_pred, topology_target_pred, R1_target_pred, R2_target_pred, V_target_pred = torch.split(selected_full_target_pred, [1,4,27,6,6,3], dim=1)

        # decode one-hot-encoding into original lattice nomenclature
        topology_target_pred = decodeOneHot(topology_target_pred)
        F1_features_target_pred = torch.cat((rho_U_target_pred,topology_target_pred),dim=1)

        # decode 6D representation into angle-axis representation
        R1_target_pred_angle_axis = rot6DToAngleAxis(R1_target_pred)
        R2_target_pred_angle_axis = rot6DToAngleAxis(R2_target_pred)
        
        # scale data to original range
        F1_features_target_pred = F1_features_scaling.unnormalize(F1_features_target_pred)
        V_target_pred = V_scaling.unnormalize(V_target_pred)

        # construct full descriptor of inversely designed lattice
        full_pred = torch.cat((sample,F1_features_target_pred,R1_target_pred_angle_axis,R2_target_pred_angle_axis,V_target_pred),dim=1)

        # add sample index to C_target
        C_target = torch.unique_consecutive(C_target, dim=0)
        C_target = torch.cat((torch.unsqueeze(torch.tensor(np.arange(num_samples)+1,device=device),1),C_target),dim=1)

        # unnormalize stiffness by Young's modulus of base material
        C_target[:,1:] *= E
        selected_C_target_pred_pred[:,1:] *= E

        # push tensors back to cpu
        full_pred = full_pred.cpu()
        C_target = C_target.cpu()
        selected_C_target_pred_pred = selected_C_target_pred_pred.cpu()
        
        # export tensors for post-processing
        exportTensor('prediction/full_pred',full_pred,['sample']+all_names)
        exportTensor('prediction/C_target',C_target,['sample']+C_names)
        exportTensor('prediction/C_target_pred_pred',selected_C_target_pred_pred,['sample']+C_names)