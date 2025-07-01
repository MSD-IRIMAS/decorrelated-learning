import numpy as np
import pandas as pd
import sys
import os
import json
import argparse
from distutils.util import strtobool

from utils import load_data, preprocess_data, create_directory, plot_loss_and_acc_curves

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

from lite import LITE
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)

import copy
import time


def get_args():

  
    dataset_names = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                     'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                     'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                     'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                     'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                     'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                     'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                     'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                     'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                     'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                     'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                     'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                     'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                     'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                     'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                     'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                     'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                     'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                     'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                     'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                     'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                     'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                     'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                     'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                     'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                     'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                     'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                     'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                     'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'Wine', 'WordSynonyms', 'Yoga', 
                     'UWaveGestureLibraryZ', 'Wafer',  'Worms', 'WormsTwoClass',]
   
    parser = argparse.ArgumentParser(description='Choose to apply which classifier on which dataset with number of runs')

    parser.add_argument('--classifier', help='which classifier to use', type=str, choices=['LITE'], default='LITE', )

    parser.add_argument('--datasets',  help='which dataset to run the experiment on.', type=str, default=dataset_names,)
    
    parser.add_argument('--runs', help='number of runs to do', type=int, default=1)

    parser.add_argument('--output-directory', help='output directory parent', type=str, default='results_new/co_models_3/',)

    args = parser.parse_args()

    return args

def train_alone_model(model, base_model_1, base_model_2, epoch):
    print('\n\nTraining epoch: ', epoch)
    model.train()    
    base_model_1.eval()    
    base_model_2.eval()    
    

    correct, total = 0, 0
    loss_total, loss_ce, loss_div_feat = 0, 0, 0

    pool_size = 1
    avg_pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        outputs, features_co = model.extract_features(inputs.float())
        _, features_base_1 = base_model_1.extract_features(inputs.float())
        _, features_base_2 = base_model_2.extract_features(inputs.float())
        
        last_layer_feat_co = features_co[-1]
        last_layer_feat_base_1 = features_base_1[-1]   
        last_layer_feat_base_2 = features_base_2[-1]   
        
             

        dot_product = torch.matmul(last_layer_feat_co, last_layer_feat_base_1.transpose(2, 1).detach())        
        num_features = last_layer_feat_co.size(1)
        mask = 1 - torch.eye(num_features, device=dot_product.device).unsqueeze(0)
        dot_prod_no_diog = dot_product * mask
        
        penalty_1 = torch.mean(torch.norm(dot_prod_no_diog, p='fro', dim=(1, 2)) / (torch.count_nonzero(dot_prod_no_diog, dim=(1,2)) +1 )) / 40        

        dot_product_2 = torch.matmul(last_layer_feat_co, last_layer_feat_base_2.transpose(2, 1).detach())        
        mask_2 = 1 - torch.eye(num_features, device=dot_product.device).unsqueeze(0)
        dot_prod_no_diog_2 = dot_product_2 * mask_2
        
        penalty_2 = torch.mean(torch.norm(dot_prod_no_diog_2, p='fro', dim=(1, 2)) / (torch.count_nonzero(dot_prod_no_diog_2, dim=(1,2)) +1 )) / 40        

        penalty = (penalty_1 + penalty_2) / 2



        loss_ce_ = criterion_CE(outputs, targets)
        loss =  0.8 * loss_ce_ + 0.2 * penalty

        loss.backward()
        optimizer.step()        

        loss_div_feat += penalty
        loss_ce += loss_ce_
        loss_total += loss

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'  % (loss_ce / (batch_idx + 1), 100. * correct / total, correct, total))
    print(loss_div_feat.item() )
    
    return loss_total.item() / (batch_idx + 1), loss_ce.item() / (batch_idx + 1), loss_div_feat.item() / (batch_idx + 1), 100. * correct / total


def test(model):
    
    model.eval()
    test_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(valloader):
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs, _ = model(inputs.float())
        loss = criterion_CE(outputs, targets)
        
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (batch_idx + 1), 100. * correct / total

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = get_args()
    # run 1                     # run 2                   # run 3                   # run 4                  # run 5
    # Set seed -> 9409,         Set seed -> 2344,         Set seed -> 7045          Set seed -> 4714         Set seed -> 37
    # Set seed -> 3804,         Set seed -> 1295,         Set seed -> 2820          Set seed -> 1357         Set seed -> 5639
    # Set seed -> 3952,         Set seed -> 6258,         Set seed -> 7541          Set seed -> 5781         Set seed -> 8976
    # Set seed -> 2561,         Set seed -> 6742,         Set seed -> 8168          Set seed -> 879          Set seed -> 8859
    # Set seed -> 5296,         Set seed -> 3319,         Set seed -> 1520          Set seed -> 6789         Set seed -> 9876
                                                                                                            
    base_seeds_1 = [37, 5639, 8976, 8859, 9876]   
    base_seeds_2 = [5639, 8976, 8859, 9876, 37]   
    co_seeds = [8976, 8859, 9876, 37, 5639]    
     
    
     
    for base_seed_1, base_seed_2, co_seed in zip(base_seeds_1, base_seeds_2, co_seeds):

        for dataset in args.datasets:
            
            output_directory = args.output_directory

            xtrain, ytrain, xtest, ytest = load_data(file_name=dataset)        
            length_TS = int(xtrain.shape[1])
            
            trainloader = preprocess_data(xtrain, ytrain, shuffle=True)
            valloader = preprocess_data(xtest, ytest, shuffle=False)

            output_directory = os.getcwd() + '/' + output_directory + f'co_seed_{co_seed}_base_seed_{base_seed_2}_{base_seed_1}/' + dataset + '/'
            create_directory(output_directory)

            set_seeds(co_seed)
            clf = LITE(output_directory=output_directory, length_TS=length_TS, n_classes=len(np.unique(ytrain)), n_filters=32)
            clf.cuda()
            clf.train()
            
            first_model_wts = copy.deepcopy(clf.state_dict())
            torch.save(first_model_wts, output_directory + f'first_model.pt')

            criterion_CE = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(clf.parameters(), )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-4)

            base_model_1 = LITE(output_directory=None, length_TS=length_TS, n_classes=len(np.unique(ytrain)), n_filters=32)
            base_model_1.load_state_dict(torch.load(f'results_oldd/base_seed_{base_seed_1}/' + dataset + '/best_model.pt'))
            base_model_1.cuda()
            base_model_1.eval()
    
            base_model_2 = LITE(output_directory=None, length_TS=length_TS, n_classes=len(np.unique(ytrain)), n_filters=32)
            base_model_2.load_state_dict(torch.load(f'results_new/co_models_2/co_seed_{base_seed_2}_base_seed_{base_seed_1}/' + dataset + '/best_model.pt'))
            base_model_2.cuda()
            base_model_2.eval()
    

            use_cuda = torch.cuda.is_available()
            print('use_cuda: ', use_cuda)

            if use_cuda:
                torch.cuda.set_device(0)
                                            
            print('length_TS ', length_TS)
            print('Base model performance before training: ')
            # val_loss, val_acc = test(base_model_1)


            min_train_loss = np.inf
            final_loss, learning_rates = [], []
            training_loss, training_acc = [], []
            val_losses, val_acces = [], []

            best_model_epoch = 0
            ce_loss_all, div_loss_all = [], []
            prev_ce_loss, lambda_, min_lambda_ = np.inf, 1, 0.9
            
            epochs = 1500
            start_time = time.time()
            for epoch in range(epochs):
                
                train_loss, ce_loss, loss_div_feat, train_acc = train_alone_model(clf, base_model_1, base_model_2, epoch)
                                            
                ce_loss_all.append(ce_loss)
                div_loss_all.append(loss_div_feat)
                
                print('Train loss: ', train_loss)
                print('CE loss: ', ce_loss)
                print('Diversity loss: ', loss_div_feat)

                training_loss.append(train_loss)
                training_acc.append(train_acc)

                val_loss, val_acc = test(clf)

                val_losses.append(val_loss)
                val_acces.append(val_acc)


                if min_train_loss >= ce_loss and epoch > 1100:
                    min_train_loss = ce_loss
                    best_model_wts = copy.deepcopy(clf.state_dict())
                    best_model_epoch = epoch

                scheduler.step(train_loss)
                learning_rates.append(optimizer.param_groups[0]['lr'])
                print('learning rate: ', optimizer.param_groups[0]['lr'])

                

            last_model_wts = copy.deepcopy(clf.state_dict())
            torch.save(best_model_wts, output_directory +  'best_model.pt')
            torch.save(last_model_wts, output_directory +  'last_model.pt')




            # Save Logs
            plot_loss_and_acc_curves(training_loss, val_losses, training_acc, val_acces, output_directory)

            duration = time.time() - start_time
            best_model = LITE(
                                output_directory=output_directory,
                                length_TS=length_TS,
                                n_classes=len(np.unique(ytrain)),
                                n_filters=32,
                            )

            best_model.load_state_dict(best_model_wts)
            best_model.cuda()
            
            print('Best Model Accuracy in below ')
            print('best model epoch: ', best_model_epoch)
            
            start_test_time = time.time()
            test(best_model)
            test_duration = time.time() - start_test_time
            print(test(best_model))
            df = pd.DataFrame(list(zip(training_loss, learning_rates)), columns =['loss', 'learning_rate'])
            index_best_model = df['loss'].idxmin()
            row_best_model = df.loc[index_best_model]
            df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

            df.to_csv(output_directory + 'history.csv', index=False)
            df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

            loss_, acc_ = test(best_model)
            df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
            df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False) 
            
            df_losses = pd.DataFrame(list(zip(ce_loss_all, div_loss_all)), columns=['ce_losses', 'div_losses'])
            df_losses.to_csv(output_directory + 'df_losses.csv', index=False) 

        
