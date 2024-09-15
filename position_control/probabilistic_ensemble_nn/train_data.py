# Infinite loop Truckmaker
from distutils.command.build_scripts import first_line_re
import time
import os
import numpy as np
import pandas as pd
import torch
from module import module
import random
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dynamics.nn_vehicle import ProbabilisticEnsembleNN

# ACTIVATION = 'softplus'
ACTIVATION = 'relu'

# Name or model and saving path
MODELNAME = 'data_generation_results_8datapoint_obs123_0819'
MODELNAME_SAVE = 'penn_model_0902'
data_file = 'data/' + MODELNAME + '.csv'
scaler_path = 'checkpoint/scaler_0902.save'
model_path = 'checkpoint/' + MODELNAME_SAVE + '.pth'

# Neural Network Paramters
device = 'cpu'
n_states = 6
n_output = 2
n_hidden = 40
n_ensemble = 3

LR = 0.0001
BATCHSIZE = 32
EPOCH = 30

def plot_gmm(gmm):
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() +
                    3, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, '-k', label='GMM')

    for i in range(pdf_individual.shape[1]):
        plt.plot(x, pdf_individual[:, i], '--', label=f'GMM Component {i+1}')

    plt.xlabel('Safety Loss Prediction')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model for Safety Loss Predictions')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Test = True
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the vehicle model
    penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

    if Test:
        penn.load_scaler(scaler_path)
        penn.load_model(model_path)
        
        # Example input array [distance, velocity, theta, gamma1, gamma2
        input_data = [0.55, 0.01, 0.001, 0.005, 0.005]
        y_pred_safety_loss, y_pred_deadlock_time, div = penn.predict(input_data)
        print("Predicted Safety Loss:", y_pred_safety_loss)
        print("Predicted Deadlock Time:", y_pred_deadlock_time)
        
        # Create GMM for safety loss predictions
        gmm_safety = penn.create_gmm(y_pred_safety_loss)
        plot_gmm(gmm_safety)
        
    else:
        # Load and preprocess data
        train_dataX, train_dataY, test_dataX, test_dataY = penn.load_and_preprocess_data(data_file, scaler_path, noise_percentage=3.0)

        # Create datasets and dataloaders
        train_dataset = module.CustomDataset(train_dataX, train_dataY)
        test_dataset = module.CustomDataset(test_dataX, test_dataY)
        train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

        
        start_epoch = 0
        best_test_rmse = 1000000
        start_time = time.time()
        for epoch in range(start_epoch, start_epoch + EPOCH):
            train_loss = penn.train(train_loader, epoch)
            test_loss, bool_best, test_rmse = penn.test(
                test_loader, epoch, verbose=False)
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                print('Saving... \n')
                state = {
                    'model': penn.state_dict(),
                    'test_rmse': test_rmse,
                    'epoch': epoch,
                    'input_state': n_states,
                }
                os.makedirs('checkpoint/', exist_ok=True)
                torch.save(penn.state_dict(
                ), 'checkpoint/' + MODELNAME_SAVE + '.pth')

        end_time = time.time()
        print('Learnig Time: {:.1f} min'.format((end_time-start_time)/60))
