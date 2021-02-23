import time
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch

import data.dataloader as dataloader
from models.model import Regression
from utils import *


def main(args, time_stamp):
    # Set the output folder
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    os.mkdir(os.path.join('./results', time_stamp))

    # Loading Dataset
    train_data = dataloader.DSD_radar_dataset(args.data_path, args.trn_file, mode='trn', exp=args.exp, log=args.log,)

    assert os.path.exists(os.path.join(args.data_path, args.tst_file))

    tst_data = dataloader.DSD_radar_dataset(args.data_path, args.tst_file, mode='tst', exp=args.exp, log=args.log,
                                            X_scaler=train_data.X_scaler, Y_scaler=train_data.Y_scaler)
    tst_loader = torch.utils.data.DataLoader(tst_data, batch_size=1, pin_memory=True)

    model_arch = args.model_file.split('_')[0]
    model_arch = [int(x) for x in model_arch.split('-')]

    print('\n -- MODEL ARCHITECTURE : %s --------------------------------------------' % (model_arch))
    # Loading Model
    model = Regression(dIn=train_data.get_x_shape()[1], dOut=train_data.get_y_shape()[1], nNeurons=model_arch)
    print('Model :', model)

    # Set the Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device : ', device)
    model.to(device)

    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()

    # Load model
    best_model_path = os.path.join('./results', args.timestamp, args.model_file)
    print('Loading Pretrained model : ', best_model_path)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Do test
    tst_loss, tst_rmse, tst_rmse_each, tst_y_data_list, tst_y_pred_list = eval_one_step(model, device, criterion, tst_loader, args.exp,  args.log)

    if args.exp == 'EXP1':
        COLs = ['Dm', 'W']
        print('Test Loss %7.5f RMSE %7.5f ( Dm %7.5f, W %7.5f )'
              % (tst_loss, tst_rmse, tst_rmse_each[0], tst_rmse_each[1]))
    elif args.exp == 'EXP2':
        COLs = ['R']
        print('Test Loss %7.5f RMSE %7.5f ( R %7.5f )'
              % (tst_loss, tst_rmse, tst_rmse_each))
    elif args.exp == 'EXP3':
        COLs = ['R']
        print('Test Loss %7.5f RMSE %7.5f ( R %7.5f )'
              % (tst_loss, tst_rmse, tst_rmse_each))
    else:
        pass

    # Save the prediction results
    print('Saving prediction results')
    for c_idx in range(tst_y_data_list.shape[1]):
        fw = open(os.path.join('./results', time_stamp, 'Predict_'+COLs[c_idx]+'_.txt'), 'w')
        fw.write('Actual'+'\t'+'Predicted'+'\n')
        for r_idx in range(tst_y_data_list.shape[0]):
            fw.write('%f\t%f\n' % (tst_y_data_list[r_idx][c_idx], tst_y_pred_list[r_idx][c_idx]))

    # Draw the prediction results
    print('Draw the prediction results')
    fig, axs = plt.subplots(1, len(COLs), constrained_layout=True, figsize=(7 * len(COLs), 7))

    if len(COLs) > 1:
        for c_idx in range(len(COLs)):
            axs[c_idx].scatter(tst_y_data_list[:, c_idx], tst_y_pred_list[:, c_idx])
            axs[c_idx].plot(tst_y_data_list[:, c_idx], tst_y_data_list[:, c_idx], linestyle='dashed', linewidth=0.5, color='black')
            axs[c_idx].set_title('{} (RMSE: {:.4f})'.format(COLs[c_idx], tst_rmse_each[c_idx]))
            axs[c_idx].set_xlabel('Actual')
            axs[c_idx].set_ylabel('Predicted')
            axs[c_idx].grid(True)
    else:
        axs.scatter(tst_y_data_list, tst_y_pred_list)
        axs.plot(tst_y_data_list, tst_y_data_list, linestyle='dashed', linewidth=0.5, color='black')
        axs.set_title('{} (RMSE: {:.4f})'.format(COLs[0], tst_rmse_each))
        axs.set_xlabel('Actual')
        axs.set_ylabel('Predicted')
        axs.grid(True)

    plt.savefig(os.path.join('./results', time_stamp, args.exp+'_Predicted_Graph.png'), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    start_time = time.time()
    start_dt = datetime.now()
    time_stamp = start_dt.strftime('%Y%m%d%H%M%S')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', required=False, default='./data', help='Path to files')
    parser.add_argument('--exp', required=True, help='Experiment mode')
    parser.add_argument('--log', action='store_true', help='Logarithm for both X and Y values')
    parser.add_argument('--trn_file', required=False, default='trn_data.mat', help='Training dataset name')
    parser.add_argument('--val_file', required=False, default='val_data.mat', help='Validation dataset name')
    parser.add_argument('--tst_file', required=False, default='test_data.mat', help='Test dataset name')
    parser.add_argument('--timestamp', required=True, help='Datetime when the model trained')
    parser.add_argument('--model_file', required=True, help='Model file')
    parser.add_argument('--manualSeed', type=int, required=False, default=2020, help='for experiment reproduction')
    args = parser.parse_args()

    # Set the seed for reproduce
    seed_torch(args.manualSeed)

    # main
    main(args, time_stamp)

    elapsed_time = time.time() - start_time
    print('\nProgram terminated : %s' % datetime.now())
    print('Duration : %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# python -m test --timestamp=20210216202222 --model_file=16-32-64-32-16_Model_76.pth

# EXP2
# python -m test --exp=EXP2 --timestamp=20210220140148 --model_file=8-16-8_Model_44.pth

# EXP3
# python -m test --exp=EXP3 --timestamp=20210220165627 --model_file=8-16-8_Model_114.pth