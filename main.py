import time
import os
import shutil
import sys
import argparse
import random
import numpy as np
from datetime import datetime
import math

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import data.dataloader as dataloader
from models.model import Regression
from utils import *

# EXP Code
# EXP1 : Z, ZDR --> Dm, W
# EXP2 : Z, ZDR --> R
# EXP3 : Z, ZDR, KDP --> R


# MODEL ARCHITECTURES
MODEL_ARCH = [
    [16, 32, 64, 128, 64, 32, 16],
    [16, 32, 64, 32, 16],
    [16, 32, 16],
    [8, 16, 32, 64, 32, 16, 8],
    [8, 16, 32, 16, 8],
    [8, 16, 8],
    [4, 8, 16, 32, 16, 8, 4],
    [4, 8, 16, 8, 4],
    [4, 8, 4],
    [4, 4, 4],
    [4, 6, 4, 2],
    [6, 4, 2],
    [6, 6, 6],
    [6, 8, 6],
]


def main(args, time_stamp):
    # Set the output folder
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    os.mkdir(os.path.join('./results', time_stamp))

    # Open the result file
    fw = open(os.path.join('./results', time_stamp, args.exp+'_output.txt'), 'w')

    print('Parameter Settings ------------------------------------------')
    for k in args.__dict__:
        print(k+'\t'+str(args.__dict__[k]))
        fw.write(k+'\t'+str(args.__dict__[k])+'\n')
    print('Time stamp' + '\t' + str(time_stamp))
    fw.write('Time stamp' + '\t' + str(time_stamp)+'\n')
    print()
    fw.write('\n')

    # Save the arguments
    if args.exp == 'EXP1':
        ##### CASE 1. 2 x 2 ----------------------------------------------------------------------------------------
        # X1 : Z, X2 : ZDR
        # Y1 : Dm, Y2 : W
        fw.write('EXP_CODE'+'\t'+'Epoch'+'\t'+'TRN_RMSE'+'\t'+'VAL_RMSE'+'\t'+'TST_RMSE'+'\t'+'Dm'+'\t'+'W'+'\n')

    elif args.exp == 'EXP2':
        ##### CASE 2. 2 x 1 ----------------------------------------------------------------------------------------
        # X1 : Z, X2 : ZDR
        # Y : R
        fw.write('EXP_CODE'+'\t'+'Epoch'+'\t'+'TRN_RMSE'+'\t'+'VAL_RMSE'+'\t'+'TST_RMSE'+'\t'+'R'+'\n')

    elif args.exp == 'EXP3':
        ##### CASE 3. 3 x 1 ----------------------------------------------------------------------------------------
        # X1 : Z, X2 : ZDR, X3: KDP
        # Y : R
        fw.write('EXP_CODE'+'\t'+'Epoch'+'\t'+'TRN_RMSE'+'\t'+'VAL_RMSE'+'\t'+'TST_RMSE'+'\t'+'R'+'\n')

    else:
        print('Experiment mode error! Check the experiment mode : ', args.exp)
        sys.exit(-1)

    # Loading Dataset
    train_data = dataloader.DSD_radar_dataset(args.data_path, args.trn_file, mode='trn', exp=args.exp, log=args.log)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    if os.path.exists(os.path.join(args.data_path, args.val_file)):
        val_data = dataloader.DSD_radar_dataset(args.data_path, args.val_file, mode='val', exp=args.exp, log=args.log,
                                                X_scaler=train_data.X_scaler, Y_scaler=train_data.Y_scaler)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, pin_memory=True)
    else:
        val_loader = None

    if os.path.exists(os.path.join(args.data_path, args.tst_file)):
        tst_data = dataloader.DSD_radar_dataset(args.data_path, args.tst_file, mode='tst', exp=args.exp, log=args.log,
                                                X_scaler=train_data.X_scaler, Y_scaler=train_data.Y_scaler)
        tst_loader = torch.utils.data.DataLoader(tst_data, batch_size=args.batch_size, pin_memory=True)
    else:
        tst_loader = None

    # For each model
    for m_idx, m_arch in enumerate(MODEL_ARCH):
        EXP_CODE = '-'.join([str(x) for x in m_arch])

        print('\n -- MODEL ID : %d / ARCHITECTURE : %s --------------------------------------------' % (m_idx, m_arch))
        # Loading Model
        model = Regression(dIn=train_data.get_x_shape()[1], dOut=train_data.get_y_shape()[1], nNeurons=m_arch)
        print('Model :', model)

        # Set the Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device : ', device)
        model.to(device)

        # Init Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Decaying the learning rate
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        criterion = torch.nn.MSELoss()
        #criterion = torch.nn.L1Loss()

        # Do training
        best_iter = -1
        best_val_loss = 0
        for i_iter in range(0, args.epoch):
            # Training
            trn_loss, trn_rmse = train_one_step(model, device, optimizer, scheduler, criterion, train_loader, args.log)

            # Validation
            val_loss, val_rmse, val_rmse_each, val_y_data_list, val_y_pred_list = eval_one_step(model, device, criterion, val_loader, args.exp, args.log)

            # Check
            if i_iter == 0 or best_val_loss > val_loss:
                # Save the best model
                best_iter = i_iter
                best_val_loss = val_loss

                save_path = os.path.join('./results', time_stamp, EXP_CODE + '_Model_' + str(best_iter) + '.pth')
                torch.save({
                    'epoch': best_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rmse': val_rmse
                }, save_path)

                # Test
                tst_loss, tst_rmse, tst_rmse_each, tst_y_data_list, tst_y_pred_list = eval_one_step(model, device, criterion, tst_loader, args.exp, args.log)

                if args.exp == 'EXP1':
                    print('Epoch %3d | Trn Loss %7.5f RMSE %7.5f | Eval Loss %7.5f RMSE %7.5f | Test Loss %7.5f RMSE %7.5f ( Dm %7.5f, W %7.5f )'
                          % (i_iter, trn_loss, trn_rmse, val_loss, val_rmse, tst_loss, tst_rmse, tst_rmse_each[0], tst_rmse_each[1]))
                    fw.write('%s\t%d\t%7.5f\t%7.5f\t%7.5f\t%7.5f\t%7.5f\n' % (EXP_CODE, i_iter, trn_rmse, val_rmse, tst_rmse, tst_rmse_each[0], tst_rmse_each[1]))
                elif args.exp == 'EXP2':
                    print('Epoch %3d | Trn Loss %7.5f RMSE %7.5f | Eval Loss %7.5f RMSE %7.5f | Test Loss %7.5f RMSE %7.5f ( R %7.5f )'
                          % (i_iter, trn_loss, trn_rmse, val_loss, val_rmse, tst_loss, tst_rmse, tst_rmse_each))
                    fw.write('%s\t%d\t%7.5f\t%7.5f\t%7.5f\t%7.5f\n' % (EXP_CODE, i_iter, trn_rmse, val_rmse, tst_rmse, tst_rmse_each))
                elif args.exp == 'EXP3':
                    print('Epoch %3d | Trn Loss %7.5f RMSE %7.5f | Eval Loss %7.5f RMSE %7.5f | Test Loss %7.5f RMSE %7.5f ( R %7.5f )'
                          % (i_iter, trn_loss, trn_rmse, val_loss, val_rmse, tst_loss, tst_rmse, tst_rmse_each))
                    fw.write('%s\t%d\t%7.5f\t%7.5f\t%7.5f\t%7.5f\n' % (EXP_CODE, i_iter, trn_rmse, val_rmse, tst_rmse, tst_rmse_each))
                else:
                    pass
            else:
                print('Epoch %3d | Trn Loss %7.5f RMSE %7.5f | Eval Loss %7.5f RMSE %7.5f'
                      % (i_iter, trn_loss, trn_rmse, val_loss, val_rmse))

    fw.close()


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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epoch', type=float, default=120, help='# of epochs')
    parser.add_argument('--manualSeed', type=int, required=False, default=2020, help='for experiment reproduction')
    args = parser.parse_args()

    # Set the seed for reproduce
    seed_torch(args.manualSeed)

    # main
    main(args, time_stamp)

    elapsed_time = time.time() - start_time
    print('\nProgram terminated : %s' % datetime.now())
    print('Duration : %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

"""
screen -r DSD_0.01
source activate yolov3
python -m main --exp=EXP3 --lr=0.01

screen -r DSD_0.001
source activate yolov3
python -m main --exp=EXP3 --lr=0.001

screen -r DSD_0.0001
source activate yolov3
python -m main --exp=EXP3 --lr=0.0001

screen -r DSD_0.00001
source activate yolov3
python -m main --exp=EXP3 --lr=0.00001

"""