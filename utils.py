
import os
import random
import numpy as np
import torch


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def mape(y_true, y_pred):
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 0.000001) ))
    return mape


def train_one_step(model, device, optimizer, scheduler, criterion, data_loader, log):
    # Do train mode
    model.train()

    avg_loss = 0.
    rmse = 0.
    cnt = 0
    for batch_idx, (x_data, y_data) in enumerate(data_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # Do prediction
        outputs = model(x_data)

        # get loss for the predicted output
        loss = criterion(outputs, y_data)

        avg_loss += loss.item()

        if log:
            outputs, y_data = torch.exp(outputs)-0.001, torch.exp(y_data)-0.001
        rmse += ((outputs.float() - y_data.float()) ** 2).sum().item()

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        cnt += 1

    scheduler.step()
    Y_ncols = data_loader.dataset[0][1].shape[0]

    rmse = np.sqrt(rmse / len(data_loader.dataset) / Y_ncols)
    avg_loss /= cnt

    return avg_loss, rmse


def eval_one_step(model, device, criterion, data_loader, exp, log):
    # Do Eval mode
    model.eval()

    y_data_list = []
    y_pred_list = []

    avg_loss = 0.
    rmse = 0.
    rmse_each = 0.
    cnt = 0
    with torch.no_grad():
        for batch_idx, (x_data, y_data) in enumerate(data_loader):
            x_data, y_data = x_data.to(device), y_data.to(device)

            # get output from the model, given the inputs
            outputs = model(x_data)

            # get loss for the predicted output
            loss = criterion(outputs, y_data)

            avg_loss += loss.item()

            if log:
                outputs, y_data = torch.exp(outputs) - 0.001, torch.exp(y_data) - 0.001
            rmse += ((outputs.float() - y_data.float()) ** 2).sum().item()

            if exp == 'EXP1':
                ##### CASE 1 2 x 2 ----------------------------------------------------------------------------------------
                rmse_each += ((outputs.float() - y_data.float()) ** 2).sum(axis=0).to('cpu').numpy()
            elif exp == 'EXP2':
                ##### CASE 2. 2 x 1 ----------------------------------------------------------------------------------------
                rmse_each += ((outputs.float() - y_data.float()) ** 2).sum().to('cpu').numpy()
            elif exp == 'EXP3':
                ##### CASE 3. 3 x 1 ----------------------------------------------------------------------------------------
                rmse_each += ((outputs.float() - y_data.float()) ** 2).sum().to('cpu').numpy()
            else:
                pass

            y_data_list.extend(y_data.to('cpu').tolist())
            y_pred_list.extend(outputs.to('cpu').tolist())

            cnt += 1

    Y_ncols = data_loader.dataset[0][1].shape[0]

    rmse = np.sqrt(rmse / len(data_loader.dataset) / Y_ncols)
    rmse_each = np.sqrt(rmse_each / len(data_loader.dataset))

    avg_loss /= cnt

    return avg_loss, rmse, rmse_each, np.array(y_data_list), np.array(y_pred_list)
