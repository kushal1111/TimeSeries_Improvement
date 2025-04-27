import json
from data_provider.data_factory import data_provider
from torch.utils.data import Subset, DataLoader
from torch.optim import lr_scheduler
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, AverageMeter
from utils.metrics import metric
from torch import optim
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(actual, predicted, timestamps=None, title="Prediction vs Actual", save_path=None):
    """
    Plot actual vs predicted values with professional formatting
    
    Args:
        actual: numpy array of shape [seq_len]
        predicted: numpy array of shape [seq_len] 
        timestamps: optional time axis labels
        title: plot title
        save_path: where to save the image (None to display)
    """
    plt.figure(figsize=(12, 6))
    
    if timestamps is None:
        plt.plot(actual, label='Actual', color='#1f77b4', linewidth=2)
        plt.plot(predicted, '--', label='Predicted', color='#ff7f0e', linewidth=2)
    else:
        plt.plot(timestamps, actual, label='Actual', color='#1f77b4', linewidth=2)
        plt.plot(timestamps, predicted, '--', label='Predicted', color='#ff7f0e', linewidth=2)
    
    plt.fill_between(
        range(len(actual)),
        actual - 0.2*np.abs(actual),
        actual + 0.2*np.abs(actual),
        color='gray',
        alpha=0.2,
        label='±20% Error Band'
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Target Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_multivariate_predictions(actual, predicted, feature_names=None, 
                                samples_to_plot=3, timesteps_to_plot=100,
                                title="Multivariate Predictions", save_path=None):
    """
    Plot actual vs predicted values for multivariate time series
    
    Args:
        actual: numpy array of shape [timesteps, n_features]
        predicted: numpy array of shape [timesteps, n_features]
        feature_names: list of feature names
        samples_to_plot: number of random samples to visualize
        timesteps_to_plot: how many timesteps to show per sample
        title: plot title
        save_path: where to save the image
    """
    if feature_names is None:
        feature_names = [f'Var_{i}' for i in range(actual.shape[1])]
    
    n_features = actual.shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 2*n_features), sharex=True)
    if n_features == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    # Plot each feature
    for i, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        # Select random starting point
        start_idx = np.random.randint(0, actual.shape[0] - timesteps_to_plot)
        end_idx = start_idx + timesteps_to_plot
        
        # Plot actual and predicted
        ax.plot(actual[start_idx:end_idx, i], label='Actual', color='#1f77b4', alpha=0.8)
        ax.plot(predicted[start_idx:end_idx, i], '--', label='Predicted', color='#ff7f0e', alpha=0.8)
        
        ax.set_ylabel(feat_name, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right')
            ax.set_title(title, fontsize=12)
    
    plt.xlabel('Time Steps', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
  

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = SummaryWriter(log_dir=f'runs/{args.model}_{args.data}')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss.lower() == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss.lower() == 'mse':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x, batch_x_mark, batch_y_mark)
                    elif 'GLAFF' in self.args.model:
                        label_len = self.args.seq_len // 2
                        batch_y_mark = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1).to(self.device)
                        outputs, _, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                total_loss.update(loss.item(), outputs.shape[0])

        self.model.train()
        return total_loss

    def train(self, setting, ft=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        #l1 = nn.L1Loss()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = AverageMeter()

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                n_batch, n_vars = batch_y.shape[0], batch_y.shape[2]
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.update(loss.item(), outputs.shape[0])
                else:
                    if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x, batch_x_mark, batch_y_mark)
                        batch_xy = torch.cat([batch_x, batch_y], dim=1)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        if ft:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                        else: 
                            outputs = outputs[:, :, f_dim:]
                            loss = criterion(outputs, batch_xy)
                        train_loss.update(loss.item(), outputs.shape[0])
                    elif 'GLAFF' in self.args.model:
                        label_len = self.args.seq_len // 2
                        batch_y_mark = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1).to(self.device)
                        outputs, reco, map1, map2 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(torch.concat([outputs, reco, map1, map2], dim=1),
                                         torch.concat([batch_y, batch_x, batch_y, batch_y], dim=1))
                        train_loss.update(loss.item(), outputs.shape[0])
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.update(loss.item(), outputs.shape[0])

                # Log training loss
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler=scheduler, printout=False)
                    scheduler.step()
                
            # Log average training loss for the epoch
            self.writer.add_scalar('Loss/train_epoch', train_loss.avg, epoch)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # Log validation and test losses
            self.writer.add_scalar('Loss/validation', vali_loss.avg, epoch)
            self.writer.add_scalar('Loss/test', test_loss.avg, epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss.avg, vali_loss.avg, test_loss.avg))
            early_stopping(vali_loss.avg, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj == 'TST':
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, with_curve=1):
        #batch_size = self.args.batch_size
        #self.args.batch_size = 1
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test') # plot curve
        scaler = test_data.scaler  # Assuming your dataset returns this
        #self.args.batch_size = batch_size
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if (not os.path.exists(folder_path)) and with_curve:
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'FITS' in self.args.model:
                        outputs, low = self.model(batch_x, batch_x_mark, batch_y_mark)
                    elif 'GLAFF' in self.args.model:
                        label_len = self.args.seq_len // 2
                        batch_y_mark = torch.concat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1).to(self.device)
                        outputs, _, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape  # (batch, pred_len, features)
                    outputs = outputs.reshape(-1, shape[-1])
                    batch_y = batch_y.reshape(-1, shape[-1])
                    
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                    outputs = outputs.reshape(shape)
                    batch_y = batch_y.reshape(shape)
                    # shape = outputs.shape
                    # outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    # batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.extend(pred)
                trues.extend(true)
                if with_curve:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape  # (batch, pred_len, features)
                        input = input.reshape(-1, shape[-1])
                        
                        input = test_data.inverse_transform(input)

                        input = outputs.reshape(shape)
                        # shape = input.shape
                        # input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

      
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Aggregate all predictions
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

                # SHAP Analysis
        # SHAP Analysis
        # Feature names (modify according to your dataset)
        feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] 
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Plot multivariate results
        plot_multivariate_predictions(
            actual=trues,
            predicted=preds,
            feature_names=feature_names,
            samples_to_plot=3,  # Will create 3 separate plots
            timesteps_to_plot=168,  # Show 1 week of hourly data
            
            title=f'Multivariate Predictions - {setting}',
            save_path=os.path.join(folder_path, 'multivariate_predictions.png')
        )
        
        # Additional per-feature metrics
        feature_metrics = {}
        for i, feat_name in enumerate(feature_names):
            feat_pred = preds[:, i]
            feat_true = trues[:, i]
            
            # Calculate metrics per feature
            mae, mse, rmse, mape, mspe = metric(feat_pred.reshape(-1,1), feat_true.reshape(-1,1))
            feature_metrics[feat_name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }
            
            # Save individual feature plots
            plot_predictions(
                actual=feat_true[:672],  # First 4 weeks of data
                predicted=feat_pred[:672],
                title=f'{feat_name} Predictions',
                save_path=os.path.join(folder_path, f'{feat_name}_predictions.png')
            )
        
        # Function to convert numpy.float32 to standard float in a dictionary
        def convert_to_float(data):
            if isinstance(data, dict):
                return {key: convert_to_float(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_to_float(item) for item in data]
            elif isinstance(data, (np.float32, np.float64)):
                return float(data)
            else:
                return data

        # Convert feature_metrics
        converted_feature_metrics = convert_to_float(feature_metrics)

        # Save JSON
        with open('output.json', 'w') as f:
            json.dump(converted_feature_metrics, f, indent=4)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # Log test metrics
        self.writer.add_scalar('Test/MAE', mae, 0)
        self.writer.add_scalar('Test/MSE', mse, 0)
        self.writer.add_scalar('Test/RMSE', rmse, 0)
        self.writer.add_scalar('Test/MAPE', mape, 0)
        self.writer.add_scalar('Test/MSPE', mspe, 0)
        
        self.writer.close()
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
