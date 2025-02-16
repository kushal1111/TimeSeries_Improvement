from data_provider.data_factory import data_provider
from torch.optim import lr_scheduler
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, AverageMeter, visual_xai
from utils.metrics import metric
from torch import optim
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = SummaryWriter(log_dir=f"./tensorboard_logs/{args.model}")

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
    
    def log_shap_lime_to_tensorboard(self, shap_values, lime_explanations, step):
        # Plot SHAP values
        shap_figure = plt.figure()
        shap.summary_plot(shap_values, show=False)
        self.writer.add_figure('SHAP Summary Plot', shap_figure, global_step=step)

        # Plot LIME explanations
        lime_figure = plt.figure()
        for i, explanation in enumerate(lime_explanations):
            plt.subplot(1, len(lime_explanations), i+1)
            explanation.as_pyplot_figure()
        self.writer.add_figure('LIME Explanations', lime_figure, global_step=step)

        # Close figures
        plt.close(shap_figure)
        plt.close(lime_figure)

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

        # Add XAI metrics tracking
        self.attention_consistency = AverageMeter()
        self.gradcam_variance = AverageMeter()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

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
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss.avg, vali_loss.avg, test_loss.avg))
            # Log training metrics
            self.writer.add_scalar("Loss/train", train_loss.avg, epoch)
            self.writer.add_scalar("Loss/val", vali_loss.avg, epoch)
            # Log model parameter histograms
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"Parameters/{name}", param, epoch)

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
        test_data, test_loader = self._get_data(flag='test') # plot curve
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

                # Add XAI visualization
                if with_curve:
                    # Get explanation maps
                    grad_cam = self.model.module.get_grad_cam().cpu().numpy()
                    attn_weights = self.model.module.attention_weights
                    
                    # Visualize alongside predictions
                    visual_xai(true, pred, grad_cam, attn_weights, os.path.join(folder_path, str(i)+'_xai.svg'))
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.extend(pred)
                trues.extend(true)
                if with_curve and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))
        # Generate SHAP values
        shap_values = self.model.integrate_shap(test_loader)

        # Generate LIME explanations
        lime_explanations = self.model.integrate_lime(test_loader)
        # Log explanations to TensorBoard at a specific step
        self.log_shap_lime_to_tensorboard(shap_values, lime_explanations, step=0)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # Log test metrics
        self.writer.add_scalar("MSE/test", mse, 0)
        self.writer.add_scalar("MAE/test", mae, 0)
        # Close the writer when done
        self.writer.close()
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