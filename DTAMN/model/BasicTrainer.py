import torch
import math
import os
import time
import copy
import numpy as np
import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics
from torch.cuda.amp import autocast, GradScaler

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.scaler_amp = GradScaler()
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.batches_seen = 0
        self.meminfo = 0

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            with autocast():
                for batch_idx, (data, target) in enumerate(val_dataloader):
                    data = data
                    label = target[..., :self.args.output_dim].clone()
                    target[..., :self.args.output_dim] = self.scaler.transform(target[..., :self.args.output_dim])
                    output = self.model(data, target)
                    if self.args.real_value:
                        output = self.scaler.inverse_transform(output)
                        # label = self.scaler.inverse_transform(label)
                    loss = self.loss(output.cuda(), label)
                    # a whole batch of Metr_LA is filtered
                    if not torch.isnan(loss):
                        total_val_loss += loss.item()
            val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss,
                                                                                                      time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            with autocast():
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data = data
                    label = target[..., :self.args.output_dim].clone()
                    target[..., :self.args.output_dim] = self.scaler.transform(target[..., :self.args.output_dim])
                    output = self.model(data, target)
                    if self.args.real_value:
                        output = self.scaler.inverse_transform(output)
                        # label = self.scaler.inverse_transform(label)
                    loss = self.loss(output.cuda(), label)
                    # a whole batch of Metr_LA is filtered
                    if not torch.isnan(loss):
                        total_test_loss += loss.item()
            test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss,
                                                                                                      time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batches_seen += 1
            data = data
            label = target[..., :self.args.output_dim].clone()  
            target[..., :self.args.output_dim] = self.scaler.transform(target[..., :self.args.output_dim])
            self.optimizer.zero_grad()

            with autocast():
                
                res = self.model(data, target, self.batches_seen)
                if self.model.training and isinstance(res, tuple):
                    output, recollection = res
                else:
                    output = res
                    recollection = None

                
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)

                loss_pred = self.loss(output.cuda(), label)

                loss_rec = 0
                if recollection is not None:
                    input_history = data[..., :self.args.input_dim].clone()
                    if self.args.real_value:
                        recollection = self.scaler.inverse_transform(recollection)
                        input_history = self.scaler.inverse_transform(input_history)
                    loss_rec = self.loss(recollection.cuda(), input_history.cuda())

                loss = loss_pred + self.args.rec_loss_weight * loss_rec

            self.scaler_amp.scale(loss).backward()

            if self.args.grad_norm:
                self.scaler_amp.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()
            total_loss += loss.item()

            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f} (Pred: {:.6f} | Rec: {:.6f})'.format(
                    epoch, batch_idx + 1, self.train_per_epoch, loss.item(),
                    loss_pred.item(), loss_rec.item() if isinstance(loss_rec, torch.Tensor) else 0.0))

        train_epoch_loss = total_loss / self.train_per_epoch

        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        else:
            max_memory = 0.0

        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, Max Memory Usage: {:.2f} GB, train time: {:.2f} s'.format(
                epoch,
                train_epoch_loss,
                max_memory,
                time.time() - epoch_time))

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        best_model = None
        best_test_model = None
        # start_time = time.time()
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        vaild_loss = []

        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            vaild_loss.append(val_epoch_loss)

            test_epoch_loss = self.test_epoch(epoch, test_dataloader)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break

            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            if isinstance(check_point, dict) and 'state_dict' in check_point:
                
                state_dict = check_point['state_dict']
                
                # args = check_point['config']
            else:
                
                state_dict = check_point
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target[..., :args.output_dim]
                output = model(data, target)
                y_true.append(label)
                y_pred.append(output)

        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = torch.cat(y_true, dim=0)
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)

        
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                   args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
                t + 1, rmse, mae, mape * 100, corr))

        
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f} ".format(
            rmse, mae, mape * 100, corr))

        
        if args.output_dim > 1:
            logger.info("--- Metrics for each feature separately (Step-by-Step) ---")
            for i in range(args.output_dim):
                
                feat_name = "Pick/In" if i == 0 else "Drop/Out"
                if args.output_dim > 2: feat_name = f"Feature {i}"

                logger.info(f"=== Detailed Metrics for {feat_name} ===")

                
                for t in range(args.horizon):
                    
                    pred_t_f = y_pred[:, t, :, i]
                    true_t_f = y_true[:, t, :, i]

                    mae, rmse, mape, _, corr = All_Metrics(pred_t_f, true_t_f, args.mae_thresh, args.mape_thresh)
                    logger.info(
                        "Feature {:d} ({}) Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
                            i, feat_name, t + 1, rmse, mae, mape * 100, corr))

                
                pred_total_f = y_pred[..., i]
                true_total_f = y_true[..., i]
                mae, rmse, mape, _, corr = All_Metrics(pred_total_f, true_total_f, args.mae_thresh, args.mape_thresh)

                logger.info(
                    "Feature {:d} ({}) Average All Horizons, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
                        i, feat_name, rmse, mae, mape * 100, corr))

                logger.info("-" * 40)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))