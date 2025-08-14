from __future__ import print_function, absolute_import
import time

import torch
from torch.cuda import amp

from .tools.contrastive_loss import *
from .utils.meters import AverageMeter
import torch.nn.functional as F

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        # amp fp16 training
        scaler = amp.GradScaler()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            imgs, labels, indexes = self._parse_data(inputs)

            with (((amp.autocast(enabled=True)))):

                features = self._forward(imgs)

                loss = self.memory(features, labels)


            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs

        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class ClusterContrastTrainer_PCLMP(object):
    def __init__(self, encoder, encoder_ema, memory=None):
        super(ClusterContrastTrainer_PCLMP, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        losses_ema = AverageMeter()
        end = time.time()
        scaler = amp.GradScaler()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            imgs, labels, indexes = self._parse_data(inputs)


            with (((amp.autocast(enabled=True)))):
                features = self._forward(imgs)

                with torch.no_grad():
                    features_ema = self._forward_ema(imgs)

                loss_cc = self.memory(features,labels,  model_name='encoder')

                loss_ema = self.memory(features, labels,  model_name='encoder_ema')

                loss = loss_cc + loss_ema

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss_cc.item())
            losses_ema.update(loss_ema.item())

            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss cc {:.3f} ({:.3f})\t'
                      'Loss ema {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ema.val, losses_ema.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _forward_ema(self, inputs):
        return self.encoder_ema(inputs)

    def _update_ema_variables(self, model, ema_model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
