import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # pylint: disable=wrong-import-position

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import timm
from datasets.test_dataset import GazeImageDataset
from datasets.ffhq_dgaze_dataset import FFHQDGazeDataset
import numpy as np
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import precision_recall_curve
from gaze_prediction import GazePredictor
from gaze_prediction.insightface.utils import face_align
from gaze_prediction.utils import angles_and_vec_from_eye
from reconstruction.gaze.utils import make_log_image, Dir


class GazeModel(pl.LightningModule):
    def __init__(self, backbone, epoch):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(backbone, num_classes=481*2*3)
        self.epoch = epoch
        #self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.L1Loss(reduction='mean')
        #self.hard_mining = False
        self.hard_mining = False
        self.num_face = 1103
        self.num_eye = 481*2

        self.test_dataset = GazeImageDataset()
        self.ffhq_dgaze_dl = DataLoader(dataset = FFHQDGazeDataset(), batch_size= 4, shuffle=False, num_workers=2)
        self.ffhq_dgaze_iter = iter(self.ffhq_dgaze_dl)

        self.gazemetric_outputs = {'y':[], 'y_hat':[]}
        self.gazemetric_tested = False
        self.gaze_predictor = GazePredictor()

    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def cal_loss(self, y_hat, y, hm=False):
        bs = y.size(0)
        y_hat = y_hat.view( (bs,-1,3) )
        loss = torch.abs(y_hat - y) #(B,K,3)
        loss[:,:,2] *= 0.5
        if hm:
            loss = torch.mean(loss, dim=(1,2)) #(B,)
            loss, _ = torch.topk(loss, k=int(bs*0.25), largest=True)
            #B = len(loss)
            #S = int(B*0.5)
            #loss, _ = torch.sort(loss, descending=True)
            #loss = loss[:S]
        loss = torch.mean(loss) * 20.0
        return loss

    def training_step(self, batch, batch_idx):

        # Test beginning of each epoch
        if batch_idx == 0:
            if not self.gazemetric_tested:
                self.gazemetric_test()
                self.gazemetric_tested = True
            self.gaze_test()

        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y, self.hard_mining)
        if batch_idx % 10 == 0:
            dgaze_loss = self.ffhq_dgaze_step()
            self.log('train/dgaze_loss', dgaze_loss, on_epoch=True)
        else:
            dgaze_loss = 0.0
        self.log('train/loss', loss, on_epoch=True, on_step=True)


        if batch_idx == 0:
            b = y.shape[0]
            gaze = self.y_to_gaze_vector(y)
            gaze_hat = self.y_to_gaze_vector(y_hat.view(b,-1,3))
            log_image = make_log_image(x, 'src', Dir(gaze, length=50, color=(0,1,0)),Dir(gaze_hat,length=50))
            self.logger.experiment.add_images('train/imgs', log_image, dataformats='NCHW', global_step=self.current_epoch)

        return loss + dgaze_loss

    def ffhq_dgaze_step(self):
        loss = 0.0
        try:
            batch = next(self.ffhq_dgaze_iter)
        except StopIteration:
            self.ffhq_dgaze_iter = iter(self.ffhq_dgaze_dl)
            batch = next(self.ffhq_dgaze_iter)

        imgs = []
        for img in batch['image']:
            img = self._process_image(img)
            if img is not None:
                imgs.append(img)
        if len(imgs)==0:
            return loss

        y  = self.backbone(torch.cat(imgs))
        y  = y.view(y.shape[0],-1,3)
        gaze_xy = self.y_to_gaze_vector(y)
        loss+= self.loss(gaze_xy, torch.zeros_like(gaze_xy))

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        if batch_idx == 0:
            b = y.shape[0]
            gaze = self.y_to_gaze_vector(y)
            gaze_hat = self.y_to_gaze_vector(y_hat.view(b,-1,3))
            log_image = make_log_image(x, 'src', Dir(gaze, length=50, color=(0,1,0)),Dir(gaze_hat,length=50))
            self.logger.experiment.add_images('val/imgs', log_image, dataformats='NCHW', global_step=self.current_epoch)

    @torch.no_grad()
    def gazemetric_test(self):
        print(f' Testing GazeMetric on test dataset')
        from gaze_prediction import GazePredictor
        gaze_predictor = GazePredictor()
        conf_matrix = ConfusionMatrix(task='binary').to(self._device)
        test_dl = DataLoader(self.test_dataset, batch_size=8, shuffle=False,num_workers=2)
        for batch_idx, batch in enumerate(test_dl):
            img = batch['image'].to(self._device)
            y = batch['label'].unsqueeze(1).float().to(self._device)
            imgs_np = np.clip((img.detach().cpu().numpy()+1)*127.5,0,255).transpose(0,2,3,1).astype(np.uint8)[..., ::-1]
            gaze_vectors = [gaze_predictor.predict(img_np) for img_np in imgs_np]
            valid_idxs = [i for i, gv in enumerate(gaze_vectors) if len(gv) == 1]

            if len(valid_idxs) == 0:
                continue
            y = y[valid_idxs]
            gaze_vectors = [torch.Tensor(gv[0]) for i, gv in enumerate(gaze_vectors) if len(gv) == 1]
            gaze_vectors = torch.stack(gaze_vectors)
            gaze_norm = gaze_vectors.norm(dim=1, keepdim=True)
            gaze_norm[gaze_norm > 0.05] = 0.05

            self.gazemetric_outputs['y'].append(y.cpu().numpy().astype(int).squeeze())
            self.gazemetric_outputs['y_hat'].append((0.05 - gaze_norm.cpu().numpy()).squeeze() * 20)
            y_hat = (gaze_norm < 0.01).float().to(self._device)
            conf_matrix.update(y_hat, y)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_matrix.compute().cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax)
        self.logger.experiment.add_figure('gazemetric/confusion_matrix↑', fig)

        self.gazemetric_outputs['y'] = np.concatenate(self.gazemetric_outputs['y'])
        self.gazemetric_outputs['y_hat'] = np.concatenate(self.gazemetric_outputs['y_hat'])

    @torch.no_grad()
    def gaze_test(self):
        print(f'Testing model on gaze dataset')
        self.backbone.eval()
        gaze_outputs   = {'y': [], 'y_hat': [],'gaze_xy':[]}
        test_dl = DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=2)
        conf_matrix = ConfusionMatrix(task='binary').to(self._device)
        log_images  = []
        for batch_idx, batch in enumerate(test_dl):
            imgs    = batch['image'].to(self._device)
            y       = batch['label'].unsqueeze(1).float().to(self._device)
            for i,img in enumerate(imgs):
                gaze_vector = self._compute_gaze_from_model(img)
                if gaze_vector is None:
                    continue
                gaze_norm = np.linalg.norm(gaze_vector)
                gaze_norm = 0.05 if gaze_norm>0.05 else gaze_norm

                gaze_outputs['gaze_xy'].append(gaze_vector)
                gaze_outputs['y'].append(y[i].cpu().numpy().astype(int).squeeze())
                gaze_outputs['y_hat'].append((0.05 - gaze_norm) * 20)

                conf_matrix.update(y[i], (torch.Tensor([gaze_norm]) < 0.01).float().to(self._device))

                if (len(log_images))<64 and y[i].item()==0:
                    log_image = make_log_image(img.unsqueeze(0),'src', Dir(torch.Tensor(gaze_vector[None,...])))
                    log_images.append(log_image)

        # Plot example images
        self.logger.experiment.add_images('test/imgs', np.concatenate(log_images), dataformats='NCHW', global_step=self.current_epoch)

        # Plot confusion matrix
        plt.cla()
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_matrix.compute().cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax)
        self.logger.experiment.add_figure('test/confusion_matrix↑', fig, global_step=self.current_epoch)

        # Plot Gaze scatter plot
        y_np = np.asarray(gaze_outputs['y'])
        gaze_xy = np.asarray(gaze_outputs['gaze_xy'])
        plt.cla()
        fig, ax = plt.subplots()
        ax.scatter(gaze_xy[y_np == 0, 0], gaze_xy[y_np == 0, 1], label='gaze', color='g')
        ax.scatter(gaze_xy[y_np == 1, 0], gaze_xy[y_np == 1, 1], label='not gaze', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        self.logger.experiment.add_figure('test/gaze_scatter', fig, global_step=self.current_epoch)


        # Plot both plots
        y_np = np.asarray(gaze_outputs['y'])
        y_hat_np = np.asarray(gaze_outputs['y_hat'])
        precision, recall, _ = precision_recall_curve(y_np, y_hat_np, drop_intermediate=True)
        plt.cla()
        fig, ax = plt.subplots()
        ax.plot(recall, precision, marker='.', label='model', color='b')

        if len(self.gazemetric_outputs['y']) > 0:
            precision, recall, _ = precision_recall_curve(self.gazemetric_outputs['y'],
                                                          self.gazemetric_outputs['y_hat'],
                                                          drop_intermediate=True)
            ax.plot(recall, precision, marker='.', label='gazemetric', color='g')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_yticks(np.arange(0, 1.0, 0.1))
        ax.legend()
        self.logger.experiment.add_figure('test/precision_recall_curve', fig, global_step=self.current_epoch)

        # Revert back to training mode
        self.backbone.train()


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0002)
        # opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        # epoch_steps = [int(self.epoch*0.4), int(self.epoch*0.7), int(self.epoch*0.9)]
        # print('epoch_steps:', epoch_steps)
        # def lr_step_func(epoch):
        #     return 0.1 ** len([m for m in epoch_steps if m <= epoch])
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         optimizer=opt, lr_lambda=lr_step_func)
        # lr_scheduler = {
        #         'scheduler': scheduler,
        #         'name': 'learning_rate',
        #         'interval':'epoch',
        #         'frequency': 1}
        # return [opt], [lr_scheduler]

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     # Manually step the scheduler at each epoch or step as required
    #     # Here, we're assuming the scheduler is updated every epoch
    #     scheduler.step(self.current_epoch)  # pass epoch if required by LambdaLR

    def _compute_gaze_from_model(self, img: torch.Tensor) -> np.ndarray or None:
        '''
        img: 1CHW [-1,1] RGB image
        '''

        img = self._process_image(img)
        if img is None:
            return None
        gaze_xy = self.y_to_gaze_vector(self.backbone(img).view(1,-1,3)).detach().cpu().numpy().squeeze()

        return gaze_xy

    def _process_image(self,img:torch.Tensor):
        img_np = np.clip((img.detach().cpu().numpy() + 1) * 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)[...,
                 ::-1]
        # Transform image according to gaze predictor model
        faces = self.gaze_predictor.app.get(img_np)
        if len(faces) != 1:
            return None
        face = faces[0]

        # Image preprocessing
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        kps = face.kps
        center = (kps[0] + kps[1]) / 2.0
        _size = max(width / 1.5, np.abs(kps[1][0] - kps[0][0])) * 1.5
        rotate = 0
        _scale = self.gaze_predictor.input_size / _size
        aimg, M = face_align.transform(img_np, center, self.gaze_predictor.input_size, _scale, rotate)

        input = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        input = np.transpose(input, (2, 0, 1))
        input = np.expand_dims(input, 0)
        input = torch.Tensor(input).cuda()
        input.div_(127.5).sub_(1.0)

        return input

    def y_to_gaze_vector(self, ys:torch.Tensor):

        ys = ys
        ys[..., 0:2] += 1
        ys[..., 0:2] *= (self.gaze_predictor.input_size // 2)
        ys[..., 2] *= 10.0

        def vec_from_eye(eye_pts):
            p_iris = eye_pts[:,self.gaze_predictor.iris_idx_481] - eye_pts[:,:32].mean(dim=1, keepdim=True)
            vec    = p_iris.mean(dim=1)
            vec    = vec/torch.linalg.norm(vec,dim=1,keepdim=True)
            return vec

        eye_l = ys[:,:self.gaze_predictor.num_eye, :]
        eye_r = ys[:,self.gaze_predictor.num_eye:, :]

        gaze_xy = (vec_from_eye(eye_r)[:,:2] + vec_from_eye(eye_l)[:,:2])/2

        return gaze_xy
