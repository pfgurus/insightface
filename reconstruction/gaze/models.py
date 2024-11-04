import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import timm
from datasets.test_dataset import GazeImageDataset
import numpy as np
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import precision_recall_curve

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
        self.gazemetric_outputs = {'y':[], 'y_hat':[]}
        self.gazemetric_tested = False

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
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

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
        from gaze_prediction import GazePredictor
        from gaze_prediction.insightface.utils import face_align
        gaze_predictor = GazePredictor()
        gaze_outputs   = {'y': [], 'y_hat': []}
        test_dl = DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=2)
        for batch_idx, batch in enumerate(test_dl):
            imgs    = batch['image'].to(self._device)
            y       = batch['label'].unsqueeze(1).float().to(self._device)
            imgs_np = np.clip((imgs.detach().cpu().numpy()+1)*127.5,0,255).transpose(0,2,3,1).astype(np.uint8)[..., ::-1]
            for i,img_np in enumerate(imgs_np):
                # Transform image according to gaze predictor model
                faces = gaze_predictor.app.get(img_np)
                if len(faces) != 1:
                    continue
                face = faces[0]

                bbox = face.bbox
                width = bbox[2] - bbox[0]
                kps = face.kps
                center = (kps[0] + kps[1]) / 2.0
                _size = max(width / 1.5, np.abs(kps[1][0] - kps[0][0])) * 1.5
                rotate = 0
                _scale = gaze_predictor.input_size / _size
                aimg, M = face_align.transform(img_np, center, gaze_predictor.input_size, _scale, rotate)

                input = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
                input = np.transpose(input, (2, 0, 1))
                input = np.expand_dims(input, 0)
                input = torch.Tensor(input).cuda()
                input.div_(127.5).sub_(1.0)

                gaze_vector = self.backbone(input)
                gaze_norm = gaze_vector.norm(dim=1, keepdim=True)
                gaze_norm[gaze_norm > 0.05] = 0.05

                gaze_outputs['y'].append(y[i].cpu().numpy().astype(int).squeeze())
                gaze_outputs['y_hat'].append((0.05 - gaze_norm.cpu().numpy()).squeeze() * 20)

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
        self.logger.experiment.add_figure('test/precision_recall_curve', fig)

        # Revert back to training mode
        self.backbone.train()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0002)
        opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        epoch_steps = [int(self.epoch*0.4), int(self.epoch*0.7), int(self.epoch*0.9)]
        print('epoch_steps:', epoch_steps)
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in epoch_steps if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # Manually step the scheduler at each epoch or step as required
        # Here, we're assuming the scheduler is updated every epoch
        scheduler.step(self.current_epoch)  # pass epoch if required by LambdaLR
