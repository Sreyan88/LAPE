import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union
#from pl_bolts.metrics import mean, precision_at_k
# from torchmetrics import Precision

from src.utils import off_diagonal, concat_all_gather, loss_fn_mse
from src.upstream.unfused.upstream_encoder import UNFUSED as UNFUSED_ENCODER


class Project(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        sizes = [in_dim, out_dim, out_dim, out_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
    def forward(self, x):
        return self.projector(x)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_cluster):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_cluster)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        return(self.linear(x))


class Upstream_Expert(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)
    CLI command::
        # cifar10
        python moco2_module.py --gpus 1
        # imagenet
        python moco2_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        config,
        base_encoder,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.base_encoder = base_encoder
  
        self.encoder_q = self.init_encoders(self.base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.p1 = Project(2048, self.config["pretrain"]["task_label"])
        self.p2 = Project(1024, self.config["pretrain"]["task_label"])
        self.p3 = Project(512, self.config["pretrain"]["task_label"])
        self.softmax = nn.Softmax()
        self.classifier = Classifier(2048, self.config["pretrain"]["task_label"])
        self.kl_divg = nn.KLDivLoss(reduction="batchmean")       
       

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = UNFUSED_ENCODER(self.config, base_encoder)
        
        return encoder_q
        

    def forward(self, img_q=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q_raw, (q1,q2,q3) = self.encoder_q(img_q)  # queries: NxC
        #q = nn.functional.normalize(q, dim=1)
        q_classifer = self.classifier(q_raw)

        return q1,q2,q3,q_classifer 

    def training_step(self, batch, batch_idx):
        
        (img_1, _), label = batch
        q1,q2,q3,q_classifier = self(img_q=img_1)

        q1_tag = self.p1(q1) #new projector for sssd
        q2_tag = self.p2(q2) #new projector for sssd
        q3_tag = self.p3(q3) #new projector for sssd
        #Cross entroy loss defined
        loss_ce1 = F.cross_entropy(q1_tag, label.long())
        loss_ce2 = F.cross_entropy(q2_tag, label.long())
        loss_ce3 = F.cross_entropy(q3_tag, label.long())
        loss_ce = self.config["pretrain"]["alpha"]*(loss_ce1 + loss_ce2 + loss_ce3) + F.cross_entropy(q_classifier, label.long())
        #KL-divergence
        q1_log_soft = F.log_softmax(q1_tag, dim=1)
        q2_log_soft = F.log_softmax(q2_tag, dim=1)
        q3_log_soft = F.log_softmax(q3_tag, dim=1)
        targets = F.softmax(q_classifier, dim=1)
        loss_kl = self.config["pretrain"]["beta"]*(self.kl_divg(q1_log_soft, targets)+self.kl_divg(q2_log_soft, targets)+self.kl_divg(q3_log_soft, targets))
        #MSE_loss
        loss_mse = self.config["pretrain"]["gamma"]*(loss_fn_mse(q1_tag, q_classifier) + loss_fn_mse(q2_tag, q_classifier) + loss_fn_mse(q3_tag, q_classifier))
        #final loss
        loss_complete = loss_mse + loss_kl + loss_ce

        log = {'train_loss': loss_complete, 'kl-loss': loss_kl, 'CE-loss': loss_ce, 'mse-loss': loss_mse}
        
        self.log_dict(log)
        return loss_complete

    def validation_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), labels = batch

        output, target,q1,q2,q3,q4,k1,k2,k3,k4 = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        loss+= self.p1(q1,k1)
        loss+= self.p2(q2,k2)
        loss+= self.p3(q3,k3)
        loss+= self.p4(q4,k4)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
