import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RefinementLoss(nn.Module):
    def __init__(self, l1_weight=1.0, mse_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        total_loss = self.l1_weight * l1 + self.mse_weight * mse
        return total_loss, {'l1': l1.item(), 'mse': mse.item()}

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def forward(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        loss = self.loss(prediction, target)
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None, device='cpu'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.device = device
        if layer_weights is None:
            self.layer_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_4': 1.0,
                'relu4_4': 1.0
            }
        else:
            self.layer_weights = layer_weights
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_4': 17,
            'relu4_4': 26
        }

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)
        loss = 0.0
        x_pred = pred
        x_target = target
        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            for name, idx in self.layer_indices.items():
                if i == idx and name in self.layer_weights:
                    weight = self.layer_weights[name]
                    loss += weight * F.l1_loss(x_pred, x_target)
        return loss

    def _normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        x = (x + 1) / 2
        return (x - mean) / std

class CombinedGANLoss(nn.Module):
    def __init__(self, adv_weight=1.0, l1_weight=100.0, perc_weight=10.0, device='cpu'):
        super().__init__()
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight
        self.gan_loss = GANLoss('vanilla')
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device)

    def forward(self, fake, real, discriminator_output):
        adv_loss = self.gan_loss(discriminator_output, True)
        l1 = self.l1_loss(fake, real)
        perc = self.perceptual_loss(fake, real)
        total_loss = self.adv_weight * adv_loss + self.l1_weight * l1 + self.perc_weight * perc
        loss_dict = {'adv': adv_loss.item(), 'l1': l1.item(), 'perceptual': perc.item()}
        return total_loss, loss_dict
