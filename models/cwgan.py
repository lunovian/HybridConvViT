import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.generator import Generator
from models.critic import Critic
from torchvision.models import vgg16, VGG16_Weights
import torch_fidelity
import numpy as np
from skimage.color import lab2rgb
from torch import nn
from scipy.stats import entropy

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])
        self.slice4 = nn.Sequential(*vgg[16:23])
        for param in self.parameters():
            param.requires_grad = False
        
        # Add a conv layer to convert 2 channels to 3
        self.adapt_channels = nn.Conv2d(2, 3, kernel_size=1, padding=0, bias=False)

    def forward(self, x, y):
        # Convert 2-channel inputs to 3-channel
        x = self.adapt_channels(x)
        y = self.adapt_channels(y)

        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        h4_x = self.slice4(h3_x)
        h4_y = self.slice4(h3_y)
        return F.mse_loss(h1_x, h1_y) + F.mse_loss(h2_x, h2_y) + F.mse_loss(h3_x, h3_y) + F.mse_loss(h4_x, h4_y)
    
class CWGAN(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=0.0001, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10, lambda_l1=1, lambda_l2=1, lambda_perceptual=1):
        super().__init__()
        self.save_hyperparameters()

        self.display_step = display_step
        self.automatic_optimization = False  # Disable automatic optimization

        self.generator = Generator(in_channels, out_channels)
        self.critic = Critic(in_channels + out_channels)
        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        self.optimizer_C = torch.optim.AdamW(self.critic.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.2, patience=5)
        self.scheduler_C = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_C, mode='min', factor=0.2, patience=5)
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_perceptual = lambda_perceptual
        
        self.recon_criterion = nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

        self.generator_losses, self.critic_losses, self.real_accuracies, self.fake_accuracies = [], [], [], []

    def configure_optimizers(self):
        return [self.optimizer_C, self.optimizer_G], [self.scheduler_C, self.scheduler_G]

    def compute_losses(self, gen_AB, AB):
        l1_loss = self.l1_loss(gen_AB, AB)
        l2_loss = self.l2_loss(gen_AB, AB)
        perceptual_loss = self.perceptual_loss(gen_AB, AB)

        total_loss = (self.lambda_l1 * l1_loss +
                      self.lambda_l2 * l2_loss +
                      self.lambda_perceptual * perceptual_loss)
        
        return l1_loss, l2_loss, perceptual_loss, total_loss

    def generator_step(self, real_images, conditioned_images):
        fake_images = self.generator(conditioned_images)
        l1_loss, l2_loss, perceptual_loss, total_loss = self.compute_losses(fake_images, real_images)
        return l1_loss, l2_loss, perceptual_loss, total_loss

    def critic_step(self, real_images, conditioned_images):
        fake_images = self.generator(conditioned_images)
        fake_logits = self.critic(fake_images, conditioned_images)
        real_logits = self.critic(real_images, conditioned_images)

        loss_C = real_logits.mean() - fake_logits.mean()

        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True).to(self.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)
        interpolated_logits = self.critic(interpolated, conditioned_images)

        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty

        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        return loss_C

    def training_step(self, batch, batch_idx):
        real, condition = batch['AB'], batch['L']
        opt_C, opt_G = self.optimizers()

        # Critic step
        opt_C.zero_grad()
        loss_C = self.critic_step(real, condition)
        self.manual_backward(loss_C)
        opt_C.step()
        self.critic_losses.append(loss_C.item())

        self.log('train/critic_loss', loss_C, on_step=True, on_epoch=True, logger=True)

        # Generator step
        opt_G.zero_grad()
        l1_loss, l2_loss, perceptual_loss, total_loss = self.generator_step(real, condition)
        self.manual_backward(total_loss)
        opt_G.step()
        self.generator_losses.append(total_loss.item())

        self.log('train/generator_loss', total_loss, on_epoch=True, logger=True)
        self.log('train/l1_loss', l1_loss, on_epoch=True, logger=True)
        self.log('train/l2_loss', l2_loss, on_epoch=True, logger=True)
        self.log('train/perceptual_loss', perceptual_loss, on_epoch=True, logger=True)

        if self.current_epoch % self.display_step == 0 and batch_idx == 0:
            self.save_and_display_progress(real, condition)

    def save_and_display_progress(self, real, condition):
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step
        fake = self.generator(condition).detach()

        # Overwrite the same files to save space
        torch.save(self.generator.state_dict(), "models/ResUnet_latest.pt")
        torch.save(self.critic.state_dict(), "models/PatchGAN_latest.pt")

        print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
        
    def validation_step(self, batch, batch_idx):
        real, condition = batch['AB'], batch['L']
        fake_images = self.generator(condition)
        l1_loss, l2_loss, perceptual_loss, total_loss = self.compute_losses(fake_images, real)

        self.log('val/generator_loss', total_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('val/l1_loss', l1_loss, logger=True, on_epoch=True)
        self.log('val/l2_loss', l2_loss, logger=True, on_epoch=True)
        self.log('val/perceptual_loss', perceptual_loss, logger=True, on_epoch=True)

        return total_loss

    def lab_to_rgb(self, L, AB):
        """
        Takes an image or a batch of images and converts from LAB space to RGB
        """
        L = L * 100  # Denormalize L to [0, 100]
        AB = (AB - 0.5) * 128 * 2  # Denormalize AB to [-128, 127]
        # Ensure L and AB have the same spatial dimensions
        L = L.permute(0, 2, 3, 1)
        AB = AB.permute(0, 2, 3, 1)
        Lab = torch.cat([L, AB], dim=3).numpy()
        rgb_imgs = []
        for img in Lab:
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)

    def display_progress(self, cond, real, fake, current_epoch=0, figsize=(20, 15)):
        """
        Save cond, real (original) and generated (fake) images in one panel
        """
        cond = cond.detach().cpu().unsqueeze(0)  # Add batch dimension
        real = real.detach().cpu().unsqueeze(0)  # Add batch dimension
        fake = fake.detach().cpu().unsqueeze(0)  # Add batch dimension

        images = [cond, real, fake]
        titles = ['input', 'real', 'generated']
        print(f'Epoch: {current_epoch}')
        
        # Debug: Print the min and max values of the images
        for idx, img in enumerate(images):
            print(f'{titles[idx]} L min: {img[:, 0, :, :].min().item()}, max: {img[:, 0, :, :].max().item()}')
            if idx != 0:
                print(f'{titles[idx]} AB min: {img[:, 1:, :, :].min().item()}, max: {img[:, 1:, :, :].max().item()}')

        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for idx, img in enumerate(images):
            if idx == 0:
                ab = torch.zeros_like(img).repeat(1, 2, 1, 1)  # Create an AB channel with zeros
                img = torch.cat([img * 100, ab], dim=1)  # Concatenate L and AB channels
                img = img.permute(0, 2, 3, 1).numpy()[0]  # Reorder dimensions and convert to numpy
                imgan = lab2rgb(img)
            else:
                imgan = self.lab_to_rgb(images[0], img)[0]
            ax[idx].imshow(imgan)
            ax[idx].axis("off")
            ax[idx].set_title(titles[idx])
        plt.show()
    
    def load_model(self, generator_path, critic_path):
        generator_state_dict = torch.load(generator_path, weights_only=True)
        critic_state_dict = torch.load(critic_path, weights_only=True)
        self.generator.load_state_dict(generator_state_dict, strict=False)
        self.critic.load_state_dict(critic_state_dict, strict=False)
        print("Successfully Loaded")
        
    def predict(self, condition):
        self.generator.eval()
        with torch.no_grad():
            condition = condition.to(next(self.generator.parameters()).device)
            fake_image = self.generator(condition)
        return fake_image