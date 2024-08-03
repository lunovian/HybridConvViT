import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from data.dataloader import setup_dataloader
from models.cwgan import CWGAN
from torchvision import transforms
import wandb

wandb.require("core")
torch.set_float32_matmul_precision('medium')

def main(train_dir, val_dir, epochs, learning_rate, display_step, batch_size, image_size):
    augmentation_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_loader = setup_dataloader(train_dir, augmentation_transforms, batch_size=batch_size)
    val_loader = setup_dataloader(val_dir, augmentation_transforms, batch_size=batch_size, shuffle=False)

    cwgan = CWGAN(in_channels=1, out_channels=2, learning_rate=learning_rate, lambda_recon=100, display_step=display_step)
    cwgan.load_model('models/ResUnet_latest.pt', 'models/PatchGAN_latest.pt')

    logger = WandbLogger(project="REColor")

    early_stopping = EarlyStopping(
        monitor='val/generator_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        callbacks=[early_stopping],
        precision="16-mixed",
        benchmark=True,
        profiler='simple',
        logger=logger
    )

    trainer.fit(cwgan, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CWGAN on a specified dataset.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--display_step', type=int, default=1, help='Frequency of displaying images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Size of input images')

    args = parser.parse_args()

    main(args.train_dir, args.val_dir, args.epochs, args.learning_rate, args.display_step, args.batch_size, args.image_size)
