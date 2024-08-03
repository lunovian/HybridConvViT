import argparse
import os
import warnings
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb
from models.cwgan import CWGAN
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch_fidelity import calculate_metrics
from scipy.stats import entropy
from torchvision.models import inception_v3, Inception_V3_Weights

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def preprocess_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    lab_image = rgb2lab(image).astype(np.float32)
    L = lab_image[:, :, 0] / 100.0
    AB = (lab_image[:, :, 1:] / 256.0) + 0.5
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)
    AB = torch.from_numpy(AB).permute(2, 0, 1).unsqueeze(0)
    return L, AB

def postprocess_image(L, AB):
    L = L.squeeze(0).squeeze(0).cpu().numpy() * 100
    AB = (AB.squeeze(0).permute(1, 2, 0).cpu().numpy() - 0.5) * 256
    lab_image = np.zeros((L.shape[0], L.shape[1], 3))
    lab_image[:, :, 0] = L
    lab_image[:, :, 1:] = AB
    rgb_image = lab2rgb(lab_image)
    return (rgb_image * 255).astype(np.uint8)

def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    os.makedirs(directory, exist_ok=True)

def save_images(images, directory, prefix):
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(os.path.join(directory, f"{prefix}_{i}.png"))

def evaluate_metrics(real_images, fake_images):
    # Convert images to numpy arrays
    real_images_np = np.array(real_images).transpose((0, 3, 1, 2)) / 255.0
    fake_images_np = np.array(fake_images).transpose((0, 3, 1, 2)) / 255.0

    # Convert to torch tensors
    real_images_torch = torch.tensor(real_images_np, dtype=torch.float32)
    fake_images_torch = torch.tensor(fake_images_np, dtype=torch.float32)

    real_dir = 'testing/real_images_temp'
    fake_dir = 'testing/fake_images_temp'
    clear_directory(real_dir)
    clear_directory(fake_dir)

    save_images((real_images_torch * 255).byte().permute(0, 2, 3, 1).numpy(), real_dir, 'real')
    save_images((fake_images_torch * 255).byte().permute(0, 2, 3, 1).numpy(), fake_dir, 'fake')

    fid_value = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=True
    )['frechet_inception_distance']

    for f in os.listdir(real_dir):
        os.remove(os.path.join(real_dir, f))
    for f in os.listdir(fake_dir):
        os.remove(os.path.join(fake_dir, f))
    os.rmdir(real_dir)
    os.rmdir(fake_dir)

    inception_score_mean, _ = inception_score(fake_images_np)

    # Setting a smaller window size for SSIM calculation and specifying data range
    ssim_values = [ssim(real, fake, multichannel=True, win_size=3, data_range=1.0) for real, fake in zip(real_images_np, fake_images_np)]
    psnr_values = [psnr(real, fake, data_range=1.0) for real, fake in zip(real_images_np, fake_images_np)]

    return fid_value, inception_score_mean, np.mean(ssim_values), np.mean(psnr_values)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i in range(0, N, batch_size):
        batch = imgs[i:i + batch_size].astype(np.float32)
        batch = torch.from_numpy(batch).type(dtype)
        preds[i:i + batch_size] = get_pred(batch)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def main(test_dir, batch_size, num_images=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    all_image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, fname))]
    if num_images:
        all_image_paths = all_image_paths[:num_images]
    dataset = ImageDataset(all_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    cwgan = CWGAN(in_channels=1, out_channels=2)
    cwgan.load_model('models/ResUnet_latest.pt', 'models/PatchGAN_latest.pt')
    cwgan.generator.eval()

    real_images, fake_images = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwgan.generator.to(device)

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            L_batch = torch.cat([preprocess_image(img)[0].to(device) for img in batch])
            gen_images_batch = cwgan.generator(L_batch)
            for i in range(gen_images_batch.size(0)):
                fake_images.append(postprocess_image(L_batch[i], gen_images_batch[i]))
                real_images.append(postprocess_image(*preprocess_image(batch[i].cpu())))

    clear_directory('testing/real_images')
    clear_directory('testing/fake_images')

    save_images(real_images, 'testing/real_images', 'real')
    save_images(fake_images, 'testing/fake_images', 'fake')

    fid, inception, ssim_val, psnr_val = evaluate_metrics(real_images, fake_images)

    print(f'FID: {fid}')
    print(f'Inception Score: {inception}')
    print(f'SSIM: {ssim_val}')
    print(f'PSNR: {psnr_val}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CWGAN on a specified dataset and calculate metrics.')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to test')

    args = parser.parse_args()

    main(args.test_dir, args.batch_size, args.num_images)
