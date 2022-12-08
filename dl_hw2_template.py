import math
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


## 11110COM 526000 Deep Learning HW2:Variational Autoencoder

## Don't change the below two functions (compute_PSNR, compute_SSIM)!!
def compute_PSNR(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Peak Signal to Noise Ratio (PSNR) function
    # img1 and img2 > [0, 1] 
    
    img1 = torch.as_tensor(img1, dtype=torch.float32)# In tensor format!!
    img2 = torch.as_tensor(img2, dtype=torch.float32)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))

def compute_SSIM(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Structure Similarity (SSIM) function
    # img1 and img2 > [0, 1]
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


## hyperparameters in AE
lr = 0.001
epoch = 100
batch = 20

## Read the data
x = torch.tensor(np.load('eye/data.npy').astype(np.float32))
y = torch.tensor(np.load('eye/label.npy').astype(np.float32))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch)
total_loader = DataLoader(dataset=dataset)

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## My AE
class ae(nn.Module):
    def __init__(self):
        super(ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*50*50, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3*50*50),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = ae().to(device)
print(model)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_his = []

## train
for e in range(epoch):
    train_loss = 0.0
    for d in loader:
        img, _ = d
        input_img = img.view(-1, 50*50*3).to(device)
        output = model(input_img).view(-1, 50, 50, 3)
        loss = loss_func(output, input_img.view(-1, 50, 50, 3))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
  ## print avg training statistics 
    train_loss = train_loss/len(loader)
    loss_his.append(train_loss.detach().cpu().item())
    if (e+1) % 10 == 0:
        print('Epoch: {} \tAverage training Loss: {:.6f}'.format(
          e+1, 
          train_loss
          ))

## evaluate PSNR and SSIM
total_PSNR = 0
total_SSIM = 0
for d in total_loader:
    img, _ = d
    input_img = img.view(-1, 50*50*3).to(device)
    output = model(input_img).view(-1, 50, 50, 3).detach().cpu()
    total_PSNR += compute_PSNR(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
    total_SSIM += compute_SSIM(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
print('TRAIN: average PSNR={}, SSIM={}'.format(total_PSNR/len(total_loader), total_SSIM/len(total_loader)))

## plot training loss curve
y_ = loss_his
x_ = np.array([i for i in range(1,101)])
plt.title('AE loss')
plt.plot(x_, y_)

## save model
torch.save(model.state_dict(), 'AE_.pth')

## load model and evaluate
m = ae().to(device)
m.load_state_dict(torch.load('AE.pth'))
m.eval()
total_PSNR = 0
total_SSIM = 0
for d in total_loader:
    img, _ = d
    input_img = img.view(-1, 50*50*3).to(device)
    output = m(input_img).view(-1, 50, 50, 3).detach().cpu()
    total_PSNR += compute_PSNR(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
    total_SSIM += compute_SSIM(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
print('PRETRAINED: average PSNR={}, SSIM={}'.format(total_PSNR/len(total_loader), total_SSIM/len(total_loader)))

## hyperparameters in VAE
batch = 20
epoch = 500
lr = 0.0005
loss_his = []

## My VAE
class vae(nn.Module):
    def __init__(self):
        super(vae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*50*50, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.mean = nn.Linear(512, 256)
        self.variance = nn.Linear(512, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3*50*50),
            nn.Sigmoid()
        )

    def encode(self,x):
        x = self.encoder(x)
        m = self.mean(x)
        v = self.variance(x)
        return m,v
    def forward(self, x):
        m,v = self.encode(x)
        e = torch.randn_like(v)
        t = m+e*torch.exp(v)
        out = self.decoder(t)
        return m, v, t, out

model2 = vae().to(device)
print(model2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

## train
for e in range(epoch):
    train_loss = 0.0
    for d in loader:
        img, _ = d
        input_img = img.view(-1, 50*50*3).to(device)
        m, v, t, output = model2(input_img)
        sigma = torch.exp(v)
        kl_loss = (sigma**2 + m**2 - torch.log(sigma) - 1/2).sum()
        loss = ((input_img-output)**2).sum()+kl_loss
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        train_loss += loss
  ## print avg training statistics
    train_loss = train_loss/len(loader)
    loss_his.append(train_loss.detach().cpu().item())
    if (e+1)%20 == 0:
        print('Epoch: {} \tAverage training Loss: {:.6f}'.format(e+1, train_loss))
        # print('kl_loss={} mse={}'.format(kl_loss, mse_loss))

## evaluate PSNR and SSIM
total_PSNR = 0
total_SSIM = 0
for d in total_loader:
    img, _ = d
    input_img = img.view(-1, 50*50*3).to(device)
    _, _,_,  output = model2(input_img)
    output = output.view(-1, 50, 50, 3).detach().cpu()
    total_PSNR += compute_PSNR(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
    total_SSIM += compute_SSIM(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
print('TRAIN: average PSNR={}, SSIM={}'.format(total_PSNR/len(total_loader), total_SSIM/len(total_loader)))

## plot training loss curve
y_ = loss_his
x_ = np.array([i for i in range(1,501)])
plt.title('VAE loss')
plt.plot(x_, y_)

## save model
torch.save(model2.state_dict(), 'VAE_.pth')

## load model and evaluate
m = vae().to(device)
m.load_state_dict(torch.load('VAE.pth'))
m.eval()
total_PSNR = 0
total_SSIM = 0
for d in total_loader:
    img, _ = d
    input_img = img.view(-1, 50*50*3).to(device)
    _, _, _, output = m(input_img)
    output = output.view(-1, 50, 50, 3).detach().cpu()
    total_PSNR += compute_PSNR(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
    total_SSIM += compute_SSIM(img.numpy().reshape(50,50,3), output.numpy().reshape(50,50,3))
print('PRETRAINED: average PSNR={}, SSIM={}'.format(total_PSNR/len(total_loader), total_SSIM/len(total_loader)))

