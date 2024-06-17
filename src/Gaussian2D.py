import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMALL_CONST = 1e-4

#######################################################################################
#  #
#                                                                        #
#######################################################################################

def sigmoid(x):
    sigma = 1. / (1. + torch.exp(-x))
    return (sigma - 0.5) * 2.
# pos: (N, 2)
# mu: (K, 2)
# sigma_x: (k)
# sigma_y: (k)
# rho: (k)
# color: (k, 3)
def construct_2DGS(pos, mu, sigma_x, sigma_y, rho, color):
    pos = torch.unsqueeze(pos, 1)
    mu = torch.unsqueeze(mu, 0)
    diff = pos - mu

    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
        torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    inv_cov = torch.inverse(covariance)

    expo = -0.5 * torch.einsum('nkj, kjp, nkp -> nk', diff, inv_cov, diff)
    alpha = torch.exp(expo)
    pred_img = torch.unsqueeze(color, 0) * torch.unsqueeze(alpha, -1)
    pred_img = torch.sum(pred_img, dim=1)
    # pred_img = torch.clamp(pred_img, min=0., max=1.)
    pred_img = sigmoid(pred_img)
    return pred_img

def normalize(X, max):
    return X / float(max)
    
def restore(X_n, max):
    return X_n * float(max)

class GS(nn.Module):
    def __init__(self, K, mu, color):
        super(GS, self).__init__()
        self.K = K

        # self.mu = torch.from_numpy(mu).float().to(device)
        self.mu = nn.Parameter(mu).to(device)

        # self.color = points
        self.color = nn.Parameter(color).to(device)

        # self.scales = nn.Parameter(torch.ones(K, 2).float() * 10.).to(device)
        # self.thetas = nn.Parameter(torch.zeros(K).float()).to(device)

        sigma_bias = 0.08
        self.sigma_x = nn.Parameter(sigma_bias + torch.rand(K) * 0.1).to(device)
        self.sigma_y = nn.Parameter(sigma_bias + torch.rand(K) * 0.1).to(device)
        self.rho = nn.Parameter(torch.rand(K) - 0.5).to(device)

    def forward(self, pos):
        self.sigma_x = nn.Parameter(torch.clamp(self.sigma_x, min=SMALL_CONST, max=1.))
        self.sigma_y = nn.Parameter(torch.clamp(self.sigma_y, min=SMALL_CONST, max=1.))
        self.rho = nn.Parameter(torch.clamp(self.rho, min=-0.9, max=0.9))
        self.color = nn.Parameter(torch.clamp(self.color, min=0., max=1.))

        return construct_2DGS(pos=pos, mu=self.mu, sigma_x=self.sigma_x, sigma_y=self.sigma_y, rho=self.rho, color=self.color)

# SSIM loss
def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        x = torch.arange(window_size)
        gauss = torch.exp(- (x - window_size // 2)**2 / float(2 * sigma**2))
        gauss = gauss / gauss.sum()
        return gauss

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()

    window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
    window = torch.autograd.Variable(window)

    return window



def ssim_loss(img1, img2, window_size=11):


    # Assuming the image is of shape [N, C, H, W]
    (_, _, channel) = img1.size()

    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)


    # Parameters for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    SSIM_numerator = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator
    MSSIM = torch.clamp((1 - SSIM) / 2, 0, 1).mean()

    return MSSIM

# Combined Loss
def combined_loss(pred, target, lambda_param=0.5):
    l1loss = nn.L1Loss()
    return (1 - lambda_param) * l1loss(pred, target) + lambda_param * ssim_loss(pred, target)

class Gaussian2D(object):
    def __init__(self, img, K, max_iters=100, lr = 0.1, momentum=0.5):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.target_img = torch.tensor(img, dtype=torch.float32, device=device)
        # self.target_img = self.target_img.float()
        self.H, self.W, self.D = img.shape    # seld.D: number of dimentions, for colored picture, 3: R, G, B
        self.K = K  # number of components/clusters
        self.max_iters = max_iters
        self.lr = lr
        self.momentum = momentum

        self.points = None
        self.mu = None
        self.pos = None

    def _init_centers(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        np.random.seed(5) #Do Not Remove Seed
        idx = np.arange(self.H * self.W)
        selected_idx = np.random.choice(idx, size=self.K)
        si_r = selected_idx // self.W
        si_c = selected_idx % self.W
        self.color = self.target_img[si_r, si_c, :]

        si_r = torch.tensor(si_r, dtype=torch.float32)
        si_r = normalize(si_r, self.H)
        si_c = torch.tensor(si_c, dtype=torch.float32)
        si_c = normalize(si_c, self.W)
        self.mu = torch.stack((si_r, si_c), dim=-1)


        pos_r = idx // self.W
        pos_r = normalize(pos_r, self.H)
        pos_c = idx % self.W
        pos_c = normalize(pos_c, self.W)
        self.pos = torch.from_numpy(np.hstack((pos_r[:, np.newaxis], pos_c[:, np.newaxis]))).float()
        self.pos = self.pos.to(device)

    def __call__(self, abs_tol=1e-16, rel_tol=1e-16):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # Create a directory with the current date and time as its name
        directory = f"./data/images/{now}_{self.K}_{self.max_iters}"
        os.makedirs(directory, exist_ok=True)

        self._init_centers()
        model = GS(self.K, self.mu, self.color)

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)     # Adam

        pbar = tqdm(range(self.max_iters))
        prev_loss = None
        loss_list = []
        iter_list = []
        i = 0
        for it in pbar:
            predicted_img = model(self.pos)
            predicted_img = predicted_img.reshape(self.H, self.W, self.D)
            # print(f"predicted img: \n{predicted_img}\n")
            # print(f"scales: {model.scales}\n")
            # print(f"thetas: {model.thetas}\n")
            # print(f"shape of predicted img: {predicted_img.shape}\n")
            loss = combined_loss(pred=predicted_img, target=self.target_img, lambda_param=0.2)  # MSELoss
            if torch.isnan(loss):
                print(f"There is a nan in pred_img? {torch.isnan(predicted_img).any()}")
                predicted_img = prev_img
                break
            # print(f"loss = {loss}\n")
            loss_list.append(loss.item())
            iter_list.append(i)
            i += 1

            optimizer.zero_grad()
            loss.backward()
            # print(f"gradient of scales: \n{model.scales.grad}\n")
            # Example code to check for NaN in gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN in gradients of {name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if it:
                diff = loss - prev_loss
                if diff < 0:
                    diff = -diff
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    print(f"diff = {diff}, current_loss = {loss}")
                    print("Training end!")
                    break
            prev_loss = loss
            prev_img = predicted_img
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))

            if i % 5 == 0:
                generated_img = Image.fromarray((predicted_img.detach().numpy() * 255.).astype(np.uint8))
                filename = f"/iter_{i}.jpg"
                filename = directory + filename
                generated_img.save(filename)

        
        print(f"Final scales: \n{model.sigma_x}\n\n{model.sigma_y}\n")
        return predicted_img.detach().numpy(), iter_list, loss_list

# the direction of two images. Both of them are from ImageNet
# img1_dir = "./data/images/anakin-its-working-medium.png"
# img2_dir = "./data/images/gmm-example4.png"
# K=1
# max_iters=10

# image = imageio.v2.imread(img2_dir)
# image = image.astype(np.float32)

# gs_img = Gaussian2D(image, K=K, max_iters=max_iters, lr=10.)()