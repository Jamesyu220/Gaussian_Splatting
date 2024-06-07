import imageio
import numpy as np
# import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
SMALL_CONST = 1e-8

class GS(nn.Module):
    def __init__(self, K, mu, points):
        super(GS, self).__init__()
        # self.img = img
        # H, W, _ = img.shape

        # self.alpha = img[:, :, 3]
        # self.alpha = nn.Parameter(self.alpha.float() / 255.)

        # self.color = img[:, :, :3]
        # self.color = nn.Parameter(self.color.float())

        # np.random.seed(5) #Do Not Remove Seed
        # idx = np.arange(H * W)
        # pos_r = idx // W
        # pos_c = idx % W
        # self.pos = torch.from_numpy(np.hstack((pos_r[:, np.newaxis], pos_c[:, np.newaxis]))[:, np.newaxis, :]).float()
        # self.pos.requires_grad_(False)

        # selected_idx = np.random.choice(idx, size=K)
        # si_r = selected_idx // W
        # si_c = selected_idx % W
        # si_r = si_r[:, np.newaxis]
        # si_c = si_c[:, np.newaxis]
        # self.mu = torch.from_numpy(np.hstack((si_r, si_c))[np.newaxis, :, :]).float()
        # self.mu = nn.Parameter(self.mu)
        self.K = K

        self.mu = torch.from_numpy(mu).to(device)
        self.mu = self.mu.float()
        self.mu = nn.Parameter(self.mu)

        if points.shape[1] == 3:
            self.include_opc = False
            self.alpha = torch.ones(K).float()
        else:
            self.include_opc = True
            self.alpha = points[:, 3]
            # self.alpha = self.alpha.unsqueeze(0)
            self.alpha = self.alpha / 255.

        self.alpha = nn.Parameter(self.alpha).to(device)   # , requires_grad=False)

        self.color = points[:, :3]
        self.color = nn.Parameter(self.color).to(device)   # , requires_grad=False)

        self.scales = nn.Parameter(torch.ones(K, 2).float() * 10.0).to(device)
        self.thetas = nn.Parameter(torch.zeros(K).float()).to(device)

    def forward(self, pos):
        
        # diff = pos - self.mu
        # cos_thetas = torch.cos(self.thetas)
        # sin_thetas = torch.sin(self.thetas)
        # R = torch.stack([
        #     torch.stack((cos_thetas, -sin_thetas), dim=-1),
        #     torch.stack((sin_thetas, cos_thetas), dim=-1)
        # ], dim=-2)

        
        # # scales_inv = torch.reciprocal(self.scales)
        # # while torch.isnan(scales_inv).any():
        # # self.scales += SMALL_CONST
        # scales_inv = torch.reciprocal(self.scales + SMALL_CONST)
            
        # sigma_inv = torch.einsum('kti, kij -> ktj', torch.einsum('kij, kj -> kij', R, scales_inv), torch.transpose(R, 1, 2)).float()
        # expo = -0.5 * torch.einsum('nkj, nkj -> nk', torch.einsum('nki, kij -> nkj', diff, sigma_inv), diff)
        # alpha = self.alpha * torch.exp(expo)
        self.alpha = nn.Parameter(torch.clamp(self.alpha, min=0., max=1.))
        self.color = nn.Parameter(torch.clamp(self.color, min=0., max=255.))
        self.scales = nn.Parameter(torch.clamp(self.scales, min=.1))

        N = pos.shape[0]
        pred_color = torch.zeros(N, 3).to(device)
        pred_alpha = torch.zeros(N, 1).to(device)
        find_nan = False
        for k in range(self.K):
            diff = pos - self.mu[k:(k+1), :]
            # if torch.isnan(diff).any():
            #     print("NAN in diff\n")
            cos_theta = torch.cos(self.thetas[k])
            sin_theta = torch.sin(self.thetas[k])
            R = torch.stack([
                torch.stack((cos_theta, -sin_theta), dim=-1),
                torch.stack((sin_theta, cos_theta), dim=-1)
            ], dim=-2)
            
            # if torch.isnan(R).any():
            #     print("NAN in R\n")
        
            # scales_inv = torch.reciprocal(self.scales)
            # while torch.isnan(scales_inv).any():
            # self.scales += SMALL_CONST
            # scales_inv = torch.reciprocal(self.scales[k, :])
            diag_inv_squared_scales = torch.diag_embed((1.0 / (self.scales[k, :] ** 2)))
            # if torch.isnan(scales_inv).any():
            #     print("NAN in scales_inv\n")
            
            sigma_inv = torch.einsum('ti, ij, jk -> tk', R, diag_inv_squared_scales, R.transpose(0, 1)).float()
            if not find_nan and torch.isinf(sigma_inv).any():
                find_nan = True
                print("INF in sigma_inv\n")
            expo = -0.5 * torch.einsum('nj, nj -> n', torch.einsum('ni, ij -> nj', diff, sigma_inv), diff)
            if not find_nan and torch.isinf(expo).any():
                find_nan = True
                print("INF in expo\n")
            alpha_k = self.alpha[k] * torch.exp(expo)
            alpha_k = alpha_k.unsqueeze(1)
            color_k = self.color[k:(k+1), :]
            if not find_nan and torch.isinf(alpha_k).any():
                find_nan = True
                print(f"NAN in self.alpha? {torch.isinf(self.alpha).any()}\n")
                print(f"scales = \n{self.scales[k, :]}\n")
                print(f"R = \n{R}\n")
                print(f"mu = \n{self.mu[k:(k+1), :]}\n")
                print("INF in alpha_k\n")
            if not find_nan and torch.isinf(color_k).any():
                find_nan = True
                print("INF in color_k\n")

            new_alpha = pred_alpha + alpha_k * (1. - pred_alpha)
            if not find_nan and torch.isnan(new_alpha).any():
                print(f"alpha_k = {alpha_k}\n")
                find_nan = True
                print("NAN in alpha\n")
            # pred_color = (pred_color * pred_alpha + color_k * alpha_k * (1. - pred_alpha)) / new_alpha
            # while torch.isnan(pred_color).any():
            # new_alpha += SMALL_CONST
            pred_color = (pred_color * pred_alpha + color_k * alpha_k * (1. - pred_alpha)) / (new_alpha + SMALL_CONST)
            if not find_nan and torch.isnan(pred_color).any():
                find_nan = True
                print("NAN in pred_color\n")

            pred_alpha = new_alpha

            pred_color = torch.clamp(pred_color, min=0., max=255.)
            pred_alpha = torch.clamp(pred_alpha, min=0., max=1.)

        if self.include_opc:
            pred_img = torch.cat((pred_color, pred_alpha * 255.), dim=-1)
        else:
            pred_img = pred_color

        return pred_img

class Gaussian2D(object):
    def __init__(self, X, K, max_iters=100, lr = 0.1, momentum=0.5):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.target_img = torch.from_numpy(X)
        # self.target_img = self.target_img.float()
        self.H, self.W, self.D = X.shape    # seld.D: number of dimentions, for colored picture, 4: R, G, B, a
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
        self.points = self.target_img[si_r, si_c, :]
        si_r = si_r[:, np.newaxis]
        si_c = si_c[:, np.newaxis]
        self.mu = np.hstack((si_r, si_c))

        pos_r = idx // self.W
        pos_c = idx % self.W
        self.pos = torch.from_numpy(np.hstack((pos_r[:, np.newaxis], pos_c[:, np.newaxis]))).to(device)
        self.pos = self.pos.float()

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
        
        self._init_centers()
        model = GS(self.K, self.mu, self.points)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)     # Adam

        pbar = tqdm(range(self.max_iters))
        prev_loss = None
        loss_list = []
        iter_list = []
        i = 0
        for it in pbar:
            predicted_img = model(self.pos)
            # print(f"predicted img: \n{predicted_img}\n")
            # print(f"scales: {model.scales}\n")
            # print(f"thetas: {model.thetas}\n")
            # print(f"shape of predicted img: {predicted_img.shape}\n")
            loss = torch.nn.L1Loss()(predicted_img, self.target_img.reshape(-1, self.D))   # MSELoss
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
                    print("Training end!")
                    break
            prev_loss = loss
            prev_img = predicted_img
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        
        print(f"Final scales: \n{model.scales}\n")
        predicted_img = predicted_img.reshape(self.H, self.W, self.D)
        return predicted_img.detach().numpy(), iter_list, loss_list

# the direction of two images. Both of them are from ImageNet
# img1_dir = "./data/images/anakin-its-working-medium.png"
# img2_dir = "./data/images/gmm-example4.png"
# K=2000
# max_iters=10

# image = imageio.v2.imread(img2_dir)
# image = image.astype(np.float32)

# gs_img = Gaussian2D(image, K=K, max_iters=max_iters, lr=10.)()