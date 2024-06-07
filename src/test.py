import numpy as np
import torch
import imageio

image = imageio.v2.imread("./data/images/gmm-example5.png").astype(np.float32)

SMALL_CONST = 1e-8
H = 2
W = 2
D = 3
print(f"target image: {image[:H, :W, :D]}\n")

np.random.seed(5) #Do Not Remove Seed
idx = np.arange(H * W)
K = 2
selected_idx = np.random.choice(idx, size=K, replace=False)
si_r = selected_idx // W
si_c = selected_idx % W
alpha = torch.from_numpy(image[si_r, si_c, 3]).float() / 255.
# print(f"alpha = {alpha}\n")
si_r = si_r[:, np.newaxis]
si_c = si_c[:, np.newaxis]
mu = torch.from_numpy(np.hstack((si_r, si_c))).float()   #[:, np.newaxis, :]
print(f"mu = {mu}\n")
scales = torch.ones(K, 2)
scales[0, 0] = 2.
thetas = torch.zeros(K)
thetas[0] = np.pi/4

pos_r = idx // W
pos_c = idx % W
pos = torch.from_numpy(np.hstack((pos_r[:, np.newaxis], pos_c[:, np.newaxis]))).float()
# print(f"pos = {pos}\n")
N = pos.shape[0]
pred_alpha = torch.zeros(N, 1)
for k in range(K):
    diff = pos - mu[k:(k+1), :]
    print(f"diff = {diff}\n")
    cos_thetas = torch.cos(thetas[k])
    sin_thetas = torch.sin(thetas[k])
    R = torch.stack([
        torch.stack((cos_thetas, -sin_thetas), dim=-1),
        torch.stack((sin_thetas, cos_thetas), dim=-1)
    ], dim=-2)
    print(f"R = \n{R}\n")
    scales_inv = torch.reciprocal(scales[k, :] + SMALL_CONST)
    sigma_inv = torch.einsum('ti, ij -> tj', torch.einsum('ij, j -> ij', R, scales_inv), torch.transpose(R, 0, 1)).float()
    print(f"sigma_inv = {sigma_inv}\n")
    print(f"diff @ sigma = {torch.einsum('ni, ij -> nj', diff, sigma_inv)}\n")
    expo = torch.einsum('nj, nj -> n', torch.einsum('ni, ij -> nj', diff, sigma_inv), diff) * (-0.5)
    print(f"expo = {expo}\n")
    print(f"alpha[k] = {alpha[k]}\n")
    alpha_k = alpha[k] * torch.exp(expo)
    alpha_k = alpha_k.unsqueeze(1)
    print(f"alpha_k = {alpha_k}\n")
    pred_alpha = pred_alpha + alpha_k * (1. - pred_alpha)

print(f"predicted alpha: {pred_alpha}\n")