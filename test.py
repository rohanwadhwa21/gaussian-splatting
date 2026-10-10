import torch
from PIL import Image
import numpy as np
import math

import time
import matplotlib.pyplot as plt

CATPATH = "/Users/rohanwadhwa/neural_graphics/A2/cat.png"
COFFEEPATH = "/Users/rohanwadhwa/neural_graphics/A2/coffee.png"
ASTROPATH = "/Users/rohanwadhwa/neural_graphics/A2/astronaut.png"


def load_target(path, max_side=256):
    img = Image.open(path).convert("RGB") # drop any alpha channel
    scale = max_side / max(img.size)
    if scale < 1:      # only shrink, never enlarge
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return torch.from_numpy(np.asarray(img)).float() / 255   # (H, W, 3) in [0, 1]



def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # TODO: build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)
    
    # N = number of gaussian blobs
    # (N, 2, 2) = a stack of 2x2 gaussians

    N = theta.shape[0]
    c, s = torch.cos(theta), torch.sin(theta)
    R = theta.new_zeros(N, 2, 2)   # same dtype/device as theta
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    
    S = torch.diag_embed(scale)
    covarianceMatrix = R @ S @ S.mT @ R.mT
    
    return covarianceMatrix



# how far is this pixel from the center of a given gaussian?
# outputs a table of every pixel against every gaussiain
def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # TODO: w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    
    dx = xy[:, 0:1] - mu[:, 0]          # (P, 1) - (N,) -> (P, N)
    dy = xy[:, 1:2] - mu[:, 1]          # (P, N)
    
    Sinv = torch.linalg.inv(Sigma)      # (N, 2, 2)
    a = Sinv[:, 0, 0]                   # (N,)
    b = Sinv[:, 0, 1]                   # (N,)
    c = Sinv[:, 1, 1]                   # (N,)
    
    q = a * dx**2 + 2 * b * dx * dy + c * dy**2
    
    return torch.exp(-0.5 * q)

def render (mu, Sigma, color, opacity, order, H, W ):
    
    # we turn the P,N table of pixels and gaussians into a single color here
    
    xy = pixel_grid(H, W)                     # (H*W, 2)
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
    alpha = opacity[None, :] * w              # (P, N)
    C = torch.zeros(H * W, 3)
    T = torch.ones(H * W)
    for i in order:                           # front to back
        a = alpha[:, i]
        C = C + (T * a)[:, None] * color[i]  
        T = T * (1 - a)
    
    return C.reshape(H, W, 3)

def pixel_grid(H, W):
    p = torch.arange(H * W)                        # pixel index: 0, 1, ..., H*W - 1
    x = p % W                                      # column: remainder after dividing by row width
    y = p // W                                     # row: how many full rows came before
    return torch.stack([x, y], dim=-1).float()     # (H*W, 2)



def fit_image(target, N, steps=2000, lr=1e-2):
    H, W = target.shape[:2]
    mu     = (torch.rand(N, 2) * torch.tensor([W, H])).requires_grad_()
    log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2)).requires_grad_()
    theta  = torch.zeros(N, requires_grad=True)
    color  = torch.zeros(N, 3, requires_grad=True)
    op_raw = torch.full((N,), -2.0, requires_grad=True)
    depth_order = torch.arange(N)

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=lr)
    psnrs = []
    for step in range(steps):
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order, H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        psnrs.append(-10 * torch.log10(loss).item())

    params = dict(mu=mu, log_s=log_s, theta=theta, color=color, op_raw=op_raw)
    return params, img.detach(), psnrs


densify_every  = 200
grad_threshold = 2e-4     # likely needs tuning (see note below)
size_threshold = 0.02     # fraction of image width
split_scale    = 1.6
prune_opacity  = 0.005


def sample_children(mu, log_s, theta):
    # Random point inside each parent's ellipse
    s = log_s.exp()                                   # (K, 2)
    z = torch.randn_like(s) * s                       # offset in blob's own axes
    c, sn = torch.cos(theta), torch.sin(theta)
    ox = c * z[:, 0] - sn * z[:, 1]                   # rotate into image axes
    oy = sn * z[:, 0] + c * z[:, 1]
    return mu + torch.stack([ox, oy], dim=-1)         # (K, 2)


def densify(gaussians, grad_mag, budget, W):
    g = {k: v.detach() for k, v in gaussians.items()}
    N = g["mu"].shape[0]

    # 1. Prune nearly transparent
    prune = g["op_raw"].sigmoid() < prune_opacity

    # 2. Under-fit candidates
    dense = (grad_mag > grad_threshold) & ~prune

    # 3. Budget: each densified Gaussian adds one net
    room = budget - (N - prune.sum().item())
    if dense.sum().item() > room:
        score = grad_mag.clone()
        score[~dense] = -1.0
        top = score.argsort(descending=True)[:max(room, 0)]
        dense = torch.zeros(N, dtype=torch.bool)
        dense[top] = True

    # 4. Clone small, split large
    small = g["log_s"].exp().max(dim=1).values <= size_threshold * W
    clone = dense & small
    split = dense & ~small

    # 5. Two children per split parent
    def child():
        return {
            "mu":     sample_children(g["mu"][split], g["log_s"][split], g["theta"][split]),
            "log_s":  g["log_s"][split] - math.log(split_scale),
            "theta":  g["theta"][split],
            "color":  g["color"][split],
            "op_raw": g["op_raw"][split],
        }
    c1, c2 = child(), child()

    # 6. Survivors + clones + children
    keep = ~prune & ~split
    new = {}
    for k in g:
        new[k] = torch.cat([g[k][keep], g[k][clone], c1[k], c2[k]]).requires_grad_()
    return new


def init_params(N, H, W, s0):
    return {
        "mu":     (torch.rand(N, 2) * torch.tensor([W, H])).requires_grad_(),
        "log_s":  torch.log(s0 * torch.ones(N, 2)).requires_grad_(),
        "theta":  torch.zeros(N, requires_grad=True),
        "color":  torch.zeros(N, 3, requires_grad=True),
        "op_raw": torch.full((N,), -2.0, requires_grad=True),
    }


def fit_image_densified(target, budget, N0=64, steps=2000, lr=1e-2, densify_until=1500):
    H, W = target.shape[:2]
    s0 = 0.5 * max(H, W) / N0 ** 0.5                  # big enough to cover the image
    params = init_params(N0, H, W, s0)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    grad_sum, n_acc = torch.zeros(N0), 0
    psnrs = []

    for step in range(steps):
        p = params
        Sigma = covariance_2d(p["log_s"].exp(), p["theta"])
        order = torch.arange(p["mu"].shape[0])
        img   = render(p["mu"], Sigma, p["color"].sigmoid(), p["op_raw"].sigmoid(), order, H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        psnrs.append(-10 * torch.log10(loss).item())

        grad_sum += p["mu"].grad.norm(dim=1)
        n_acc += 1

        if (step + 1) % densify_every == 0 and step < densify_until:
            params = densify(params, grad_sum / n_acc, budget, W)
            opt = torch.optim.Adam(list(params.values()), lr=lr)
            grad_sum, n_acc = torch.zeros(params["mu"].shape[0]), 0
            print(f"step {step}: {params['mu'].shape[0]} Gaussians")

    return params, img.detach(), psnrs

def render_fast(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W).to(mu.device)                  # (P, 2)
    alpha = opacity[None, :] * gaussian_weight(xy, mu, Sigma)   # (P, N)
    alpha = alpha[:, order]                              # columns in front-to-back order
    color = color[order]                                 # rows in the same order
    after = torch.cumprod(1 - alpha, dim=1)              # light left after each Gaussian
    T = torch.cat([torch.ones_like(after[:, :1]), after[:, :-1]], dim=1)   # light before each
    C = (T * alpha) @ color                              # (P, N) @ (N, 3) -> (P, 3)
    return C.reshape(H, W, 3)

def render_tiled(mu, Sigma, color, opacity, order, H, W, tile=16, k=3.0, pix_offset=0.0):
    dev = mu.device
    N = mu.shape[0]
    nTy, nTx = math.ceil(H / tile), math.ceil(W / tile)

    # 1. Which tiles does each Gaussian touch? (bounding box of its k-sigma ellipse)
    with torch.no_grad():
        rx = k * Sigma[:, 0, 0].sqrt()
        ry = k * Sigma[:, 1, 1].sqrt()
        tx0 = torch.floor((mu[:, 0] - rx) / tile)
        tx1 = torch.floor((mu[:, 0] + rx) / tile)
        ty0 = torch.floor((mu[:, 1] - ry) / tile)
        ty1 = torch.floor((mu[:, 1] + ry) / tile)
        t = torch.arange(nTy * nTx, device=dev)
        tile_x, tile_y = (t % nTx)[:, None].float(), (t // nTx)[:, None].float()
        hit = (tx0 <= tile_x) & (tile_x <= tx1) & (ty0 <= tile_y) & (tile_y <= ty1)   # (nTiles, N)

        # 2. Per tile: the Gaussians that hit it, front-to-back, padded to K
        rank = torch.empty(N, device=dev, dtype=torch.long)
        rank[order] = torch.arange(N, device=dev)
        key = torch.where(hit, rank[None, :], N)           # non-hits sort last
        key, idx = key.sort(dim=1)
        K = max(int((key < N).sum(dim=1).max()), 1)
        idx, valid = idx[:, :K], (key[:, :K] < N)          # (nTiles, K)

    # 3. Pixel coordinates of every tile
    ly, lx = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
    px = (tile_x * tile + lx.reshape(1, -1)).float() + pix_offset   # (nTiles, tile*tile)
    py = (tile_y * tile + ly.reshape(1, -1)).float() + pix_offset

    # 4. Same math as render_fast, but only against each tile's K Gaussians
    Sinv = torch.linalg.inv(Sigma)
    a, b, c = Sinv[idx, 0, 0], Sinv[idx, 0, 1], Sinv[idx, 1, 1]     # (nTiles, K)
    dx = px[:, :, None] - mu[idx, 0][:, None, :]                     # (nTiles, tp, K)
    dy = py[:, :, None] - mu[idx, 1][:, None, :]
    q = a[:, None] * dx**2 + 2 * b[:, None] * dx * dy + c[:, None] * dy**2
    alpha = (opacity[idx] * valid)[:, None, :] * torch.exp(-0.5 * q)
    after = torch.cumprod(1 - alpha, dim=2)
    T = torch.cat([torch.ones_like(after[..., :1]), after[..., :-1]], dim=2)
    C = (T * alpha) @ color[idx]                                     # (nTiles, tp, 3)

    # 5. Stitch tiles back into an image
    C = C.reshape(nTy, nTx, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(nTy * tile, nTx * tile, 3)
    return C[:H, :W]

device = get_device()

def load_target(path, side=128):
    img = Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)
    return (torch.from_numpy(np.asarray(img)).float() / 255).to(device)   # (H, W, 3)

def fit_image(target, N, steps=2000, lr=1e-2, seed=0):
    torch.manual_seed(seed)                      # same random init for every run
    H, W = target.shape[:2]
    dev = target.device
    mu     = (torch.rand(N, 2, device=dev) * torch.tensor([W, H], device=dev)).requires_grad_()
    log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=dev)).requires_grad_()
    theta  = torch.zeros(N, device=dev, requires_grad=True)
    color  = torch.zeros(N, 3, device=dev, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=dev, requires_grad=True)
    order  = torch.arange(N, device=dev)

    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=lr)
    for step in range(steps):
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render_fast(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Final PSNR: one clean render with the final parameters
    with torch.no_grad():
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render_fast(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        psnr  = -10 * torch.log10(((img - target) ** 2).mean())
    return img, psnr.item()

def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z)
    q = q / q.norm(dim=1, keepdim=True)               # unit length, so R is a valid rotation
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]   # each (N,)
    R = q.new_zeros(q.shape[0], 3, 3)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R                                          # (N, 3, 3)


def covariance_3d(scale, quat):
    # scale: (N, 3) positive, quat: (N, 4)
    R = quaternion_to_rotation(quat)
    S = torch.diag_embed(scale)                       # (N, 3, 3)
    return R @ S @ S.mT @ R.mT                        # (N, 3, 3)


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3), Sigma3: (N, 3, 3), one camera: R_wc (3, 3), t (3,), K (3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 1. world -> camera
    mu_cam = mu3 @ R_wc.T + t                         # (N, 3)
    xc, yc, zc = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]

    # 2. project the center
    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    mu2 = torch.stack([u, v], dim=-1)                 # (N, 2)

    # 3. covariance into camera frame
    Scam = R_wc @ Sigma3 @ R_wc.T                     # (N, 3, 3)

    # 4. Jacobian of the projection at each center, then project the covariance
    J = mu3.new_zeros(mu3.shape[0], 2, 3)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * xc / zc**2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * yc / zc**2
    Sig2 = J @ Scam @ J.mT                            # (N, 2, 2)

    depth = zc                                        # (N,) sort on this, front = smallest
    return mu2, Sig2, depth


# ---------- P5 sweep ----------
def p5():

    images = ["coffee", "astronaut", "cat"]
    counts = [256, 1024, 4096]
    results = {}

    for name in images:
        target = load_target(f"{name}.png")
        for N in counts:
            t0 = time.time()
            img, psnr = fit_image(target, N)
            results[(name, N)] = psnr
            plt.imsave(f"{name}_N{N}.png", img.clamp(0, 1).cpu().numpy())
            print(f"{name:10s} N={N:5d}  params={N*9:6d}  PSNR={psnr:.2f} dB  ({time.time()-t0:.0f}s)")

    # Table for the writeup
    print(f"\n{'image':10s}" + "".join(f"{N:>10d}" for N in counts))
    for name in images:
        print(f"{name:10s}" + "".join(f"{results[(name, N)]:10.2f}" for N in counts))

    # PSNR vs N, one curve per image
    for name in images:
        plt.plot(counts, [results[(name, N)] for N in counts], marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xticks(counts, [str(N) for N in counts])
    plt.xlabel("Number of Gaussians N")
    plt.ylabel("PSNR (dB)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("psnr_vs_N.png", dpi=150)


#p5()

import json, math, random

# ====================== P7: 3D scene reconstruction ======================

def load_cameras(root, split="frames"):
    # split = "frames" (training views) or "val_frames" (held-out views)
    with open(f"{root}/cameras.json") as f:
        meta = json.load(f)
    K = torch.tensor(meta["K"], dtype=torch.float32, device=device)
    cams = []
    for fr in meta[split]:
        img = Image.open(f"{root}/{fr['file']}").convert("RGB")
        cams.append({
            "R": torch.tensor(fr["R_wc"], dtype=torch.float32, device=device),
            "t": torch.tensor(fr["t"], dtype=torch.float32, device=device),
            "K": K, "H": meta["height"], "W": meta["width"],
            "image": (torch.from_numpy(np.array(img)).float() / 255).to(device),   # (H, W, 3)
            "file": fr["file"],
        })
    return cams


NEAR = 0.2     # ignore Gaussians closer than this to the camera (or behind it)
DILATE = 0.3   # pixels^2 added to every 2D covariance, so no blob is thinner than a pixel


def render_view(params, cam, pix_offset=0.5):
    Sig3 = covariance_3d(params["log_s"].exp(), params["quat"])                     # (N, 3, 3)
    mu2, Sig2, depth = project_gaussian(params["mu3"], Sig3, cam["R"], cam["t"], cam["K"])

    keep = depth > NEAR
    mu2, Sig2, depth = mu2[keep], Sig2[keep], depth[keep]
    Sig2 = Sig2 + DILATE * torch.eye(2, device=Sig2.device)
    color   = params["color"].sigmoid()[keep]
    opacity = params["op_raw"].sigmoid()[keep]

    order = torch.argsort(depth)                                                     # nearest first
    return render_tiled(mu2, Sig2, color, opacity, order, cam["H"], cam["W"], pix_offset=pix_offset)


def init_params_3d(N):
    quat = torch.zeros(N, 4, device=device)
    quat[:, 0] = 1.0                                                                 # identity rotation
    return {
        "mu3":    ((torch.rand(N, 3, device=device) * 2 - 1) * 1.5).requires_grad_(),  # cloud in [-1.5, 1.5]^3
        "log_s":  torch.log(0.08 * torch.ones(N, 3, device=device)).requires_grad_(),
        "quat":   quat.requires_grad_(),
        "color":  torch.zeros(N, 3, device=device, requires_grad=True),
        "op_raw": torch.full((N,), -2.0, device=device, requires_grad=True),
    }


def fit_scene(train_cams, N=4000, iters=1500, lr=1e-2, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    params = init_params_3d(N)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    for step in range(iters):
        cam  = random.choice(train_cams)
        img  = render_view(params, cam)
        loss = ((img - cam["image"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.5f}  PSNR {-10 * math.log10(loss.item()):.2f}")
    return params


def evaluate(params, cams):
    with torch.no_grad():
        imgs = [render_view(params, c) for c in cams]
    psnrs = [-10 * torch.log10(((i - c["image"]) ** 2).mean()).item() for i, c in zip(imgs, cams)]
    return sum(psnrs) / len(psnrs), psnrs, imgs


def save_side_by_side(imgs, cams, idxs, path):
    rows = [np.concatenate([imgs[i].clamp(0, 1).cpu().numpy(), cams[i]["image"].cpu().numpy()], axis=1)
            for i in idxs]                                   # each row: render | ground truth
    Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(path)


def run_p7():
    train_cams = load_cameras("spheres", "frames")
    params = fit_scene(train_cams, N=4000, iters=1500)
    mean_psnr, psnrs, imgs = evaluate(params, train_cams)
    print(f"Mean train PSNR over {len(train_cams)} views: {mean_psnr:.2f} dB")
    save_side_by_side(imgs, train_cams, [0, 12, 30], "p7_train_renders.png")
    torch.save({k: v.detach().cpu() for k, v in params.items()}, "p7_params.pt")
    return params



# ====================== P8: densification in 3D ======================
densify_every  = 200
densify_until  = 1000
grad_quantile  = 0.85    # densify the top 15% by gradient (see note below)
size_threshold = 0.06    # world units, not pixels
split_scale    = 1.6
prune_opacity  = 0.005


def sample_children_3d(mu, log_s, quat):
    z = torch.randn_like(log_s) * log_s.exp()              # offset in the blob's own axes
    R = quaternion_to_rotation(quat)
    return mu + (R @ z[:, :, None]).squeeze(-1)            # rotate into world axes


def densify_3d(params, grad_mag, budget, grad_threshold):
    g = {k: v.detach() for k, v in params.items()}
    N = g["mu3"].shape[0]
    prune = g["op_raw"].sigmoid() < prune_opacity
    dense = (grad_mag > grad_threshold) & ~prune
    room = budget - (N - int(prune.sum()))
    if int(dense.sum()) > room:
        score = grad_mag.clone(); score[~dense] = -1.0
        top = score.argsort(descending=True)[:max(room, 0)]
        dense = torch.zeros(N, dtype=torch.bool, device=dense.device); dense[top] = True
    small = g["log_s"].exp().max(dim=1).values <= size_threshold
    clone, split = dense & small, dense & ~small

    def child():
        return {
            "mu3":    sample_children_3d(g["mu3"][split], g["log_s"][split], g["quat"][split]),
            "log_s":  g["log_s"][split] - math.log(split_scale),
            "quat":   g["quat"][split], "color": g["color"][split], "op_raw": g["op_raw"][split],
        }
    c1, c2 = child(), child()
    keep = ~prune & ~split
    new = {k: torch.cat([g[k][keep], g[k][clone], c1[k], c2[k]]).requires_grad_() for k in g}
    return new, keep


def rebuild_optimizer(old_opt, old_params, new_params, keep, lr):
    # survivors keep their Adam moments; clones/children start at zero
    new_opt = torch.optim.Adam(list(new_params.values()), lr=lr)
    n_keep = int(keep.sum())
    for k, p_old in old_params.items():
        st = old_opt.state.get(p_old)
        if not st:
            continue
        p_new = new_params[k]
        n_new = p_new.shape[0] - n_keep
        pad = lambda m: torch.cat([m[keep], m.new_zeros(n_new, *m.shape[1:])])
        new_opt.state[p_new] = {"step": st["step"].clone(),
                                "exp_avg": pad(st["exp_avg"]), "exp_avg_sq": pad(st["exp_avg_sq"])}
    return new_opt


def fit_scene_densified(train_cams, budget=4000, N0=1000, iters=1500, lr=1e-2, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    params = init_params_3d(N0)
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    grad_sum, n_vis = torch.zeros(N0, device=device), torch.zeros(N0, device=device)
    for step in range(iters):
        cam  = random.choice(train_cams)
        img  = render_view(params, cam)
        loss = ((img - cam["image"]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        gnorm = params["mu3"].grad.norm(dim=1)
        grad_sum += gnorm; n_vis += (gnorm > 0).float()   # only count views where the blob was drawn
        opt.step()

        if (step + 1) % densify_every == 0 and step < densify_until:
            grad_mag = grad_sum / n_vis.clamp(min=1)
            seen = grad_mag[n_vis > 0]
            thr = torch.quantile(seen, grad_quantile).item() if len(seen) else float("inf")
            old_params, old_opt = params, opt
            params, keep = densify_3d(old_params, grad_mag, budget, thr)
            opt = rebuild_optimizer(old_opt, old_params, params, keep, lr)
            N = params["mu3"].shape[0]
            grad_sum, n_vis = torch.zeros(N, device=device), torch.zeros(N, device=device)
            print(f"step {step+1}: {N} Gaussians (pruned {int((~keep).sum())} incl. splits, thr {thr:.2e})")
        if step % 100 == 0:
            print(f"step {step:5d}  PSNR {-10 * math.log10(loss.item()):.2f}")
    return params


def run_p8():
    train_cams = load_cameras("spheres", "frames")
    val_cams   = load_cameras("spheres", "val_frames")

    plain = {k: v.to(device) for k, v in torch.load("p7_params.pt").items()}   # from run_p7
    dens  = fit_scene_densified(train_cams, budget=4000, N0=1000, iters=1500)
    torch.save({k: v.detach().cpu() for k, v in dens.items()}, "p8_params.pt")

    results = {}
    for name, p in [("plain P7", plain), ("densified P8", dens)]:
        tr, _, _ = evaluate(p, train_cams)
        va, _, val_imgs = evaluate(p, val_cams)
        results[name] = val_imgs
        print(f"{name:13s} N={p['mu3'].shape[0]:5d}  train PSNR {tr:.2f}  held-out PSNR {va:.2f}")

    i = 0   # same held-out view for both
    row = [results["plain P7"][i], results["densified P8"][i], val_cams[i]["image"]]
    row = np.concatenate([r.detach().clamp(0, 1).cpu().numpy() for r in row], axis=1)
    Image.fromarray((row * 255).astype(np.uint8)).save("p8_heldout_compare.png")   # plain | densified | GT

#run_p8()

# ====================== P9: novel-view evaluation ======================
def rot_about_axis(axis, phi):
    a = axis / axis.norm()
    Kx = torch.zeros(3, 3, device=a.device)
    Kx[0, 1], Kx[0, 2] = -a[2], a[1]
    Kx[1, 0], Kx[1, 2] = a[2], -a[0]
    Kx[2, 0], Kx[2, 1] = -a[1], a[0]
    return torch.eye(3, device=a.device) + math.sin(phi) * Kx + (1 - math.cos(phi)) * (Kx @ Kx)


def orbit_cams(cams, n_frames=12):
    # axis = normal of the plane the training cameras lie on (convention-free)
    C = torch.stack([-c["R"].T @ c["t"] for c in cams])           # camera centers in world
    axis = torch.linalg.svd(C - C.mean(0))[2][-1]
    base = cams[0]
    out = []
    for i in range(n_frames):
        Q = rot_about_axis(axis, 2 * math.pi * i / n_frames)
        out.append({**base, "R": base["R"] @ Q})                  # rotates the camera about the origin
    return out


def run_p9():
    train_cams = load_cameras("spheres", "frames")
    val_cams   = load_cameras("spheres", "val_frames")
    params = {k: v.to(device) for k, v in torch.load("p8_params.pt").items()}   # densified fit

    tr, _, _ = evaluate(params, train_cams)
    va, va_list, _ = evaluate(params, val_cams)
    print(f"train {tr:.2f} dB | held-out {va:.2f} dB (min {min(va_list):.2f}, max {max(va_list):.2f}) | gap {tr - va:.2f} dB")

    ocams = orbit_cams(train_cams, 12)
    with torch.no_grad():
        frames = [(render_view(params, c).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for c in ocams]
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save("p9_orbit.gif", save_all=True, append_images=ims[1:], duration=120, loop=0)
    Image.fromarray(np.concatenate([np.concatenate(frames[:6], 1), np.concatenate(frames[6:], 1)], 0)).save("p9_orbit_grid.png")

run_p9()




