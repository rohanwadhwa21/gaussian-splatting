"""Gaussian splatting assignment, P1-P7 in one file.

Usage (run from the folder that holds the PNGs and the spheres/ folder):
    python gs.py p4     # densified vs fixed-count fit (2D)
    python gs.py p5     # quality vs. count sweep (2D)
    python gs.py p7     # 3D reconstruction of the spheres scene
"""
import json, math, random, sys, time
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()


# ============================== P1 ==============================

def covariance_2d(scale, theta):
    N = theta.shape[0]
    c, s = torch.cos(theta), torch.sin(theta)
    R = theta.new_zeros(N, 2, 2)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    S = torch.diag_embed(scale)
    return R @ S @ S.mT @ R.mT


def gaussian_weight(xy, mu, Sigma):
    dx = xy[:, 0:1] - mu[:, 0]          # (P, N)
    dy = xy[:, 1:2] - mu[:, 1]
    Sinv = torch.linalg.inv(Sigma)
    a, b, c = Sinv[:, 0, 0], Sinv[:, 0, 1], Sinv[:, 1, 1]
    q = a * dx**2 + 2 * b * dx * dy + c * dy**2
    return torch.exp(-0.5 * q)


# ============================== P2 ==============================

def pixel_grid(H, W):
    p = torch.arange(H * W)
    return torch.stack([p % W, p // W], dim=-1).float()


def render(mu, Sigma, color, opacity, order, H, W):
    # Reference loop version (clear, slow). Training uses render_tiled below.
    xy = pixel_grid(H, W).to(mu.device)
    alpha = opacity[None, :] * gaussian_weight(xy, mu, Sigma)
    C = torch.zeros(H * W, 3, device=mu.device)
    T = torch.ones(H * W, device=mu.device)
    for i in order:
        a = alpha[:, i]
        C = C + (T * a)[:, None] * color[i]
        T = T * (1 - a)
    return C.reshape(H, W, 3)


def render_tiled(mu, Sigma, color, opacity, order, H, W, tile=16, k=3.0, pix_offset=0.0):
    # Same compositing as render(), but each 16x16 tile only sees the Gaussians whose
    # 3-sigma box touches it.
    dev = mu.device
    N = mu.shape[0]
    nTy, nTx = math.ceil(H / tile), math.ceil(W / tile)
    with torch.no_grad():
        rx = k * Sigma[:, 0, 0].sqrt()
        ry = k * Sigma[:, 1, 1].sqrt()
        tx0 = torch.floor((mu[:, 0] - rx) / tile); tx1 = torch.floor((mu[:, 0] + rx) / tile)
        ty0 = torch.floor((mu[:, 1] - ry) / tile); ty1 = torch.floor((mu[:, 1] + ry) / tile)
        t = torch.arange(nTy * nTx, device=dev)
        tile_x, tile_y = (t % nTx)[:, None].float(), (t // nTx)[:, None].float()
        hit = (tx0 <= tile_x) & (tile_x <= tx1) & (ty0 <= tile_y) & (tile_y <= ty1)
        rank = torch.empty(N, device=dev, dtype=torch.long)
        rank[order] = torch.arange(N, device=dev)
        key = torch.where(hit, rank[None, :], N)
        key, idx = key.sort(dim=1)
        K = max(int((key < N).sum(dim=1).max()), 1)
        idx, valid = idx[:, :K], (key[:, :K] < N)
    ly, lx = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
    px = (tile_x * tile + lx.reshape(1, -1)).float() + pix_offset
    py = (tile_y * tile + ly.reshape(1, -1)).float() + pix_offset
    Sinv = torch.linalg.inv(Sigma)
    a, b, c = Sinv[idx, 0, 0], Sinv[idx, 0, 1], Sinv[idx, 1, 1]
    dx = px[:, :, None] - mu[idx, 0][:, None, :]
    dy = py[:, :, None] - mu[idx, 1][:, None, :]
    q = a[:, None] * dx**2 + 2 * b[:, None] * dx * dy + c[:, None] * dy**2
    alpha = (opacity[idx] * valid)[:, None, :] * torch.exp(-0.5 * q)
    after = torch.cumprod(1 - alpha, dim=2)
    T = torch.cat([torch.ones_like(after[..., :1]), after[..., :-1]], dim=2)
    C = (T * alpha) @ color[idx]
    C = C.reshape(nTy, nTx, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(nTy * tile, nTx * tile, 3)
    return C[:H, :W]


# ============================== P3 ==============================

def load_target(path, side=128):
    img = Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)
    return (torch.from_numpy(np.array(img)).float() / 255).to(device)


def psnr_of(img, target):
    return (-10 * torch.log10(((img - target) ** 2).mean())).item()


def init_params(N, H, W, s0):
    return {
        "mu":     (torch.rand(N, 2, device=device) * torch.tensor([W, H], device=device)).requires_grad_(),
        "log_s":  torch.log(s0 * torch.ones(N, 2, device=device)).requires_grad_(),
        "theta":  torch.zeros(N, device=device, requires_grad=True),
        "color":  torch.zeros(N, 3, device=device, requires_grad=True),
        "op_raw": torch.full((N,), -2.0, device=device, requires_grad=True),
    }


MIN_SCALE = 0.3   # pixels. Thinner blobs fall between pixel centers, and a near-zero scale
                  # makes Sigma nearly singular, so torch.linalg.inv returns NaN in float32.

def render_2d(p, H, W):
    Sigma = covariance_2d(p["log_s"].exp().clamp(min=MIN_SCALE), p["theta"])
    order = torch.arange(p["mu"].shape[0], device=device)
    return render_tiled(p["mu"], Sigma, p["color"].sigmoid(), p["op_raw"].sigmoid(), order, H, W)


def fit_image(target, N, steps=2000, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    H, W = target.shape[:2]
    p = init_params(N, H, W, 0.02 * max(H, W))
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    for step in range(steps):
        loss = ((render_2d(p, H, W) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        img = render_2d(p, H, W)
    return p, img, psnr_of(img, target)


# ============================== P4 ==============================

densify_every  = 200
grad_threshold = 2e-4   # paper's value, defined for coordinates in [-1, 1]
size_threshold = 0.02
split_scale    = 1.6
prune_opacity  = 0.005


def sample_children(mu, log_s, theta):
    s = log_s.exp()
    z = torch.randn_like(s) * s
    c, sn = torch.cos(theta), torch.sin(theta)
    ox = c * z[:, 0] - sn * z[:, 1]
    oy = sn * z[:, 0] + c * z[:, 1]
    return mu + torch.stack([ox, oy], dim=-1)


def densify(gaussians, grad_mag, budget, W):
    g = {k: v.detach() for k, v in gaussians.items()}
    N = g["mu"].shape[0]
    prune = g["op_raw"].sigmoid() < prune_opacity
    # Our positions are in pixels. A move of 1 unit in [-1, 1] coords is W/2 pixels,
    # so the same gradient is W/2 times smaller per pixel. Rescale the threshold to match.
    thr = grad_threshold / (W / 2)
    dense = (grad_mag > thr) & ~prune
    room = budget - (N - prune.sum().item())
    if dense.sum().item() > room:
        score = grad_mag.clone()
        score[~dense] = -1.0
        top = score.argsort(descending=True)[:max(room, 0)]
        dense = torch.zeros(N, dtype=torch.bool, device=grad_mag.device)
        dense[top] = True
    small = g["log_s"].exp().max(dim=1).values <= size_threshold * W
    clone = dense & small
    split = dense & ~small

    def child():
        return {
            "mu":     sample_children(g["mu"][split], g["log_s"][split], g["theta"][split]),
            "log_s":  g["log_s"][split] - math.log(split_scale),
            "theta":  g["theta"][split],
            "color":  g["color"][split],
            "op_raw": g["op_raw"][split],
        }
    c1, c2 = child(), child()
    keep = ~prune & ~split
    return {k: torch.cat([g[k][keep], g[k][clone], c1[k], c2[k]]).requires_grad_() for k in g}


def fit_image_densified(target, budget, N0=64, steps=2000, lr=1e-2, densify_until=1500, seed=0):
    torch.manual_seed(seed)
    H, W = target.shape[:2]
    p = init_params(N0, H, W, 0.5 * max(H, W) / N0 ** 0.5)
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    grad_sum, n_acc = torch.zeros(N0, device=device), 0
    for step in range(steps):
        loss = ((render_2d(p, H, W) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        grad_sum += p["mu"].grad.norm(dim=1)
        n_acc += 1
        if (step + 1) % densify_every == 0 and step < densify_until:
            g = grad_sum / n_acc
            p = densify(p, g, budget, W)
            opt = torch.optim.Adam(list(p.values()), lr=lr)
            grad_sum, n_acc = torch.zeros(p["mu"].shape[0], device=device), 0
            print(f"  step {step + 1}: {p['mu'].shape[0]} Gaussians "
                  f"(median g = {g.median().item():.2e}, {(g > grad_threshold / (W / 2)).float().mean().item():.0%} above threshold)",
                  flush=True)
    with torch.no_grad():
        img = render_2d(p, H, W)
    return p, img, psnr_of(img, target)


def run_p4(name="astronaut", budget=1024):
    target = load_target(f"{name}.png")
    print(f"P4: {name}, budget {budget}")
    t0 = time.time()
    p_d, img_d, psnr_d = fit_image_densified(target, budget)
    N_final = p_d["mu"].shape[0]
    print(f"densified: N={N_final}  PSNR={psnr_d:.2f} dB  ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    _, img_f, psnr_f = fit_image(target, N_final)
    print(f"fixed:     N={N_final}  PSNR={psnr_f:.2f} dB  ({time.time() - t0:.0f}s)", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3.3))
    for a, im, title in zip(ax, [target, img_d, img_f],
                            ["target (128x128)", f"densified, N={N_final}\n{psnr_d:.2f} dB",
                             f"fixed count, N={N_final}\n{psnr_f:.2f} dB"]):
        a.imshow(im.clamp(0, 1).cpu().numpy()); a.set_title(title, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(f"p4_{name}_densified_vs_fixed.png", dpi=150, bbox_inches="tight"); plt.close()
    with open(f"p4_{name}_results.txt", "w") as f:
        f.write(f"{name} 128x128, 2000 steps, seed 0\n")
        f.write(f"densified (N0=64, budget {budget}): final N={N_final}, PSNR={psnr_d:.2f} dB\n")
        f.write(f"fixed count N={N_final}: PSNR={psnr_f:.2f} dB\n")


# ============================== P5 ==============================

def run_p5():
    images, counts, results = ["coffee", "astronaut", "cat"], [256, 1024, 4096], {}
    for name in images:
        target = load_target(f"{name}.png")
        for N in counts:
            t0 = time.time()
            _, img, psnr = fit_image(target, N)
            results[(name, N)] = psnr
            plt.imsave(f"{name}_N{N}.png", img.clamp(0, 1).cpu().numpy())
            print(f"{name:10s} N={N:5d}  PSNR={psnr:.2f} dB  ({time.time() - t0:.0f}s)", flush=True)
    for name in images:
        plt.plot(counts, [results[(name, N)] for N in counts], marker="o", label=name)
    plt.xscale("log", base=2); plt.xticks(counts, [str(N) for N in counts])
    plt.xlabel("Number of Gaussians N"); plt.ylabel("PSNR (dB)")
    plt.legend(); plt.grid(alpha=0.3); plt.savefig("psnr_vs_N.png", dpi=150, bbox_inches="tight"); plt.close()


# ============================== P6 ==============================

def quaternion_to_rotation(q):
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
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
    return R


def covariance_3d(scale, quat):
    R = quaternion_to_rotation(quat)
    S = torch.diag_embed(scale)
    return R @ S @ S.mT @ R.mT


def project_gaussian(mu3, Sigma3, R_wc, t, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu_cam = mu3 @ R_wc.T + t
    xc, yc, zc = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    mu2 = torch.stack([fx * xc / zc + cx, fy * yc / zc + cy], dim=-1)
    Scam = R_wc @ Sigma3 @ R_wc.T
    J = mu3.new_zeros(mu3.shape[0], 2, 3)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * xc / zc**2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * yc / zc**2
    return mu2, J @ Scam @ J.mT, zc


# ============================== P7 ==============================

def load_cameras(root, split="frames"):
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
            "image": (torch.from_numpy(np.array(img)).float() / 255).to(device),
            "file": fr["file"],
        })
    return cams


NEAR = 0.2
DILATE = 0.3


def render_view(params, cam, pix_offset=0.5):
    Sig3 = covariance_3d(params["log_s"].exp(), params["quat"])
    mu2, Sig2, depth = project_gaussian(params["mu3"], Sig3, cam["R"], cam["t"], cam["K"])
    keep = depth > NEAR
    mu2, Sig2, depth = mu2[keep], Sig2[keep], depth[keep]
    Sig2 = Sig2 + DILATE * torch.eye(2, device=Sig2.device)
    color = params["color"].sigmoid()[keep]
    opacity = params["op_raw"].sigmoid()[keep]
    order = torch.argsort(depth)
    return render_tiled(mu2, Sig2, color, opacity, order, cam["H"], cam["W"], pix_offset=pix_offset)


def init_params_3d(N):
    quat = torch.zeros(N, 4, device=device)
    quat[:, 0] = 1.0
    return {
        "mu3":    ((torch.rand(N, 3, device=device) * 2 - 1) * 1.5).requires_grad_(),
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
        cam = random.choice(train_cams)
        loss = ((render_view(params, cam) - cam["image"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step:5d}  loss {loss.item():.5f}  PSNR {-10 * math.log10(loss.item()):.2f}", flush=True)
    return params


def evaluate(params, cams):
    with torch.no_grad():
        imgs = [render_view(params, c) for c in cams]
    psnrs = [psnr_of(i, c["image"]) for i, c in zip(imgs, cams)]
    return sum(psnrs) / len(psnrs), psnrs, imgs


def save_side_by_side(imgs, cams, idxs, path):
    fig, ax = plt.subplots(len(idxs), 2, figsize=(5, 2.5 * len(idxs)))
    for r, i in enumerate(idxs):
        ax[r, 0].imshow(imgs[i].clamp(0, 1).cpu().numpy()); ax[r, 0].set_title(f"render {cams[i]['file']}", fontsize=9)
        ax[r, 1].imshow(cams[i]["image"].cpu().numpy()); ax[r, 1].set_title("ground truth", fontsize=9)
        ax[r, 0].axis("off"); ax[r, 1].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def run_p7():
    train_cams = load_cameras("spheres", "frames")
    t0 = time.time()
    params = fit_scene(train_cams, N=4000, iters=1500)
    mean_psnr, psnrs, imgs = evaluate(params, train_cams)
    print(f"Mean train PSNR over {len(train_cams)} views: {mean_psnr:.2f} dB  ({time.time() - t0:.0f}s)")
    save_side_by_side(imgs, train_cams, [0, 12, 30], "p7_train_renders.png")
    torch.save({k: v.detach().cpu() for k, v in params.items()}, "p7_params.pt")
    with open("p7_results.txt", "w") as f:
        f.write("N=4000, 1500 iters, Adam lr 1e-2, init: means uniform in [-1.5,1.5]^3, scale 0.08, "
                "identity rotation, gray color, opacity sigmoid(-2)\n")
        f.write(f"mean train PSNR over {len(train_cams)} views: {mean_psnr:.2f} dB\n")
        for c, s in zip(train_cams, psnrs):
            f.write(f"{c['file']}: {s:.2f}\n")


if __name__ == "__main__":
    print("device:", device)
    for part in sys.argv[1:]:
        {"p4": run_p4, "p5": run_p5, "p7": run_p7}[part]()
