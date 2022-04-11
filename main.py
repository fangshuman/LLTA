import os
import time
import argparse
from tqdm import tqdm

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Gaussian, get_Gaussian_kernel
from models import make_Gamma_Wrap, make_model
from dataset import make_loader, save_image


gamma_num_dic = {
    "resnet18": 4,  
    "resnet34": 12, 
    "resnet50": 12, 
    "resnet101": 29,
    "resnet152": 46,

    "densenet121": 54,
    "densenet169": 78,
    "densenet201": 94,
}
GMIN = 0.0
GMAX = 1e5



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack-method", type=str, default="LLTA")
    parser.add_argument("--input-dir", type=str, default="images")
    parser.add_argument("--label-path", type=str, default="TrueLabel.npy")

    parser.add_argument("--source-model", default="resnet50")
    parser.add_argument("--total-num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--nsample", type=int, default=5)

    parser.add_argument("--task-num", type=int, default=5)
    parser.add_argument("--spt-size", type=int, default=20)
    parser.add_argument("--qry-size", type=int, default=10)
    parser.add_argument("--spt-prob-m", type=float, default=0.5)
    parser.add_argument("--qry-prob-m", type=float, default=0.5)
    parser.add_argument("--spt-prob-d", type=float, default=0.5)
    parser.add_argument("--qry-prob-d", type=float, default=0.5)
    parser.add_argument("--spt-region", type=float, default=0.1)
    parser.add_argument("--qry-region", type=float, default=0.1)
    parser.add_argument("--msa", type=float, default=16, help="meta learning step alpha")
    parser.add_argument("--msb", type=float, default=1.6, help="meta learning step beta")

    parser.add_argument("--eps", type=float, default=16)
    parser.add_argument("--eps-iter", type=float, default=1.6)
    parser.add_argument("--niters", type=int, default=10)

    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    return args


def get_l2grad(x, y, model, gammas):
    zero_delta = torch.zeros_like(x)
    zero_delta.requires_grad_()
    loss = F.cross_entropy(model(x + zero_delta, gammas), y, reduction='sum')
    loss.backward()
    grad = zero_delta.grad.data
    l2grad = grad.norm(p=2, dim=(1,2,3)) 
    return l2grad

def optimize_parameter(x, y, model, gammas, nsample):
    # to ensure x and y are leaf nodes
    x = x.data
    y = y.data
    device = x.device

    cur_l2grad = get_l2grad(x, y, model, gammas)

    gamma_delta = gaussian.sample((nsample, x.shape[0], gammas.shape[-1]), device=device)
    prob_q = gaussian.prob(gamma_delta)
    new_l2grad = torch.stack([
        get_l2grad(x, y, model, gammas=torch.clamp(gammas + gd, GMIN, GMAX))
        for gd in gamma_delta
    ])

    diff = new_l2grad - cur_l2grad
    prob_p = ((-diff / 1.0).exp() * (diff < 0)).unsqueeze(-1)

    opt_gamma_delta = ((prob_p / prob_q) * gamma_delta).sum(0)
    Z = (prob_p / prob_q).sum(0)
    return opt_gamma_delta / (Z + 1e-12)

def create_model_task_set(gammas, set_size, prob, region):
    device = gammas.device
    aug_size = int(set_size*prob)
    gamma_sets = []
    for gamma in gammas:
        gamma_set = gamma.repeat(set_size, 1)
        aug_delta = (torch.rand_like(gamma_set) - 0.5) * 2 * region
        aug_mask = (torch.rand((set_size, 1), device=device) < prob)
        gamma_sets.append(gamma_set + aug_mask*aug_delta)
    # import ipdb; ipdb.set_trace()
    gamma_sets = torch.cat(gamma_sets)
    gamma_sets = torch.clamp(gamma_sets, GMIN, GMAX)
    return gamma_sets

def create_data_task_set(x, set_size, prob):
    def input_diversity(img):
        size = img.size(2)
        resize = int(size / 0.875)

        rnd = torch.randint(size, resize + 1, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
        h_rem = resize - rnd
        w_hem = resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (size, size), mode="nearest")
        return padded

    aug_ls = []
    for _ in range(set_size):
        p = torch.rand(1).item()
        if p >= prob:
            aug_ls.append(x)
        else:
            aug_ls.append(input_diversity(x))
    aug_ls = torch.stack(aug_ls)
    inputs = torch.cat([
        aug_ls[:, i, :, :, :]
        for i in range(aug_ls.shape[1])
    ])
    return inputs


def perturb(x, y, model, args):
    device = x.device
    epoch = args.epoch
    niters = args.niters
    eps = args.eps
    eps_iter = args.eps_iter

    gamma_num = gamma_num_dic[args.source_model]

    nsample = args.nsample
    task_num = args.task_num
    spt_size = args.spt_size
    qry_size = args.qry_size
    spt_prob_m = args.spt_prob_m
    qry_prob_m = args.qry_prob_m
    spt_prob_d = args.spt_prob_d
    qry_prob_d = args.qry_prob_d
    spt_region = args.spt_region
    qry_region = args.qry_region
    step_a = args.msa
    step_b = args.msb
    m = 1.0       # Combine with MI
    nsig = 3      # Combine with TI
    kernlen = 7   # Combine with TI
    
    batch_size, c, w, h = x.shape

    delta = torch.zeros_like(x)
    delta.requires_grad_()
    g = torch.zeros_like(x)

    for i in range(niters):
        gammas = torch.full((batch_size, gamma_num), 0.5, dtype=torch.float32, device=device) # reinit
        for epc in range(epoch):
            gamma_delta = optimize_parameter(x + delta, y, model, gammas, nsample=nsample)
            gammas = torch.clamp(gammas + gamma_delta, GMIN, GMAX)

        # create suppory set
        spt_gammas = create_model_task_set(
            gammas, 
            set_size=spt_size,
            prob=spt_prob_m,
            region=spt_region,
        )
        spt_data = create_data_task_set(
            (x + delta).data,
            set_size=spt_size,
            prob=spt_prob_d,
        )

        # create query set
        qry_gammas = create_model_task_set(
            gammas, 
            set_size=qry_size,
            prob=qry_prob_m,
            region=qry_region,
        )
        qry_data = create_data_task_set(
            (x + delta).data,
            set_size=qry_size,
            prob=qry_prob_d,
        )

        y_repeat = y.unsqueeze(0).repeat(qry_size, 1)
        y_expand = torch.cat([
            y_repeat[:, ii]
            for ii in range(batch_size)
        ])

        grads = torch.zeros_like(x)
        for _ in range(task_num):
            # sample batch of tasks
            select_idx = torch.as_tensor([
                random.sample(range(bs*spt_size, (bs+1)*spt_size), qry_size)
                for bs in range(batch_size)
            ])
            select_idx = select_idx.view(-1)
            spt_batch_gammas = spt_gammas[select_idx]
            spt_batch_x = spt_data[select_idx]

            # get gradient on batch support set
            spt_delta = torch.zeros_like(spt_batch_x)
            spt_delta.requires_grad_()  
            spt_loss = F.cross_entropy(model(spt_batch_x.data + spt_delta, spt_batch_gammas), y_expand, reduction='sum')
            spt_loss.backward()
            spt_delta.data = spt_delta.data + step_a * spt_delta.grad.sign()
            spt_delta.data = torch.clamp(spt_delta.data, -eps, eps)
            spt_delta.data = torch.clamp(spt_batch_x.data + spt_delta, 0., 1.) - spt_batch_x.data 
            # import ipdb; ipdb.set_trace()

            # accumulate loss on query set
            qry_loss = F.cross_entropy(model(qry_data.data + spt_delta, qry_gammas), y_expand, reduction='sum')
            qry_loss.backward()

            grads += spt_delta.grad.data.view(qry_size, batch_size, c, w, h).sum(0)
            spt_delta.grad.data.zero_()


        # update delta 
        # combine with MI-FGSM
        if "mi" in args.attack_method:
            g = m * g + grads.norm(p=2, dim=(1,2,3)) 
            grads = g
        
        # combine with TI-FGSM
        if "ti" in args.attack_method:
            kernel = get_Gaussian_kernel(nsig=nsig, kernlen=kernlen)
            kernel = torch.FloatTensor(kernel).expand(x.size(1), x.size(1), kernlen, kernlen)
            kernel = kernel.to(x.device)
            grads = F.conv2d(grads, kernel, padding=kernlen // 2)

        delta.data = delta.data + step_b * grads.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x.data + delta, 0., 1.) - x

    x_adv = torch.clamp(x + delta, 0., 1.)
    return x_adv


if __name__ == '__main__':

    args = get_args()
    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # as reproducibility docs
    torch.manual_seed(seed)  # as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # as reproducibility docs
    torch.backends.cudnn.deterministic = True  # as reproducibility docs

    args.eps /= 255.0
    args.eps_iter /= 255.0
    args.msa /= 255.0
    args.msb /= 255.0
    print(args)
    
    # create dataloader
    batch_size = args.batch_size
    img_list, data_loader = make_loader(
        image_dir=args.input_dir,
        label_path=args.label_path,
        batch_size=args.batch_size,
        total=args.total_num,
        size=224,
    )

    # define gaussian distribution
    global gaussian 
    gaussian = Gaussian(loc=0., scale=args.sigma)

    arch = args.source_model
    ori_model = make_model(arch).cuda().eval()
    model = make_Gamma_Wrap(arch).cuda().eval()

    total_time = 0
    for input, _, index in tqdm(data_loader):
        input = input.cuda()
        with torch.no_grad():
            _, label = torch.max(ori_model(input), dim=1)

        start_time = time.time()
        input_adv = perturb(input, label, model, args)
        total_time += (time.time() - start_time)
        save_image(input_adv.detach().cpu().numpy(), index, img_list, args.output_dir)

