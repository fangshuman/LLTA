import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(sys.path)
from models import make_model
from dataset import make_loader


target_model_config = {
    'vgg16_bn': 100, 
    'resnet18': 250, 'resnet34': 250, 'resnet50': 250, 'resnet101': 250, 'resnet152': 250,
    'densenet121': 250, 'densenet169': 250, 'densenet201': 250,
    'inceptionv3': 250, 'inceptionv4': 250, 'inceptionresnetv2': 250,
}


def evaluate_with_natural_model(arch, cln_dir, adv_dir, label_path, total_num):

    model = make_model(arch=arch)
    size = model.input_size[1]
    model = model.cuda()

    _, cln_data_loader = make_loader(
        image_dir=cln_dir,
        label_path=label_path,
        batch_size=target_model_config[arch],
        total=total_num,
        size=size,
    )

    _, adv_data_loader = make_loader(
        image_dir=adv_dir,
        label_path=label_path,
        batch_size=target_model_config[arch],
        total=total_num,
        size=size,
    )

    model.eval()
    total = 0
    cln_count = 0
    adv_count = 0
    success = 0
    with torch.no_grad():
        for (cln_x, cln_y, _), (adv_x, adv_y, _) in zip(cln_data_loader, adv_data_loader):
            cln_x = cln_x.cuda()
            adv_x = adv_x.cuda()
            
            _, cln_preds = torch.max(model(cln_x), dim=1)
            _, adv_preds = torch.max(model(adv_x), dim=1)

            total += cln_x.size(0)
            cln_count += (cln_preds.detach().cpu() == cln_y).sum().item()
            adv_count += (adv_preds.detach().cpu() == cln_y).sum().item()
            success += (cln_preds != adv_preds).sum().item()

    cln_acc  = cln_count * 100.0 / total
    adv_acc  = adv_count * 100.0 / total
    suc_rate = success * 100.0 / total
    return cln_acc, adv_acc, suc_rate




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-dir", type=str, required=True)
    parser.add_argument("--cln-dir", type=str, default="images")
    parser.add_argument("--label-path", type=str, default="TrueLabel.npy")
    parser.add_argument('--target-model', nargs="+", default="")
    parser.add_argument("--total-num", type=int, default=1000)
    args = parser.parse_args()
    
    if args.target_model == "":
        args.target_model = target_model_config.keys()
    assert set(args.target_model).issubset(set(target_model_config.keys()))

    print(args)

    acc_list = []
    for target_model_name in args.target_model:
        _, _, suc_rate = evaluate_with_natural_model(
            target_model_name, 
            args.cln_dir, args.adv_dir, args.label_path, 
            args.total_num
        )
        print(f"{target_model_name}: {suc_rate}")
        acc_list.append(suc_rate)
        torch.cuda.empty_cache()

    print((sum(acc_list) - acc_list[3]) / (len(acc_list) - 1))

