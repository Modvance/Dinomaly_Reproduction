# Dinomaly (MVTec) experiment script using **official DINOv3** backbone with native tokens.
# This file is intentionally separate from `dinomaly_mvtec_uni.py` to avoid disturbing the baseline code.

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import ConcatDataset

from models.uad import ViTill
from models.dinov3_native import DinoV3NativeTokenEncoder, load_dinov3_backbone
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention2
from dataset import get_data_transforms
from dataset import MVTecDataset
import argparse
from utils import evaluation_batch, global_cosine, global_cosine_hm_percent, WarmCosineScheduler
from functools import partial
from optimizers import StableAdamW
import warnings
import logging
import time

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def format_mean_summary(i_auroc, i_ap, i_f1, p_auroc, p_ap, p_f1, p_aupro):
    return (
        'I-AUROC      {:.2f}\n'
        'I-AP         {:.2f}\n'
        'I-F1         {:.2f}\n'
        'P-AUROC      {:.2f}\n'
        'P-AP         {:.2f}\n'
        'P-F1         {:.2f}\n'
        'P-AUPRO      {:.2f}\n'
    ).format(
        i_auroc * 100, i_ap * 100, i_f1 * 100,
        p_auroc * 100, p_ap * 100, p_f1 * 100, p_aupro * 100
    )


def format_training_efficiency(total_time_s, total_iters, batch_size):
    time_per_iter_s = total_time_s / total_iters if total_iters else 0
    iters_per_sec = total_iters / total_time_s if total_time_s > 0 else 0
    samples_per_sec = iters_per_sec * batch_size if total_time_s > 0 else 0
    return (
        'Training efficiency:\n'
        '  total_time_s       {:.2f}\n'
        '  total_iters        {}\n'
        '  batch_size         {}\n'
        '  time_per_iter_s    {:.4f}\n'
        '  iters_per_sec      {:.4f}\n'
        '  samples_per_sec   {:.2f}\n'
    ).format(total_time_s, total_iters, batch_size, time_per_iter_s, iters_per_sec, samples_per_sec)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list, args, device, gpu_ids):
    setup_seed(1)

    total_iters = 10000
    batch_size = args.batch_size
    image_size = args.image_size
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )

    # ---- Official DINOv3 backbone (native tokens) ----
    # DINOv3 uses patch_size=16, so crop_size must be divisible by 16 (default here: 384).
    # target_layers are block indices for get_intermediate_layers.
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    backbone = load_dinov3_backbone(
        hub_model=args.dinov3_model,
        source=args.dinov3_source,
        repo_or_dir=args.dinov3_repo,
        pretrained=True,
    )
    encoder = DinoV3NativeTokenEncoder(backbone=backbone, target_layers=target_layers)

    embed_dim = getattr(backbone, "embed_dim", 768)
    num_heads = getattr(backbone, "num_heads", 12)

    # freeze backbone (as in baseline)
    for p in backbone.parameters():
        p.requires_grad = False

    num_decoder_layers = 8
    # group reconstruction (same idea as baseline)
    _indices_enc = np.arange(len(target_layers))
    _splits_enc = np.array_split(_indices_enc, args.num_fuse_groups)
    fuse_layer_encoder = [list(s) for s in _splits_enc]
    _indices_dec = np.arange(num_decoder_layers)
    _splits_dec = np.array_split(_indices_dec, args.num_fuse_groups)
    fuse_layer_decoder = [list(s) for s in _splits_dec]

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=args.bn_dropout)])
    attn_cls = LinearAttention2 if args.decoder_attn == 'linear' else Attention
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=attn_cls)
        for _ in range(num_decoder_layers)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0,
    ).to(device)

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print_fn('Using DataParallel on GPUs {}'.format(gpu_ids))

    trainable = nn.ModuleList([bottleneck, decoder])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('DINOv3 native config: model={} crop_size={} target_layers={}'.format(
        args.dinov3_model, crop_size, target_layers))
    print_fn('Ablation config: bn_dropout={} decoder_attn={} num_fuse_groups={} loss_type={}'.format(
        args.bn_dropout, args.decoder_attn, args.num_fuse_groups, args.loss_type))

    final_means = None
    it = 0
    start_t = time.perf_counter()
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            optimizer.zero_grad()

            en, de = model(img)
            if args.loss_type == 'loose':
                p_final = 0.9
                p = min(p_final * it / 1000, p_final)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            else:
                loss = global_cosine(en, de)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())

            if (it + 1) % 5000 == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=args.workers)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                mean_auroc_sp = np.mean(auroc_sp_list)
                mean_ap_sp = np.mean(ap_sp_list)
                mean_f1_sp = np.mean(f1_sp_list)
                mean_auroc_px = np.mean(auroc_px_list)
                mean_ap_px = np.mean(ap_px_list)
                mean_f1_px = np.mean(f1_px_list)
                mean_aupro_px = np.mean(aupro_px_list)
                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        mean_auroc_sp, mean_ap_sp, mean_f1_sp,
                        mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
                final_means = (mean_auroc_sp, mean_ap_sp, mean_f1_sp, mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px)
                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    total_s = time.perf_counter() - start_t
    log_path = os.path.join(args.save_dir, args.save_name, 'log.txt')
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            if final_means is not None:
                f.write(format_mean_summary(*final_means))
            f.write(format_training_efficiency(total_s, total_iters, batch_size))
    except Exception as e:
        print('Failed to append summary to {}: {}'.format(log_path, e))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = os.environ.get('CUDA_LAUNCH_BLOCKING', '1')

    parser = argparse.ArgumentParser(description='Dinomaly + official DINOv3 (native tokens) experiment')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='exp_dinov3_native')

    # DINOv3 official hub loading
    parser.add_argument('--dinov3_source', type=str, default='github', choices=('github', 'local'),
                        help="torch.hub source: github (default) or local")
    parser.add_argument('--dinov3_repo', type=str, default='facebookresearch/dinov3',
                        help="Repo or local dir for torch.hub (default: facebookresearch/dinov3)")
    parser.add_argument('--dinov3_model', type=str, default='dinov3_vitb16',
                        help="Hub entrypoint, e.g. dinov3_vits16, dinov3_vitb16, dinov3_vitl16")

    # Data / training (DINOv3 patch16 => crop_size must be divisible by 16)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=384, help='Must be divisible by 16 for DINOv3')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)

    # Keep your existing ablation knobs
    parser.add_argument('--bn_dropout', type=float, default=0.2)
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=('linear', 'softmax'))
    parser.add_argument('--num_fuse_groups', type=int, default=2, choices=(1, 2, 4, 8))
    parser.add_argument('--loss_type', type=str, default='loose', choices=('loose', 'full'))

    parser.add_argument('--gpus', type=str, default='1')
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        gpu_ids = []
    else:
        device = torch.device('cuda', gpu_ids[0] if gpu_ids else 0)

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn('device={} gpus={}'.format(device, gpu_ids if gpu_ids else 'cpu only'))

    train(item_list, args, device=device, gpu_ids=gpu_ids)

