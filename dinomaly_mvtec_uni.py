# Dinomaly MVTec unified training script.
#
# 消融实验：通过命令行参数切换四个关键组件，建议每次用不同 --save_name 便于对比。
#   1) 预训练 backbone: --encoder_name (dinov2reg_vit_base_14 | mae_vit_base_16 | ...)
#   2) 噪声瓶颈:       --bn_dropout (0.2 默认 | 0 无dropout | 0.4 Real-IAD)
#   3) 解码器注意力:   --decoder_attn (linear 默认 | softmax)
#   4) 分组重建:       --num_fuse_groups (2 默认 | 1 不分组 | 8 逐层)
#   5) 宽松损失:       --loss_type (loose 默认 | full 全量余弦)
#
# 多卡/效率（例如 7x RTX 3090）：建议 --gpus 0,1,2,3,4,5,6 --batch_size 64 --workers 12 --amp
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import sys
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import time

# 引入 TailedCore 的类别基数估计工具
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TAILEDCORE_ROOT = os.path.join(PROJECT_ROOT, '..', 'TailedCore')
if TAILEDCORE_ROOT not in sys.path:
    sys.path.append(TAILEDCORE_ROOT)

from src import class_size

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def format_mean_summary(i_auroc, i_ap, i_f1, p_auroc, p_ap, p_f1, p_aupro):
    """Format mean metrics as percentage summary block (same as mvtec_log.txt tail)."""
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
    """Format training time and throughput for log tail."""
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed, cudnn_benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark


def train(item_list, run_args=None, device=None, gpu_ids=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if gpu_ids is None:
        gpu_ids = [0] if torch.cuda.is_available() else []
    use_amp = getattr(run_args, 'amp', False) if run_args else False
    cudnn_benchmark = getattr(run_args, 'cudnn_benchmark', False) if run_args else False
    num_workers = getattr(run_args, 'workers', 4) if run_args else 4
    setup_seed(1, cudnn_benchmark=cudnn_benchmark)

    total_iters = 10000
    batch_size = getattr(run_args, 'batch_size', 16) if run_args else 16
    image_size = 448
    crop_size = 392

    # image_size = 448
    # crop_size = 448

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
    # 论文原配置：num_workers=4，无 pin_memory / persistent_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True)
    # test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #                         for test_data in test_data_list]

    # 预训练 backbone：由 --encoder_name 指定，便于消融实验切换
    # ViT: dinov2reg_vit_{small,base,large}_14, dinov2_vit_base_14, dino_vit_base_16, mae_vit_base_16, ...
    # ResNet: resnet50, wide_resnet50_2, resnet101, ...
    encoder_name = run_args.encoder_name if run_args is not None else 'dinov2reg_vit_base_14'
    bn_dropout = getattr(run_args, 'bn_dropout', 0.2) if run_args else 0.2
    decoder_attn = getattr(run_args, 'decoder_attn', 'linear') if run_args else 'linear'
    num_fuse_groups = getattr(run_args, 'num_fuse_groups', 2) if run_args else 2
    loss_type = getattr(run_args, 'loss_type', 'loose') if run_args else 'loose'

    num_decoder_layers = 8
    use_resnet = 'resnet' in encoder_name.lower()
    if use_resnet:
        target_layers = [0, 1, 2, 3]
        num_encoder_layers = 4
    else:
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        num_encoder_layers = 8
    _indices_enc = np.arange(num_encoder_layers)
    _splits_enc = np.array_split(_indices_enc, num_fuse_groups)
    fuse_layer_encoder = [list(s) for s in _splits_enc]
    _indices_dec = np.arange(num_decoder_layers)
    _splits_dec = np.array_split(_indices_dec, num_fuse_groups)
    fuse_layer_decoder = [list(s) for s in _splits_dec]

    encoder = vit_encoder.load(encoder_name)

    if use_resnet:
        embed_dim, num_heads = 768, 12
    elif 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large, or resnet.")

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bn_dropout))

    bottleneck = nn.ModuleList(bottleneck)

    attn_cls = LinearAttention2 if decoder_attn == 'linear' else Attention
    for i in range(num_decoder_layers):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=attn_cls)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0,
        return_global_embeddings=True,
    )
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print_fn('Using DataParallel on GPUs {}'.format(gpu_ids))
    trainable = nn.ModuleList([bottleneck, decoder])
    if use_resnet:
        for p in encoder.backbone.parameters():
            p.requires_grad = False
        trainable.append(encoder)

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('Ablation config: encoder={} bn_dropout={} decoder_attn={} num_fuse_groups={} loss_type={}'.format(
        encoder_name, bn_dropout, decoder_attn, num_fuse_groups, loss_type))
    print_fn('Efficiency: workers={} cudnn_benchmark={} amp={}'.format(num_workers, cudnn_benchmark, use_amp))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    final_means = None
    it = 0
    train_start_time = time.perf_counter()
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # 1) 先通过冻结的 ViT 编码器，提取每个样本的全局嵌入，用于 Tail 类别基数估计
            with torch.no_grad():
                if isinstance(model, nn.DataParallel):
                    global_emb = model.module.extract_global_embeddings(img)
                else:
                    global_emb = model.extract_global_embeddings(img)

                class_sizes = class_size.sample_few_shot(
                    X=global_emb.detach(),
                    th_type='symmin',
                    vote_type='mean',
                    return_class_sizes=True,
                )
                num_samples_per_class = class_size.predict_num_samples_per_class(class_sizes)
                K_max = class_size.predict_max_K(num_samples_per_class, percentile=0.15)

                # 将 κ_i → 动态 Dropout 丢弃率 p_i
                kappa = class_sizes.to(global_emb.device)
                kappa_clamped = torch.clamp(kappa, min=float(K_max))
                kappa_max = kappa_clamped.max()
                if kappa_max <= K_max:
                    t = torch.zeros_like(kappa_clamped)
                else:
                    t = (kappa_clamped - float(K_max)) / (kappa_max - float(K_max))
                p_tail, p_head = 0.05, 0.5
                dropout_rates = p_tail + t * (p_head - p_tail)

                # 噪声样本：极小 κ_i 视为噪声，直接全掩码（p_i = 1.0）
                noise_threshold = max(1e-6, 0.5 * float(K_max))
                is_noise = kappa < noise_threshold
                dropout_rates = torch.where(is_noise, torch.ones_like(dropout_rates), dropout_rates)

                if it % 100 == 0:
                    try:
                        kappa_min = class_sizes.min().item()
                        kappa_max_val = class_sizes.max().item()
                        kappa_mean = class_sizes.mean().item()
                        p_min = dropout_rates.min().item()
                        p_max = dropout_rates.max().item()
                        p_mean = dropout_rates.mean().item()
                        print_fn(
                            f"kappa stats: min={kappa_min:.2f}, max={kappa_max_val:.2f}, "
                            f"mean={kappa_mean:.2f}, K_max={float(K_max):.2f}; "
                            f"p_i: min={p_min:.2f}, max={p_max:.2f}, mean={p_mean:.2f}"
                        )
                    except Exception as e:
                        print_fn(f"Failed to compute kappa / p_i stats: {e}")

            # 2) 使用自适应噪声瓶颈进行重建训练
            with torch.cuda.amp.autocast(enabled=use_amp):
                en, de, _ = model(img, dropout_rates=dropout_rates)

                if loss_type == 'loose':
                    p_final = 0.9
                    p = min(p_final * it / 1000, p_final)
                    loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
                else:
                    loss = global_cosine(en, de)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 5000 == 0:
                # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
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

    train_end_time = time.perf_counter()
    total_train_time_s = train_end_time - train_start_time
    print_fn('Training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(
        total_train_time_s, total_train_time_s / 60, total_train_time_s / total_iters))

    # Append average summary and training efficiency to log file (same format as mvtec_log.txt)
    if run_args is not None:
        log_path = os.path.join(run_args.save_dir, run_args.save_name, 'log.txt')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                if final_means is not None:
                    f.write(format_mean_summary(*final_means))
                f.write(format_training_efficiency(total_train_time_s, total_iters, batch_size))
        except Exception as e:
            print('Failed to append summary to {}: {}'.format(log_path, e))

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    # 论文原配置使用 CUDA_LAUNCH_BLOCKING=1；需要加速时可设为 0
    os.environ['CUDA_LAUNCH_BLOCKING'] = os.environ.get('CUDA_LAUNCH_BLOCKING', '1')
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1')
    # 消融：替换预训练 backbone。可选: dinov2reg_vit_{small,base,large}_14, dinov2_vit_base_14,
    #       dino_vit_base_16, ibot_vit_base_16, mae_vit_base_16, beitv2_vit_base_16, beit_vit_base_16,
    #       digpt_vit_base_16, deit_vit_base_16
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14',
                        help='Encoder: ViT (e.g. dinov2reg_vit_base_14, mae_vit_base_16) or ResNet (e.g. resnet50, wide_resnet50_2)')
    # 消融：噪声瓶颈 dropout（默认 0.2，Real-IAD 用 0.4；0 表示无 dropout）
    parser.add_argument('--bn_dropout', type=float, default=0.2, help='Bottleneck MLP dropout rate (ablation: 0, 0.2, 0.4)')
    # 消融：解码器注意力（linear=非聚焦线性注意力, softmax=标准 Softmax 注意力）
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=('linear', 'softmax'),
                        help='Decoder attention: linear (default) or softmax for ablation')
    # 消融：分组重建的组数（默认 2；1=不分组，8=逐层）
    parser.add_argument('--num_fuse_groups', type=int, default=2, choices=(1, 2, 4, 8),
                        help='Number of encoder/decoder fuse groups for grouped reconstruction (ablation: 1,2,4,8)')
    # 消融：损失类型（loose=宽松损失只优化难样本, full=全量余弦损失）
    parser.add_argument('--loss_type', type=str, default='loose', choices=('loose', 'full'),
                        help='Reconstruction loss: loose (default) or full for ablation')
    # 多 GPU：逗号分隔的 GPU id；论文原配置为单卡 cuda:1
    parser.add_argument('--gpus', type=str, default='1',
                        help='Comma-separated GPU ids (default: 1, same as paper)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default 16 as in paper)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader num_workers (default 4 as in paper)')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable cudnn.benchmark for speed (default: off, paper setting)')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision FP16 (default: off, paper setting)')
    args = parser.parse_args()
    #
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        gpu_ids = []
    else:
        device = torch.device('cuda', gpu_ids[0] if gpu_ids else 0)
    #
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    print_fn('device={} gpus={}'.format(device, gpu_ids if gpu_ids else 'cpu only'))

    train(item_list, args, device=device, gpu_ids=gpu_ids)


# # 基线（默认 Dinomaly）
# python dinomaly_mvtec_uni.py --save_name exp_baseline

# # 1) 换预训练模型：MAE backbone
# python dinomaly_mvtec_uni.py --encoder_name mae_vit_base_16 --save_name exp_encoder_mae

# # 2) 消融噪声瓶颈：无 dropout
# python dinomaly_mvtec_uni.py --bn_dropout 0 --save_name exp_bn_drop0

# # 3) 消融注意力：解码器用 Softmax 注意力
# python dinomaly_mvtec_uni.py --decoder_attn softmax --save_name exp_attn_softmax

# # 4) 消融分组重建：1 组（不分组）
# python dinomaly_mvtec_uni.py --num_fuse_groups 1 --save_name exp_fuse1

# # 5) 消融宽松损失：全量余弦损失
# python dinomaly_mvtec_uni.py --loss_type full --save_name exp_loss_full

# # 组合：无 dropout + 全量损失
# python dinomaly_mvtec_uni.py --bn_dropout 0 --loss_type full --save_name exp_bn0_loss_full

# tmux new -s lym
# tmux attach -t lym