import os
import argparse
import subprocess
from time import sleep
from itertools import product
import torch
import logging
import re


TIME_TO_IDLE = 60 # seconds
CHECK_INTERVAL = 15
GPU_UTIL_THRESH = 1
GPU_MEMORY_USAGE_THRESH = 20
NUM_WORKERS = 8

def setup_logging():
    logging.basicConfig(filename='grid_search_python/log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def log_and_print(text):
    logging.info(text)
    print(text)


def parse_args():
    parser = argparse.ArgumentParser(description='Train multiple models with varying parameters.')
    parser.add_argument('--blr', type=float, default=1.0e-4, help='Base learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Default min lr of pretraining.')
    parser.add_argument('--available_devices', type=int, nargs='+', default=[0,1,2,3,4], help='GPU device ID.')
    parser.add_argument('--t_0_values', type=float, nargs='+', default=[0.8, 1.0], help='List of t0 parameters for freezing.')
    parser.add_argument('--singlestage', action='store_true', help='Use single stage training.')
    parser.add_argument('--not_scale_lr', action='store_true', help='Disable learning rate scaling.')
    parser.add_argument('--dataset_type', type=str, default='5', help='Imagenet dataset type (5, 10, 25, 100).')
    parser.add_argument('--multipliers', type=float, nargs='+', default=[0.0, 0.8], help='List of initial LR stage multiplier powers.')
    parser.add_argument('--epoch_percentages', type=float, nargs='+', default=[0.2, 1.0, 5.0], help='Percentages of epochs for training stages.')
    return parser.parse_args()

def get_idle_cuda_devices(device_count):
    time_passed = 0
    gpu_util_idle_time_list = list(0 for _ in range(device_count))
    while time_passed < TIME_TO_IDLE:
        # gpu_util_str = subprocess.getoutput(f"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").strip()
        # memory_usage_str = subprocess.getoutput(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
        # gpu_util_list = gpu_util_str.split("\n")
        # memory_usage_list = memory_usage_str.split("\n")
        gpu_util_and_memory_str = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").strip()
        gpu_util_list = []
        memory_usage_list = []
        for line in gpu_util_and_memory_str.split("\n"):
            gpu_util, memory_usage = line.split(", ")
            gpu_util_list.append(gpu_util)
            memory_usage_list.append(memory_usage)
        for gpu_id, gpu_util in enumerate(gpu_util_list):
            memory_usage = memory_usage_list[gpu_id]
            if int(gpu_util) <= GPU_UTIL_THRESH and int(memory_usage) <= GPU_MEMORY_USAGE_THRESH:
                gpu_util_idle_time_list[gpu_id] += CHECK_INTERVAL
        sleep(CHECK_INTERVAL)
        time_passed += CHECK_INTERVAL
    gpu_idle_bool_list = [idle_time == TIME_TO_IDLE for idle_time in gpu_util_idle_time_list]
    idle_devices = [index for index in range(device_count) if gpu_idle_bool_list[index]]
    return idle_devices


def find_latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    checkpoint_files = [f for f in os.listdir(directory) if re.match(r'checkpoint-\d+\.pth', f)]
    if not checkpoint_files:
        return None
    checkpoint_epochs = [int(re.search(r'checkpoint-(\d+)\.pth', f).group(1)) for f in checkpoint_files]
    latest_checkpoint_epoch = max(checkpoint_epochs)
    return f"{directory}/checkpoint-{latest_checkpoint_epoch}.pth"

def test_find_latest_checkpoint(combinations, args):
    epoch_mapping = {'5': 2000, '10': 1000, '25': 400, '50': 200, '100': 100}
    epochs_base = epoch_mapping[args.dataset_type]
    available_finetune_folders = 0
    for combination in combinations:
        percentage_finetune, percentage_pretrain, t_0_value, multiplier = combination
        pre_train_epoch = int(epochs_base * percentage_pretrain)
        fine_tune_epoch = int(epochs_base * percentage_finetune)
        pretrain_checkpoint_epoch = pre_train_epoch - 1

        stage_setup = f"{'single_stage_exps' if args.singlestage else 'multi_stage_exps'}_IN{args.dataset_type}"
        lr_setup = f"t{t_0_value}_lrgrid_notscalelr" if args.not_scale_lr else f"t{t_0_value}_lrgrid_warmupfixed"
        group_name = f"{stage_setup}/{lr_setup}/"

        output_finetune_dir = f"grid_search_python/finetune/{group_name}freezeout_cubic_t{t_0_value}_blr{args.blr}_initiallrstagemultiplierpow{multiplier}_ftepoch{fine_tune_epoch}_ptepoch{pre_train_epoch}_checkpoint-{pretrain_checkpoint_epoch}"

        if os.path.exists(output_finetune_dir):
            latest_checkpoint = find_latest_checkpoint(output_finetune_dir)
            log_and_print(f"Combination: {combination}")
            log_and_print(f"Latest checkpoint for {output_finetune_dir}: {latest_checkpoint}")
            if latest_checkpoint:
                available_finetune_folders +=1
    log_and_print(f"available_finetune_folders are: {available_finetune_folders}")

def pretrain_and_finetune(epochs_base, percentage_finetune, percentage_pretrain, t_0_value, multiplier, master_port, device, args):
    """
        Prepares and executes commands for pretraining and finetuning models based on provided configurations.

        Parameters:
        - epochs_base (int): Base number of epochs determined by dataset type.
        - percentage_pretrain (float): Percentage multiplier for pretraining epochs.
        - percentage_finetune (float): Percentage multiplier for finetuning epochs.
        - t_0_value (float): t0 for frezing.
        - multiplier (float): Learning rate multiplier.
        - master_port (int): Master port number for distributed training.
        - args (Namespace): Arguments namespace containing runtime configurations.
    """
    pre_train_epoch = int(epochs_base * percentage_pretrain)
    fine_tune_epoch = int(epochs_base * percentage_finetune)
    pretrain_checkpoint_epoch = pre_train_epoch - 1 # Finetuned model.
    finetune_checkpoint_epoch = fine_tune_epoch - 1 # Finetuned model.

    pre_train_warmup = pre_train_epoch // 10
    fine_tune_warmup = fine_tune_epoch // 5

    stage_setup = f"{'single_stage_exps' if args.singlestage else 'multi_stage_exps'}_IN{args.dataset_type}"
    lr_setup = f"t{t_0_value}_lrgrid_notscalelr" if args.not_scale_lr else f"t{t_0_value}_lrgrid_warmupfixed"
    group_name = f"{stage_setup}/{lr_setup}/"

    output_pretrain_dir = f"grid_search_python/pretrain/{group_name}freezeout_cubic_t{t_0_value}_blr{args.blr}_initiallrstagemultiplierpow{multiplier}_epoch{pre_train_epoch}"
    output_finetune_dir = f"grid_search_python/finetune/{group_name}freezeout_cubic_t{t_0_value}_blr{args.blr}_initiallrstagemultiplierpow{multiplier}_ftepoch{fine_tune_epoch}_ptepoch{pre_train_epoch}_checkpoint-{pretrain_checkpoint_epoch}"

    data_path = f"/raid/utku/datasets/imagenet{args.dataset_type}"
    scale_lr_flags = "--not_scale_lr --non_layerwise_lr" if args.not_scale_lr else ""
    stage_count_flag = "--stage_count 1" if args.singlestage else "--stage_count 4"

    pretrain_final_checkpoint_path = f"{output_pretrain_dir}/checkpoint-{pretrain_checkpoint_epoch}.pth"
    pretrain_command = ""
    if not os.path.exists(pretrain_final_checkpoint_path):
        pretrain_command = (
            f"CUDA_VISIBLE_DEVICES={device} bash record.sh python3 -m torch.distributed.launch --nproc_per_node=1 --master_port={master_port} "
            f"run_pretrain.py --epochs {pre_train_epoch} --initial_lr_stage_multiplier_pow {multiplier} "
            f"--batch_size 1024 --warmup_epochs {pre_train_warmup} --blr {args.blr} --world_size 1 --accum_iter 2 "
            f"--model MIM_vit_base_patch16 --data_path {data_path}/train --output_dir {output_pretrain_dir} "
            f"--log_dir {output_pretrain_dir} --min_lr {args.min_lr} {scale_lr_flags} {stage_count_flag} "
            f"--how_scale cubic --t_0 {t_0_value} --num_workers {NUM_WORKERS}"
        )
        # run in non-blocking manner.
    else:
        logging.info(f"Skipping pretraining as checkpoint already exists: {pretrain_final_checkpoint_path}")

    finetune_final_checkpoint_path = f"{output_finetune_dir}/checkpoint-{finetune_checkpoint_epoch}.pth"
    finetune_resume_checkpoint_path = find_latest_checkpoint(output_finetune_dir) # None if no checkpoint exists.
    finetune_command = ""
    if not os.path.exists(finetune_final_checkpoint_path):
        if finetune_resume_checkpoint_path:
            finetune_command = (
                f"CUDA_VISIBLE_DEVICES={device} bash record.sh python3 -m torch.distributed.launch --nproc_per_node=1 --master_port={master_port} "
                f"run_finetune.py --world_size 1 --accum_iter 2 --batch_size 512 --model vit_base_patch16 "
                f"--resume {finetune_resume_checkpoint_path} --epochs {fine_tune_epoch} --warmup_epochs {fine_tune_warmup} "
                f"--lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 "
                f"--mixup 0.8 --cutmix 1.0 --dist_eval --data_path {data_path} --output_dir {output_finetune_dir} "
                f"--log_dir {output_finetune_dir} --num_workers {NUM_WORKERS}"
            )
        elif not os.path.exists(finetune_final_checkpoint_path):
            finetune_command = (
                f"CUDA_VISIBLE_DEVICES={device} bash record.sh python3 -m torch.distributed.launch --nproc_per_node=1 --master_port={master_port} "
                f"run_finetune.py --world_size 1 --accum_iter 2 --batch_size 512 --model vit_base_patch16 "
                f"--finetune {pretrain_final_checkpoint_path} --epochs {fine_tune_epoch} --warmup_epochs {fine_tune_warmup} "
                f"--lr 4e-3 --min_lr 1e-6 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 "
                f"--mixup 0.8 --cutmix 1.0 --dist_eval --data_path {data_path} --output_dir {output_finetune_dir} "
                f"--log_dir {output_finetune_dir} --num_workers {NUM_WORKERS}"
            )
    else:
        logging.info(f"Skipping finetuning as checkpoint already exists: {finetune_final_checkpoint_path}")
    combined_pretrain_and_finetune_command = (pretrain_command + ";" if pretrain_command else "") + finetune_command
    # run in non-blocking manner.
    if combined_pretrain_and_finetune_command != ";":
        logging.info(f"Started train with pretrain epochs: {pre_train_epoch}, finetune epochs: {fine_tune_epoch}, multiplier: {multiplier}!")
        logging.info(f"Started train with command:\n{combined_pretrain_and_finetune_command}")
        subprocess.Popen(combined_pretrain_and_finetune_command, shell=True)

def train_completed(epochs_base, percentage_finetune, percentage_pretrain, t_0_value, multiplier, args):
    pre_train_epoch = int(epochs_base * percentage_pretrain)
    fine_tune_epoch = int(epochs_base * percentage_finetune)
    pretrain_checkpoint_epoch = pre_train_epoch - 1 # Finetuned model.
    finetune_checkpoint_epoch = fine_tune_epoch - 1 # Finetuned model.

    stage_setup = f"{'single_stage_exps' if args.singlestage else 'multi_stage_exps'}_IN{args.dataset_type}"
    lr_setup = f"t{t_0_value}_lrgrid_notscalelr" if args.not_scale_lr else f"t{t_0_value}_lrgrid_warmupfixed"
    group_name = f"{stage_setup}/{lr_setup}/"

    output_pretrain_dir = f"grid_search_python/pretrain/{group_name}freezeout_cubic_t{t_0_value}_blr{args.blr}_initiallrstagemultiplierpow{multiplier}_epoch{pre_train_epoch}"
    output_finetune_dir = f"grid_search_python/finetune/{group_name}freezeout_cubic_t{t_0_value}_blr{args.blr}_initiallrstagemultiplierpow{multiplier}_ftepoch{fine_tune_epoch}_ptepoch{pre_train_epoch}_checkpoint-{pretrain_checkpoint_epoch}"

    pretrain_final_checkpoint_path = f"{output_pretrain_dir}/checkpoint-{pretrain_checkpoint_epoch}.pth"
    finetune_final_checkpoint_path = f"{output_finetune_dir}/checkpoint-{finetune_checkpoint_epoch}.pth"
    return os.path.exists(pretrain_final_checkpoint_path), os.path.exists(finetune_final_checkpoint_path)


def main():
    args = parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    available_devices = args.available_devices
    
    # Map dataset types to base epochs
    epoch_mapping = {'5': 2000, '10': 1000, '25': 400, '50': 200, '100': 100}
    epochs_base = epoch_mapping[args.dataset_type]

    raw_combinations = list(product(args.epoch_percentages, args.epoch_percentages, args.t_0_values, args.multipliers))
    logging.info(f"Raw combinations: {len(raw_combinations)}")
    for raw_combination in raw_combinations:
        logging.info(f"{raw_combination}")
    
    combinations = []
    for raw_combination in raw_combinations:
        percentage_finetune, percentage_pretrain, t_0_value, multiplier = raw_combination
        if t_0_value == 1.0 and multiplier != 0.0:
            continue
        combinations.append(raw_combination)
    logging.info(f"Filtered combinations: {len(combinations)}")
    for combination in combinations:
        logging.info(f"{combination}")

    logging.info(f"Completed combinations:")
    for combination in combinations:
        percentage_finetune, percentage_pretrain, t_0_value, multiplier = combination
        pretrain_completed, finetune_completed = train_completed(epochs_base, percentage_finetune, percentage_pretrain, t_0_value, multiplier, args)
        logging.info(f"{combination} --- pretrain_completed: {pretrain_completed}, finetune_completed {finetune_completed}")

    # Run the test for find_latest_checkpoint
    test_find_latest_checkpoint(combinations, args)

    device_count = torch.cuda.device_count()
    incremental_index = 0
    logging.info(f"Number of trains left is: {len(combinations)-incremental_index}.")
    while incremental_index < len(combinations):
        # find whicever device is idle here. Do it with multiprocessing for available_devices. Find all available devices without looping over the candidates.
        idle_devices = get_idle_cuda_devices(device_count=device_count)
        idle_available_devices = [idle_device for idle_device in idle_devices if idle_device in available_devices]
        if len(idle_available_devices) != 0:
            logging.info(f"IDLE available devices are: {idle_available_devices}")
        # run in parallel for idle_available_devices
        for idle_available_device in idle_available_devices:
            if incremental_index == len(combinations):
                continue
            master_port = 29508 + idle_available_device
            percentage_finetune, percentage_pretrain, t_0_value, multiplier = combinations[incremental_index]
            logging.info(f"\n")
            logging.info(f"Run train for: {t_0_value}, {multiplier}, {percentage_pretrain}, {percentage_finetune}")
            # This is non-blocking (can run multiple runs simultaneously.)
            pretrain_and_finetune(epochs_base, percentage_finetune, percentage_pretrain, t_0_value, multiplier, master_port, idle_available_device, args)
            incremental_index += 1
        if len(idle_available_devices) != 0:
            logging.info(f"Number of trains left is: {len(combinations)-incremental_index}.")

    logging.info("Pretrain / Finetune completed for all settings!")

if __name__ == "__main__":
    setup_logging()
    main()
