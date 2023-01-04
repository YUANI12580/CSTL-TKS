conf = {
    "WORK_PATH": "/home/Projects/CSTL-TKS/work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "/home/Projects/CASIA-B/Dataset-64-44",  # your_dataset_path
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 8),
        'restore_iter': 0,
        'total_iter': 150000,
        'margin': 0.2,
        'num_workers': 15,
        'frame_num': 30,
        'model_name': 'CSTL-TKS_Gait-35-Dilation-GMP-Excited',
    },
}