import json
import os
import time


class Config:
    def __init__(self, args):
        self.dataset = args.dataset
        self.mode = args.mode
        self.tag = args.tag
        self.device = args.device
        self.top_ratio = args.top_ratio
        self.epochs = args.epochs
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.grad_clip = args.grad_clip
        self.early_stopping = getattr(args, 'early_stopping', False)
        self.patience = getattr(args, 'patience', 15)
        self.use_hdf5 = getattr(args, 'use_hdf5', False)
        self.save_all_metric_ckpts = getattr(args, 'save_all_metric_ckpts', False)
        self.fold_id = getattr(args, 'fold_id', None)
        self.run_id = args.run_id
        self.seed = args.seed

        # Override with the DATA_ROOT / OUTPUT_ROOT environment variables.
        self.data_root = os.getenv('DATA_ROOT', './data')
        self.output_root = os.getenv('OUTPUT_ROOT', './outputs')

        # torchaudio.load() decodes the audio track straight out of the video file, which needs
        # the ffmpeg backend (see the README's conda-forge ffmpeg=6 note).
        if self.dataset == "tvsum":
            self.audio_dir = os.path.join(self.data_root, 'tvsum/features/audio_emb')
            self.visual_dir = os.path.join(self.data_root, 'tvsum/features/visual_emb')
            self.label_dir = os.path.join(self.data_root, 'tvsum/tvsum_gtscore')
            self.raw_audio_dir = os.path.join(self.data_root, 'tvsum/video')
            self.mel_spec_dir = os.path.join(self.data_root, 'tvsum/features/melspec')
            self.split_file = None
            self.use_split_json = False
            visual_dim = 512
        elif self.dataset == "mrhisum":
            self.audio_dir = os.path.join(self.data_root, 'mrhisum/audio_emb')
            self.visual_dir = os.path.join(self.data_root, 'mrhisum/visual_emb')
            self.label_dir = os.path.join(self.data_root, 'mrhisum/gtscore')
            self.raw_video_dir = os.path.join(self.data_root, 'mrhisum/video')
            self.raw_audio_dir = os.path.join(self.data_root, 'mrhisum/video')
            self.mel_spec_dir = os.path.join(self.data_root, 'mrhisum/audio/melspec')
            self.split_file = os.path.join(os.getcwd(), 'dataset/mrhisum_split.json')
            self.use_split_json = True
            visual_dim = 1024
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.best_metric_name = "F1"

        self.model_config = {
            'audio_dim': 2048,
            'visual_dim': visual_dim,
            'hidden_dim': 512,
            'dropout': 0.1,
            'n_heads': 4,
        }

        self.audio_config = {
            'sample_rate': 16000,
            'n_fft': 2048,
            'win_length': 2048,
            'hop_length': 256,
            'n_mels': 128,
            'fmin': 0,
            'fmax': 8000,
            'target_fps': 1,
        }

        self.visual_config = {'resize_dim': 224}

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if args.repeat:
            base_exp_name = f"{self.dataset}_{self.tag}" if self.tag else f"{self.dataset}_repeat_{timestamp}"
            run_dir = os.path.join(self.output_root, base_exp_name, f"run_{args.run_id}_seed_{args.seed}")
            self.save_dir = os.path.join(run_dir, f"fold_{self.fold_id}") if self.fold_id is not None else run_dir
        else:
            if self.fold_id is not None:
                self.exp_name = f"{self.dataset}_{self.tag}_fold_{self.fold_id}_{timestamp}"
            else:
                self.exp_name = f"{self.dataset}_{self.tag}_{timestamp}" if self.tag else f"{self.dataset}_{timestamp}"
            self.save_dir = os.path.join(self.output_root, self.exp_name)

        if getattr(args, 'skip_dir_creation', False):
            self.tensorboard_logdir = None
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            self.tensorboard_logdir = os.path.join(self.save_dir, "tensorboard")
            os.makedirs(self.tensorboard_logdir, exist_ok=True)

        # Written on every improvement of best_metric_name and reloaded to report test scores.
        self.ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        self.config_path = os.path.join(self.save_dir, "config.txt")
        self.log_dir = self.save_dir
        self.log_path = os.path.join(self.save_dir, "result_log.txt")

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=str)
