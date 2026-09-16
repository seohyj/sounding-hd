import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from networks.backbones import resnet


class VisualFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device

        self.model = resnet.generate_model(
                model_depth=34,
                n_classes=400,
                n_input_channels=3,
                shortcut_type='B',
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                widen_factor=1.0
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.fc = nn.Identity()
        self.model.eval().to(device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

    def extract(self, frames):
        processed = torch.stack([self.transform(f.permute(1, 2, 0).byte().cpu().numpy()) for f in frames])
        processed = processed.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(processed)
        return feat.squeeze(0)