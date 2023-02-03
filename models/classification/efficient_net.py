import torch
import os

import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(
        self,
        net_v = 'b0',
        in_channels=1,
        args=None,
    ) -> None:

        super().__init__()
        self.net_v = net_v
        self.in_channels = in_channels
        self.args = args
        torch.hub.set_dir('/hkfs/work/workspace/scratch/zk6393-test_zrrr/autoPET_challenge/cache')
        self.net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', f'nvidia_efficientnet_{self.net_v}', pretrained=False)


        if self.in_channels != 3:
            if 'b0' in self.net_v:
                self.net.stem.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            elif 'b4' in self.net_v:
                self.net.stem.conv = nn.Conv2d(in_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            else:
                print(f"[ERROR] unknown EfficientNet version: {self.net_v}") 
        self.net.classifier.fc = nn.Linear(self.net.classifier.fc.in_features, 1)

        self.linear_act = nn.Sigmoid()

        self.model = nn.Sequential(self.net,
                                   self.linear_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model(x)
        return x

    def load_pretrained_unequal(self, file):
        # load the weight file and copy the parameters
        if os.path.isfile(file):
            print('Loading pre-trained weight file.')
            if "net" in torch.load(file).keys():
                weight_dict = torch.load(file)["net"]
            else:
                weight_dict = torch.load(file)
            model_dict = self.state_dict()

            for name, param in weight_dict.items():
                if name in model_dict:
                    if param.size() == model_dict[name].size():

                        model_dict[name].copy_(param)
                        #model_dict[name] = param
                    else:
                        print(
                            f' WARNING parameter size not equal. Skipping weight loading for: {name} '
                            f'File: {param.size()} Model: {model_dict[name].size()}')
                else:
                    print(f' WARNING parameter from weight file not found in model. Skipping {name}')
            print('Loaded pre-trained parameters from file.')

        else:
            raise ValueError(f"Weight file {file} does not exist")

