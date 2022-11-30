# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import os

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
from monai.transforms import Resize

__all__ = ["Mirror_UNet", "Mirror_Unet"]


@export("monai.networks.nets")
@alias("Mirror_Unet")
class Mirror_UNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        task: str = 'segmentation',
        gpu = 0,
        depth = 1,
        level = 3,
        sliding_window = False,
        separate_outputs = False,
        learnable_th_arg = False,
        mirror_th = 0.3
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.task = task
        self.gpu = gpu
        self.depth = depth
        self.level = level
        self.task = task
        self.sliding_window = sliding_window
        self.separate_outputs = separate_outputs
        self.mirror_th = mirror_th
        self.learnable_th_arg = learnable_th_arg
        self.learnable_th = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        print('Initial Threshold:', self.learnable_th)


        self.down_1 = nn.ModuleList()
        self.up_1 = nn.ModuleList()

        self.common_downs = nn.ModuleList()
        self.common_ups = nn.ModuleList()

        self.down_2 = nn.ModuleList()
        self.up_2 = nn.ModuleList()

        self.dec = nn.ModuleList()


        device = torch.device(f"cuda:{gpu}")
        self.device = device
        channel_list = [self.in_channels] + list(self.channels)
        out_c_list = [self.out_channels] + list(self.channels)


        if self.depth == 1:
            if self.level < 3:
                offset = 2 - self.level
                self.common_down_indices = [len(self.channels) - 2 - offset]
                self.common_up_indices = [-1]
            else:
                self.common_down_indices = [-1] # only bottom layer or upsampling layer
                offset = self.level - 4
                self.common_up_indices = [len(self.channels) - 2 - offset]



        elif self.depth == 2:
            self.common_down_indices = [len(self.channels) - 2]
            self.common_up_indices = [len(self.channels) - 2]
        elif self.depth == 3:
            self.common_down_indices = [len(self.channels) - 2, len(self.channels) - 3]
            self.common_up_indices = [len(self.channels) - 2, len(self.channels) - 3]



        for i, c in enumerate(channel_list):
            if i > len(self.channels) - 2:
                break
            is_top = (i == 0)
            if i in self.common_down_indices: #== len(self.channels) - 2: # arrived at level = 2
                #self.common_down = self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device)
                self.common_downs.append(self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device))
            else:
                self.down_1.append(self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device))
                self.down_2.append(self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device))

            ###### Upsampling Layers ########

            if i == len(self.channels) - 2:
                up_in = channel_list[i + 1] + channel_list[i + 2]

            else:
                up_in = channel_list[i + 1] * 2


            if self.task in ['transference', 'fission', 'fission_classification'] and out_c_list[i] == 2:
                if i in self.common_up_indices:
                    self.common_ups.append(self._get_up_layer(up_in, 1, 2, is_top).to(device))

                else:
                    self.up_1.append(self._get_up_layer(up_in, 1, 2, is_top).to(device))


            else:
                if i in self.common_up_indices:
                    self.common_ups.append(self._get_up_layer(up_in, out_c_list[i], 2, is_top).to(device))

                else:
                    self.up_1.append(self._get_up_layer(up_in, out_c_list[i], 2, is_top).to(device))


            if i not in self.common_up_indices: # and not (out_c_list[i] == 2 and self.args.task == 'fission'):
                self.up_2.append(self._get_up_layer(up_in, out_c_list[i], 2, is_top).to(device))

        if self.depth == 1 and self.level != 3: # only case where bottom layer is not shared
            self.bottom_layer_1 = self._get_bottom_layer(self.channels[-2], self.channels[-1])
            self.bottom_layer_2 = self._get_bottom_layer(self.channels[-2], self.channels[-1])
        else:
            self.bottom_layer = self._get_bottom_layer(self.channels[-2], self.channels[-1])
            self.bottom_layer_1, self.bottom_layer_2 = None, None

        class_bottleneck_size = self.channels[-1]

        if self.task in ['fission', 'fission_classification']:
            dec_channels = out_c_list[:-2]
            for i, c in enumerate(dec_channels[::-1]):
                c_out = 1 if c == 2 else c


                if i == 0:
                    self.dec.append(self._get_up_layer(2 * self.channels[-1], c_out, 2, False).to(device))
                else:
                    self.dec.append(self._get_up_layer(dec_channels[::-1][i - 1], c_out, 2, False).to(device))

        if self.task == 'fission_classification':
                self.flatten_channels = Convolution(
                            self.dimensions,
                            class_bottleneck_size,
                            1,
                            strides=1,
                            kernel_size=1,
                            act=self.act,
                            norm=self.norm,
                            dropout=self.dropout,
                            bias=self.bias,
                            adn_ordering=self.adn_ordering,
                        ).to(device)
                in_dense = 216 if self.sliding_window else 5000

                self.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_dense, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256, 1),
                                        nn.Sigmoid()
                                        ).to(device)





    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume one channel for each modality

        common_bottom = not (self.depth == 1 and self.level != 3)


        x_1 = x[:,0].unsqueeze(dim=1) # CT

        down_x_1 = []
        for i, d in enumerate(self.down_1):
            if i == self.common_down_indices[0]:
                for c_d in self.common_downs:
                    x_1 = c_d(x_1)
                    down_x_1.append(x_1)

            x_1 = d(x_1)
            down_x_1.append(x_1)

        if (self.level == 2 and self.depth == 1) or self.depth >= 2:
            for d in self.common_downs:
                x_1 = d(x_1)
                down_x_1.append(x_1)


        if not common_bottom:
            bottom_x_1 = self.bottom_layer_1(x_1)
        else:

            bottom_x_1 = self.bottom_layer(x_1)

        x_1 = torch.cat([bottom_x_1, down_x_1[-1]], dim=1)



        up_x_1 = []
        passed_common = False
        for i, up in enumerate(self.up_1[::-1]):
            if i == len(self.channels) -2 - self.common_up_indices[0]:

                for j, c_up in enumerate(self.common_ups[::-1]):
                    x_1 = torch.cat([c_up(x_1), down_x_1[-j - i - 2]], dim=1)
                passed_common = True

            if passed_common:
                if len(down_x_1) < abs(-i -len(self.common_ups) - 2):
                    break
                x_1 = torch.cat([up(x_1), down_x_1[-i - len(self.common_ups) - 2]], dim=1)
            else:
                if len(down_x_1) < abs(-i - 2):
                    break

                x_1 = torch.cat([up(x_1), down_x_1[-i - 2]], dim=1)
        x_1 = self.up_1[0](x_1)


        x_2 = x[:,1].unsqueeze(dim=1) # PET

        down_x_2 = []
        for i, d in enumerate(self.down_2):
            if i == self.common_down_indices[0]:
                for c_d in self.common_downs:
                    x_2 = c_d(x_2)
                    down_x_2.append(x_2)

            x_2 = d(x_2)
            down_x_2.append(x_2)

        if (self.level == 2 and self.depth == 1) or self.depth >= 2:
            for d in self.common_downs:
                x_2 = d(x_2)
                down_x_2.append(x_2)


        if not common_bottom:
            bottom_x_2 = self.bottom_layer_2(x_2)
        else:

            bottom_x_2 = self.bottom_layer(x_2)



        x_2 = torch.cat([bottom_x_2, down_x_2[-1]], dim=1)



        passed_common = False
        for i, up in enumerate(self.up_2[::-1]):
            if i == len(self.channels) -2 - self.common_up_indices[0]:
                for j, c_up in enumerate(self.common_ups[::-1]):
                    x_2 = torch.cat([c_up(x_2), down_x_2[-j - i - 2]], dim=1)
                passed_common = True
            if passed_common:
                if len(down_x_2) < abs(-i -len(self.common_ups) - 2):
                    break
                x_2 = torch.cat([up(x_2), down_x_2[-i - len(self.common_ups) - 2]], dim=1)
            else:
                if len(down_x_2) < abs(-i - 2):
                    break

                x_2 = torch.cat([up(x_2), down_x_2[-i - 2]], dim=1)

        x_2 = self.up_2[0](x_2)

        x_3 = None
        if self.task in ['fission', 'fission_classification']:
            x_3 = torch.cat([bottom_x_1, bottom_x_2], dim=1)

            for i, d in enumerate(self.dec):
                x_3 = d(x_3)

        cls = None
        if self.task == 'fission_classification':
            out = self.flatten_channels(bottom_x_2)

            cls = self.fc(out)


        return self.process_output(x_1, x_2, x_3, cls)




    def process_output(self, x_1: torch.Tensor, x_2: torch.Tensor, x_3: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        if self.separate_outputs:
            if self.task not in  ['fission', 'fission_classification']:
                return x_1, x_2
            else:
                return x_1, x_3, x_2 # CT, PET, SEG


        if self.task == 'segmentation':
            if self.learnable_th_arg:
                out = torch.clamp(self.learnable_th, 0, 1) * x_1 + (1 - torch.clamp(self.learnable_th, 0, 1)) * x_2
            else:
                out = self.mirror_th * x_1 + (1 - self.mirror_th) * x_2 # For decision fusion segmentation (exp_1)

            return out
        elif self.task == 'reconstruction' or self.task == 'transference':

            x_12 = torch.cat((x_1, x_2), dim=1)
            return x_12
        elif self.task == 'fission':
            x_123 = torch.cat((x_1, x_3, x_2), dim=1)
            return x_123
        elif self.task == 'fission_classification':
            x_4 = torch.ones(x_1.shape).to(self.device)
            x_4 *= cls[..., None, None, None]
            x_1234 = torch.cat((x_1, x_3, x_2, x_4), dim=1)

            return x_1234
        else:
            raise ValueError(f"Task {self.task} is not supported!")


    def load_pretrained_unequal(self, file):
        # load the weight file and copy the parameters
        if os.path.isfile(file):
            print('Loading pre-trained weight file.')
            if 'net' in torch.load(file).keys():
                weight_dict = torch.load(file)["net"]
            else:
                weight_dict = torch.load(file)
            model_dict = self.state_dict()
            n_params = 0
            n_not_found = 0
            n_mismatch = 0
            for name, param in weight_dict.items():
                if name in model_dict:
                    if param.size() == model_dict[name].size():

                        model_dict[name].copy_(param)
                        #model_dict[name] = param
                    else:
                        print(
                            f' WARNING parameter size not equal. Skipping weight loading for: {name} '
                            f'File: {param.size()} Model: {model_dict[name].size()}')
                        n_mismatch += 1
                else:
                    print(f' WARNING parameter from weight file not found in model. Skipping {name}')
                    n_not_found += 1
                n_params += 1
            print('Loaded pre-trained parameters from file.')
            print('Not found [%]', round(100 * n_not_found / n_params, 2), 'Mismatch [%]', round(100 * n_mismatch / n_params, 2))

        else:
            raise ValueError(f"Weight file {file} does not exist")

    def load_pretrained_transference(self, file_1, file_2):


        for i, file in enumerate([file_1, file_2]):
            # load the weight file and copy the parameters
            if os.path.isfile(file):
                if i == 0:
                    print('Loading pre-trained weight file for CT task.')
                else:
                    print('Loading pre-trained weight file for PET task.')
                if 'net' in torch.load(file).keys():
                    weight_dict = torch.load(file)["net"]
                else:
                    weight_dict = torch.load(file)

                model_dict = self.state_dict()
                n_params = 0
                n_not_found = 0
                n_mismatch = 0
                for param_name, param in weight_dict.items():

                    name = param_name[6:] # ommit model. prefix

                    if (f'path_{i+1}.' + name) in model_dict:


                        if param.size() == model_dict[(f'path_{i+1}.' + name)].size():

                            model_dict[(f'path_{i+1}.' + name)].copy_(param)
                            #model_dict[name] = param
                        else:
                            print(param.size(), model_dict[(f'path_{i+1}.' + name)].size())
                            print(
                                f' WARNING parameter size not equal. Skipping weight loading for: {name} '
                                f'File: {param.size()} Model: {model_dict[name].size()}')
                            n_mismatch += 1
                    else:
                        print(f' WARNING parameter from weight file not found in model. Skipping {name}')
                        n_not_found += 1
                    n_params += 1
                print('Loaded pre-trained parameters from file.')
                print('Not found [%]', round(100 * n_not_found / n_params, 2), 'Mismatch [%]', round(100 * n_mismatch / n_params, 2))

            else:
                raise ValueError(f"Weight file {file} does not exist")
