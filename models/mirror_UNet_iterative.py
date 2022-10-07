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
        args = None,
    ) -> None:

        super().__init__()
        #ztorch.manual_seed(0)

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
        self.args = args
        self.learnable_th = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        print('Initial Threshold:', self.learnable_th)


        self.down_1 = nn.ModuleList()
        self.up_1 = nn.ModuleList()

        self.down_2 = nn.ModuleList()
        self.up_2 = nn.ModuleList()

        device = torch.device(f"cuda:{args.gpu}")
        channel_list = [self.in_channels] + list(self.channels)
        out_c_list = [self.out_channels] + list(self.channels)
        for i, c in enumerate(channel_list):
            if i > len(self.channels) - 2:
                break
            is_top = (i == 0)
            if i == len(self.channels) - 2:
                self.common_down = self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device)
            else:
                self.down_1.append(self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device))
                self.down_2.append(self._get_down_layer(c, channel_list[i + 1], 2, is_top).to(device))

            if i == len(self.channels) - 2:
                up_in = channel_list[i + 1] + channel_list[i + 2]
            else:
                up_in = channel_list[i + 1] * 2
            if args.task == 'transference' and out_c_list[i] == 2:
                self.up_1.append(self._get_up_layer(up_in, 1, 2, is_top).to(device))
            else:
                self.up_1.append(self._get_up_layer(up_in, out_c_list[i], 2, is_top).to(device))
            self.up_2.append(self._get_up_layer(up_in, out_c_list[i], 2, is_top).to(device))


        self.bottom_layer_1 = self._get_bottom_layer(self.channels[-2], self.channels[-1])
        self.bottom_layer_2 = self._get_bottom_layer(self.channels[-2], self.channels[-1])





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

        x_1 = x[:,0].unsqueeze(dim=1) # CT

        down_x_1 = []
        for d in self.down_1:
            x_1 = d(x_1)
            down_x_1.append(x_1)
        x_1 = self.common_down(x_1)
        down_x_1.append(x_1)

        x_1 = torch.cat([self.bottom_layer_1(x_1), down_x_1[-1]], dim=1)
        for i, up in enumerate(self.up_1[::-1]):
            if len(down_x_1) < abs(-i - 2):
                break
            x_1 = torch.cat([up(x_1), down_x_1[-i - 2]], dim=1)
        x_1 = self.up_1[0](x_1)


        x_2 = x[:,1].unsqueeze(dim=1) # PET

        down_x_2 = []
        for d in self.down_2:
            x_2 = d(x_2)
            down_x_2.append(x_2)
        x_2 = self.common_down(x_2)
        down_x_2.append(x_2)

        x_2 = torch.cat([self.bottom_layer_2(x_2), down_x_2[-1]], dim=1)
        for i, up in enumerate(self.up_2[::-1]):
            if len(down_x_2) < abs(-i - 2):
                break
            x_2 = torch.cat([up(x_2), down_x_2[-i - 2]], dim=1)
        x_2 = self.up_2[0](x_2)


        if self.args.separate_outputs:
            return x_1, x_2

        if self.task == 'segmentation':
            if self.args.learnable_th:
                out = torch.clamp(self.learnable_th, 0, 1) * x_1 + (1 - torch.clamp(self.learnable_th, 0, 1)) * x_2
            else:
                out = self.args.mirror_th * x_1 + (1 - self.args.mirror_th) * x_2 # For decision fusion segmentation (exp_1)

            return out
        elif self.task == 'reconstruction' or self.task == 'transference':
            x_12 = torch.cat((x_1, x_2), dim=1)
            return x_12
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
