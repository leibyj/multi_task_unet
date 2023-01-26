from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint

# Model adapted from https://github.com/MIC-DKFZ/nnUNet

class ConvDropoutNormNonlin(nn.Module):
    """
    Basic building block for convolution steps
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # return self.lrelu(self.instnorm(x))
        return self.lrelu(x)


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs=2,
                 conv_kwargs=None, first_stride=None):
        '''
        stacks ConvDropoutNormNonlin layers. 
        first_stride will only be applied to first layer in the stack. The other parameters affect all layers
        '''
        super(StackedConvLayers, self).__init__()
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv_kwargs = conv_kwargs
        # self.conv_op = conv_op

        # if given first stride, change that for first conv
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        self.blocks = nn.Sequential(
            *([ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, conv_kwargs=self.conv_kwargs_first_conv)] +
              [ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, conv_kwargs=self.conv_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


class UNet(nn.Module):
    def __init__(self, in_channels, base_num_feats, num_classes, num_pool, 
                 stride_kernel_sizes, conv_kernel_sizes):
        super(UNet, self).__init__()

        # Conv params
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        transpconv = nn.ConvTranspose3d

        ### Encoder 
        self.Encoder = []

        input_features = in_channels
        output_features = base_num_feats
        for d in range(num_pool):
            # determine the first stride
            if d != 0:
                first_stride = stride_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.Encoder.append(StackedConvLayers(input_features, output_features, conv_kwargs=self.conv_kwargs, first_stride=first_stride))

            input_features = output_features
            output_features = int(np.round(output_features * 2))

            # limit how many channels we have towards the end of the encoder (320 was max from nnUNet run)
            output_features = min(output_features, 320)

        # Bottleneck
        # set parameters... 
        first_stride = stride_kernel_sizes[-1]
        final_num_features = output_features
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.Encoder.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_convs=1, conv_kwargs=self.conv_kwargs,
                              first_stride=first_stride),
            StackedConvLayers(output_features, final_num_features, num_convs=1, conv_kwargs=self.conv_kwargs)))        

        # register
        self.Encoder= nn.ModuleList(self.Encoder)

        ### Decoder
        self.Decoder = []
        self.tu = []
        self.seg_outputs = []

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.Encoder[
                -(2 + u)].output_channels  # self.Encoder[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match that of skip
            # the following convs work on that number of features
            final_num_features = nfeatures_from_skip
            
            self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, stride_kernel_sizes[-(u + 1)],
                                          stride_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.Decoder.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_convs=1,
                                  conv_kwargs=self.conv_kwargs),
                StackedConvLayers(nfeatures_from_skip, final_num_features, num_convs=1, conv_kwargs=self.conv_kwargs)
                ))

        # 1x1 convs for the intermediate (and final) decoder outputs -> auxiliary (and final) segmentation tasks
        for ds in range(len(self.Decoder)):
            self.seg_outputs.append(nn.Conv3d(self.Decoder[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, bias=False))

        self.Decoder = nn.ModuleList(self.Decoder)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        self.final_nonlin = lambda x: F.softmax(x, 1)

    def forward(self, x):
        skips = []
        seg_outputs = []
        dec_outputs =[]
        for d in range(len(self.Encoder) - 1):
            x = self.Encoder[d](x)
            skips.append(x)

        db = self.Encoder[-1](x)

        for u in range(len(self.tu)):
            if u == 0:
                x = self.tu[u](db)
            else:
                x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.Decoder[u](x)
            # returns the decoder outputs before sigmoid activation
            dec_outputs.append(x)
            # outputs softmax
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        return seg_outputs[-1], skips, db, dec_outputs


class Encoder(nn.Module):
    def __init__(self, in_channels, base_num_feats, num_classes, num_pool, 
                 stride_kernel_sizes, conv_kernel_sizes):
        super(Encoder, self).__init__()
         # Conv params
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        transpconv = nn.ConvTranspose3d

        ### Encoder 
        self.Encoder = []

        input_features = in_channels
        output_features = base_num_feats
        for d in range(num_pool):
            # determine the first stride
            if d != 0:
                first_stride = stride_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.Encoder.append(StackedConvLayers(input_features, output_features, conv_kwargs=self.conv_kwargs, first_stride=first_stride))

            input_features = output_features
            output_features = int(np.round(output_features * 2))

            # limit how many channels we have towards the end of the encoder (320 was max from nnUNet run)
            output_features = min(output_features, 320)

        # Bottleneck
        # set parameters... 
        first_stride = stride_kernel_sizes[-1]
        final_num_features = output_features
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.Encoder.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_convs=1, conv_kwargs=self.conv_kwargs,
                              first_stride=first_stride),
            StackedConvLayers(output_features, final_num_features, num_convs=1, conv_kwargs=self.conv_kwargs)))        

        # register
        self.Encoder= nn.ModuleList(self.Encoder)
    def forward(self, x, devs=[]):
        """
        devs: list of torch devices to split the model...
        """
        skips = []
        # for d in range(len(self.Encoder) - 1):
        #     x = self.Encoder[d](x)
        #     skips.append(x)
        # db = self.Encoder[-1](x)
        # skips = []
        x = x.to(devs[0])
        x = torch.utils.checkpoint.checkpoint(self.Encoder[0].to(devs[0]), x, use_reentrant=False)
        skips.append(x.to(devs[1]))
        x = torch.utils.checkpoint.checkpoint(self.Encoder[1].to(devs[2]), x.to(devs[2]), use_reentrant=False)
        x = x.to(devs[1])
        skips.append(x)
        # x = torch.utils.checkpoint.checkpoint(self.Encoder[1], x)
        
        for d in range(2, len(self.Encoder) - 1):
            x = torch.utils.checkpoint.checkpoint(self.Encoder[d].to(devs[1]), x, use_reentrant=False)
            skips.append(x)
        db = torch.utils.checkpoint.checkpoint(self.Encoder[-1].to(devs[1]), x, use_reentrant=False)
        
        return skips, db
        
