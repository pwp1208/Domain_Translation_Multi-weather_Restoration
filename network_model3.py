import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import matplotlib.pyplot as plt
import numpy as np

class GDFN(nn.Module):
    def __init__(self, channels, outch, kernel, pad, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=kernel, padding=pad,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, outch, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class conv_block_my(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1):

        super(conv_block_my, self).__init__()

        nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels,
            kernel_size = kernel_size, stride=stride, padding = padding, dilation=dilation)

        self.conv_block_my = nn.Sequential(
            nn_Conv2d(in_channels,  out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, inputs):
       
        outputs = self.conv_block_my(inputs)
        return outputs

class msc_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        
        super(msc_block, self).__init__()        

        self.psi1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        self.psi2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding = 1)
        self.psi3 = nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding = 2)
        
        self.conv_cat = conv_block_my(output_channels*3, output_channels, kernel_size = 3, stride = 1)
        
    def forward(self, enc1):
        shorcut = enc1.clone()
        conv_enc1 = self.psi1(enc1)
        conv_enc2 = self.psi2(enc1)
        conv_enc3 = self.psi3(enc1)

        cat_all = torch.cat((conv_enc1,conv_enc2, conv_enc3),dim=1)
        output_msc = self.conv_cat(cat_all)        

        return output_msc


class mrb_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mrb_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 7, dilation=7)

        self.conv_cat = conv_block_my(out_channels*3, out_channels, kernel_size = 1, stride = 1, padding = 0, dilation=1)

    def forward(self, x):
        conv_rec1 = self.conv1(x)
        conv_rec5 = self.conv5(x)
        conv_rec7 = self.conv7(x)

        cat_all = torch.cat((conv_rec1,conv_rec5, conv_rec7),dim=1)
        output_vrf = self.conv_cat(cat_all)

        return output_vrf

class multi_channel_sep_conv(nn.Module):
    def __init__(self, in_channels):
        super(multi_channel_sep_conv, self).__init__()

        self.conv = conv_block_my(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.cat_conv1 = GDFN(channels=(in_channels//4)*2, outch=in_channels//4, kernel=1, pad=0, expansion_factor=2.66)#nn.Conv2d((in_channels//4)*2, (in_channels//4),  kernel_size = 1, stride=1)#depthwise_separable_conv((in_channels//4)*2,in_channels//4,in_channels//4)
        self.cat_conv2 = GDFN(channels=(in_channels//4)*2, outch=in_channels//4, kernel=3, pad=1, expansion_factor=2.66)#nn.Conv2d((in_channels//4)*2, (in_channels//4),  kernel_size = 1, stride=1)
        self.cat_conv3 = GDFN(channels=(in_channels//4)*2, outch=in_channels//4, kernel=5, pad=2, expansion_factor=2.66)#nn.Conv2d((in_channels//4)*2, (in_channels//4),  kernel_size = 1, stride=1)
        self.cat_conv4 = GDFN(channels=(in_channels//4)*2, outch=in_channels//4, kernel=7, pad=3, expansion_factor=2.66)#nn.Conv2d((in_channels//4)*2, (in_channels//4),  kernel_size = 1, stride=1)


    
    def forward(self, x1, x2):
        b, ch, m,n = x1.shape
        slot_size = ch//4
        x1_slot1 = x1[:,0:slot_size,:,:]
        x1_slot2 = x1[:,slot_size:slot_size*2,:,:]
        x1_slot3 = x1[:,slot_size*2:slot_size*3,:,:]
        x1_slot4 = x1[:,slot_size*3:slot_size*4,:,:]


        x2_slot1 = x2[:,0:slot_size,:,:]
        x2_slot2 = x2[:,slot_size:slot_size*2,:,:]
        x2_slot3 = x2[:,slot_size*2:slot_size*3,:,:]
        x2_slot4 = x2[:,slot_size*3:slot_size*4,:,:]

        slot1 = torch.cat((x1_slot1,x2_slot1),dim=1)
        slot1 = self.cat_conv1(slot1)

        slot2 = torch.cat((x1_slot2,x2_slot2),dim=1)
        slot2 = self.cat_conv2(slot2)

        slot3 = torch.cat((x1_slot3,x2_slot3),dim=1)
        slot3 = self.cat_conv3(slot3)

        slot4 = torch.cat((x1_slot4,x2_slot4),dim=1)
        slot4 = self.cat_conv4(slot4)


        slot_combine = torch.cat((slot1,slot2,slot3,slot4), dim=1)

        out = self.conv(slot_combine)

        return out

class skip_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(skip_block, self).__init__()

        self.multi_scale = msc_block(in_channels, out_channels)
        self.multi_receptive = mrb_block(in_channels, out_channels)
        self.feat_merge = multi_channel_sep_conv(in_channels)

    def forward(self, x):
        msc = self.multi_scale(x)
        mrb = self.multi_receptive(x)

        merge_feat = self.feat_merge(msc, mrb)

        return merge_feat


class TransposedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super(TransposedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          )
        return x


class decoder(nn.Module):
    def __init__(self, factor):


        super(decoder,self).__init__()

        self.skip3 = skip_block(256//factor, 256//factor) 
        self.skip2 = skip_block(128//factor, 128//factor)
        self.skip1 = skip_block(64//factor, 64//factor)   
        
        self.conv_skip2 =  nn.Conv2d(256//factor+256//factor,  256//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)      

        self.Up4 = TransposedConv2d(256//factor, 128//factor)
        self.Up_conv4 =  nn.Conv2d(128//factor,  128//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_skip3 =  nn.Conv2d(128//factor+128//factor,  128//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.Up3 = TransposedConv2d(128//factor, 64//factor)
        self.Up_conv3 =  nn.Conv2d(64//factor,  64//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_skip4 =  nn.Conv2d(64//factor+64//factor,  64//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.Up2 = TransposedConv2d(64//factor, 32//factor)
        self.Up_conv2 =  nn.Conv2d(32//factor,  32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.tanh_out =  nn.Conv2d(32//factor,  3,  1)  

    def forward(self, x5_res3, x4, x3, x2):

        skip_3 = self.skip3(x4)
        cat_skip_3 = torch.cat([x5_res3,skip_3], axis=1)
        merge_cat_2 = self.conv_skip2(cat_skip_3)
        
        act_dec2 = self.Up4(merge_cat_2)#self.Up4(act1_dec1)
        act1_dec2 = self.Up_conv4(act_dec2)


        skip_2 = self.skip2(x3)
        cat_skip_2= torch.cat([act1_dec2,skip_2], axis=1)
        merge_cat_3 = self.conv_skip3(cat_skip_2)

        act_dec3 = self.Up3(merge_cat_3)
        act1_dec3 = self.Up_conv3(act_dec3)


        skip_1 = self.skip1(x2)
        cat_skip_1= torch.cat([act1_dec3,skip_1], axis=1)
        merge_cat_4 = self.conv_skip4(cat_skip_1)

        act_dec4 = self.Up2(merge_cat_4)
        act1_dec4 = self.Up_conv2(act_dec4)

        out1 = self.tanh_out(act1_dec4)

        return out1


class My_net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, factor = 1):

        super(My_net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2)
        self.Conv1 = nn.Conv2d(in_channels, 32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.Conv2 = nn.Conv2d(32//factor, 64//factor,  kernel_size = 3, stride=2, padding = 1, dilation=1)#depthwise_separable_conv(8,  16, 16)#128
        self.Conv3 = nn.Conv2d(64//factor, 128//factor,  kernel_size = 3, stride=2, padding = 1, dilation=1)#depthwise_separable_conv(16, 32, 32)#64
        self.Conv4 = nn.Conv2d(128//factor, 256//factor,  kernel_size = 3, stride=2, padding = 1, dilation=1)#depthwise_separable_conv(32, 64, 64)#32
                
        self.res_block = DeformableConv2d(256//factor, 256//factor)

        self.dec1 = decoder(factor)

    def forward(self, in1):

        x1 = self.Conv1(in1)#256*256*8

        x2 = self.Conv2(x1)#
        x3 = self.Conv3(x2)#
        x4 = self.Conv4(x3)#

        x5_res1 = self.res_block(x4)
        x5_res2 = self.res_block(x5_res1)
        x5_res3 = self.res_block(x5_res2)

        out1 = self.dec1(x5_res3, x4, x3, x2)       

        return out1


