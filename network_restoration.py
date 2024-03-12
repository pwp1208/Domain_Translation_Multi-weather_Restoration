import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import matplotlib.pyplot as plt
import numpy as np

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.kv_conv = DeformableConv2d(channels * 2, channels *2)#nn.Conv2d(channels * 2, channels *2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, q):
        b, c, h, w = x.shape
        k, v = self.kv_conv(self.kv(x)).chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out, q

class Nested_MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(Nested_MDTA, self).__init__()
        self.pack_attention = MDTA(channels, num_heads)
        self.unpack_attention = MDTA(channels, num_heads)

    def forward(self,x, p):
        packed_context, query = self.pack_attention(x, p)
        unpacked_context, _ = self.unpack_attention(packed_context, query)
        return unpacked_context, packed_context

class FFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(FFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class LunaTransformerEncoderLayer(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(LunaTransformerEncoderLayer, self).__init__()
        self.luna_attention = Nested_MDTA(channels, num_heads)
        self.feed_forward = FFN(channels, expansion_factor)
        self.packed_context_layer_norm = nn.LayerNorm(channels)
        self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        # self.unpacked_context_layer_norm = nn.LayerNorm(channels)
        self.feed_forward_layer_norm = nn.LayerNorm(channels)
        self.conv = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)

    def forward(self, x, p):
        b, c, h, w = x.shape
        unpacked_context, packed_context = self.luna_attention(x,p)

        packed_context = self.packed_context_layer_norm((packed_context + p).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        unpacked_context = self.unpacked_context_layer_norm((unpacked_context + x).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        outputs = self.feed_forward(unpacked_context)

        outputs = self.feed_forward_layer_norm((outputs + unpacked_context).reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)

        out_cat =  torch.cat([outputs, packed_context],dim=1)

        return self.conv(out_cat)

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

class min_max_block(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(min_max_block, self).__init__()

        self.activation = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride=1, padding = padding, dilation=1)

    def forward(self, x):

        conv_out = self.conv1(x)
        
        maximum_tuple = torch.max(conv_out, dim=1)
        maximum = maximum_tuple[0].to("cuda:0")

        minimum_tuple = torch.min(conv_out, dim=1)
        minimum = minimum_tuple[0].to("cuda:0")       

        sub_activation = self.activation(torch.unsqueeze(maximum - minimum, axis=1))

        out = x * sub_activation

        return out

class merge_slots(nn.Module):
    def __init__(self, in_channels,kernel_size=3,padding=1):
        super(merge_slots, self).__init__()

        self.merge = feature_alignment(in_channels=in_channels,kernel_size=kernel_size, stride=1, padding=padding)
        self.conv_all = conv_block_my(in_channels*4, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3, x4):

        df1 = self.merge(x1, x2)
        df2 = self.merge(x3, df1)
        df3 = self.merge(x4, df2)

        all_together = torch.cat((x1, df1, df2, df3),dim=1)

        out = self.conv_all(all_together)
   
        return out

class multi_channel_sep_conv(nn.Module):
    def __init__(self, in_channels):
        super(multi_channel_sep_conv, self).__init__()

        self.conv = conv_block_my(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.cat_conv1 = merge_slots(in_channels=in_channels//4,kernel_size=1,padding=0)#DeformableConv2d(in_channels=(in_channels//4)*4, out_channels=in_channels//4, kernel_size=1, padding=0)
        self.cat_conv2 = merge_slots(in_channels=in_channels//4,kernel_size=3,padding=1)#DeformableConv2d(in_channels=(in_channels//4)*4, out_channels=in_channels//4, kernel_size=3, padding=1)
        self.cat_conv3 = merge_slots(in_channels=in_channels//4,kernel_size=5,padding=2)#DeformableConv2d(in_channels=(in_channels//4)*4, out_channels=in_channels//4, kernel_size=5, padding=2)
        self.cat_conv4 = merge_slots(in_channels=in_channels//4,kernel_size=7,padding=3)#DeformableConv2d(in_channels=(in_channels//4)*4, out_channels=in_channels//4, kernel_size=7, padding=3)

    
    def forward(self, x1, x2, x3, x4):
   
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

        x3_slot1 = x3[:,0:slot_size,:,:]
        x3_slot2 = x3[:,slot_size:slot_size*2,:,:]
        x3_slot3 = x3[:,slot_size*2:slot_size*3,:,:]
        x3_slot4 = x3[:,slot_size*3:slot_size*4,:,:]

        x4_slot1 = x4[:,0:slot_size,:,:]
        x4_slot2 = x4[:,slot_size:slot_size*2,:,:]
        x4_slot3 = x4[:,slot_size*2:slot_size*3,:,:]
        x4_slot4 = x4[:,slot_size*3:slot_size*4,:,:]

        slot1 = self.cat_conv1(x1_slot1,x2_slot1,x3_slot1,x4_slot1)

        slot2 = self.cat_conv2(x1_slot2,x2_slot2,x3_slot2,x4_slot2)

        slot3 = self.cat_conv3(x1_slot3,x2_slot3,x3_slot3,x4_slot3)

        slot4 = self.cat_conv4(x1_slot4,x2_slot4,x3_slot4,x4_slot4)

        slot_combine = torch.cat((slot1,slot2,slot3,slot4), dim=1)

        out = self.conv(slot_combine)

        return out


class feat_extract_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(feat_extract_block, self).__init__()

        self.multi_scale = msc_block(in_channels, out_channels)
        self.multi_receptive = mrb_block(in_channels, out_channels)
        self.conv_cat = conv_block_my(out_channels*2, out_channels, kernel_size = 1, stride = stride, padding = 0, dilation=1)

    def forward(self, x):

        msc = self.multi_scale(x)
        mrb = self.multi_receptive(x)

        cat_all = torch.cat((msc,mrb),dim=1)
        output = self.conv_cat(cat_all)

        return output

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

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

class feature_alignment(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(feature_alignment, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels*2, 
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
                                      out_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x, hfa_in):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        f_in = torch.cat([x, hfa_in], dim=1)
        # print(f_in.shape)
        offset = self.offset_conv(f_in)#.clamp(-max_offset, max_offset)
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

        self.Up5 = TransposedConv2d(256//factor, 128//factor)
        self.Up_conv5 =  nn.Conv2d(128//factor,  128//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_skip2 =  LunaTransformerEncoderLayer(128//factor, 8, 2.33) #nn.Conv2d(128//factor+128//factor,  128//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)
        

        self.Up4 = TransposedConv2d(128//factor, 64//factor)
        self.Up_conv4 =  nn.Conv2d(64//factor,  64//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_skip3 =  LunaTransformerEncoderLayer(64//factor, 8, 2.33) #nn.Conv2d(64//factor+64//factor,  64//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.Up3 = TransposedConv2d(64//factor, 32//factor)
        self.Up_conv3 =  nn.Conv2d(32//factor,  32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_skip4 =  LunaTransformerEncoderLayer(32//factor, 8, 2.33) #nn.Conv2d(32//factor+32//factor,  32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)


        self.tanh_out =  nn.Conv2d(32//factor,  3,  1)  

    def forward(self, x5_res3, x4, x3, x2):


        act_dec1 = self.Up5(x5_res3)
        act1_dec1 = self.Up_conv5(act_dec1)

        merge_cat_2 = self.conv_skip2(act1_dec1,x4)
        
        act_dec2 = self.Up4(merge_cat_2)#self.Up4(act1_dec1)
        act1_dec2 = self.Up_conv4(act_dec2)

        merge_cat_3 = self.conv_skip3(act1_dec2,x3)

        act_dec3 = self.Up3(merge_cat_3)
        act1_dec3 = self.Up_conv3(act_dec3)

        merge_cat_4 = self.conv_skip4(act1_dec3,x2)

        out1 = self.tanh_out(merge_cat_4)

        return out1

class restore_net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, factor = 1):

        super(restore_net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2)
        self.Conv11 = nn.Conv2d(in_channels, 32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.Conv12 = nn.Conv2d(in_channels, 32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.Conv13 = nn.Conv2d(in_channels, 32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.Conv14 = nn.Conv2d(in_channels, 32//factor,  kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.feat_extract11 = feat_extract_block(32//factor, 32//factor, stride=1)
        self.feat_extract12 = feat_extract_block(32//factor, 32//factor, stride=1)
        self.feat_extract13 = feat_extract_block(32//factor, 32//factor, stride=1)
        self.feat_extract14 = feat_extract_block(32//factor, 32//factor, stride=1)
        self.skip_l1 = multi_channel_sep_conv(32//factor)

        self.feat_extract21 = feat_extract_block(32//factor, 64//factor, 2)
        self.feat_extract22 = feat_extract_block(32//factor, 64//factor, 2)
        self.feat_extract23 = feat_extract_block(32//factor, 64//factor, 2)
        self.feat_extract24 = feat_extract_block(32//factor, 64//factor, 2)
        self.skip_l2 = multi_channel_sep_conv(64//factor)

        self.feat_extract31 = feat_extract_block(64//factor, 128//factor, 2)
        self.feat_extract32 = feat_extract_block(64//factor, 128//factor, 2)
        self.feat_extract33 = feat_extract_block(64//factor, 128//factor, 2)
        self.feat_extract34 = feat_extract_block(64//factor, 128//factor, 2)
        self.skip_l3 = multi_channel_sep_conv(128//factor)

        self.feat_extract41 = feat_extract_block(128//factor, 256//factor, 2)
        self.feat_extract42 = feat_extract_block(128//factor, 256//factor, 2)
        self.feat_extract43 = feat_extract_block(128//factor, 256//factor, 2)
        self.feat_extract44 = feat_extract_block(128//factor, 256//factor, 2)

        self.res_block1 = DeformableConv2d((256//factor) * 4, 256//factor)
        self.res_block2 = DeformableConv2d(256//factor, 256//factor)
        self.res_block3 = DeformableConv2d(256//factor, 256//factor)

        self.dec1 = decoder(factor)


    def forward(self, in1, in2, in3, in4):

        x11 = self.Conv11(in1)#256*256*8
        x12 = self.Conv12(in2)#256*256*8
        x13 = self.Conv13(in3)#256*256*8
        x14 = self.Conv14(in4)#256*256*8


        x21 = self.feat_extract11(x11)#
        x22 = self.feat_extract12(x12)#
        x23 = self.feat_extract13(x13)#
        x24 = self.feat_extract14(x14)#


        x31 = self.feat_extract21(x21)#
        x32 = self.feat_extract22(x22)#
        x33 = self.feat_extract23(x23)#
        x34 = self.feat_extract24(x24)#

        x41 = self.feat_extract31(x31)#
        x42 = self.feat_extract32(x32)#
        x43 = self.feat_extract33(x33)#
        x44 = self.feat_extract34(x34)#

        x51 = self.feat_extract41(x41)#
        x52 = self.feat_extract42(x42)#
        x53 = self.feat_extract43(x43)#
        x54 = self.feat_extract44(x44)#

        skip1 = self.skip_l1(x21,x22,x23,x24)
        skip2 = self.skip_l2(x31,x32,x33,x34)
        skip3 = self.skip_l3(x41,x42,x43,x44)

        cat_all= torch.cat([x51,x52,x53,x54], axis=1)

        x5_res1 = self.res_block1(cat_all)
        x5_res2 = self.res_block2(x5_res1)
        x5_res3 = self.res_block3(x5_res2)

        out1 = self.dec1(x5_res3, skip3, skip2, skip1)      

        return out1


