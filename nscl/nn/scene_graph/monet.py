import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectEncoder(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder, or
    deconvolutional decoder.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{in}, W_{in})`
    """
    def __init__(self, in_channels, z_dim):
        super(ObjectEncoder, self).__init__()

        enc_convs = [nn.Conv2d(in_channels, out_channels=64,
                               kernel_size=4, stride=2, padding=1)]
        enc_convs.extend([nn.Conv2d(in_channels=64, out_channels=64,
                                    kernel_size=4, stride=2, padding=1)
                          for i in range(3)])
        self.enc_convs = nn.ModuleList(enc_convs)

        self.fc = nn.Sequential(nn.Linear(in_features=4096, out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256,
                                          out_features=z_dim))



    def encoder(self, x):
        batch_size = x.size(0)
        for module in self.enc_convs:
            x = F.relu(module(x))

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        rep = self.encoder(x)
        return rep



class UNetBlock(nn.Module):
    """Convolutional block for UNet, containing: conv -> instance-norm -> relu
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the block
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class UNet(nn.Module):
    """UNet, based on 'U-Net: Convolutional Networks for Biomedical
    Image Segmentation' by O. Ronneberger et al. It consists of contracting and
    expanding paths that at each block double and expand the size,
    respectively. Skip tensors are concatenated to the expanding path.
    A last 1x1 convolution reduces the number of channels to 1.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the network
        init_channels (int): Number of channels produced by the first block.
            This is doubled in subsequent blocks in the path. Default: 32
        depth (int): number of blocks in each path. Default: 3
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """
    def __init__(self, in_channels, out_channels, init_channels=32, depth=3):
        super(UNet, self).__init__()
        self.depth = depth

        self.down_blocks = nn.ModuleList()
        n_channels = init_channels
        for i in range(depth):
            self.down_blocks.append(UNetBlock(in_channels, n_channels))
            in_channels = n_channels
            n_channels *= 2
        n_channels //= 2

        mid_block = [UNetBlock(n_channels, n_channels * 2),
                     UNetBlock(n_channels * 2, n_channels)]
        self.mid_block = nn.Sequential(*mid_block)
        in_channels = 2 * n_channels
        n_channels //= 2

        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(UNetBlock(in_channels, n_channels))
            in_channels = 2 * n_channels
            n_channels //= 2
        n_channels *= 2

        self.last_conv = nn.Conv2d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_tensors = []
        for i, module in enumerate(self.down_blocks):
            x = module(x)
            skip_tensors.append(x)
            x = F.interpolate(x, scale_factor=0.5)

        x = self.mid_block(x)

        for block, skip in zip(self.up_blocks, reversed(skip_tensors)):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat((skip, x), dim=1)
            x = block(x)

        x = self.last_conv(x)

        return x


class AttentionNetwork(nn.Module):
    """A network that takes an image and a scope, to generate a mask for the
    part of the image that needs to be explained, and a scope for the next
    step.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` (image),
                 :math:`(N, 1, H_{in}, W_{in})` (scope)
        - Output: :math:`(N, 1, H_{out}, W_{out})` (mask),
                  :math:`(N, 1, H_{out}, W_{out})` (next scope)
    """
    def __init__(self, in_channels):
        super(AttentionNetwork, self).__init__()
        self.unet = UNet(in_channels, out_channels=1)

    def forward(self, x, log_scope):
        x = torch.cat((log_scope, x), dim=1)
        x = self.unet(x)
        log_mask = log_scope + F.logsigmoid(x)
        log_scope = log_scope + F.logsigmoid(-x)
        return log_mask, log_scope


class MONet(nn.Module):
    def __init__(self, im_x_size, im_y_size, im_channels, output_dims, args=None):
        super(MONet, self).__init__()

        self.object_encoder = ObjectEncoder(im_channels + 1, output_dims[1])
        self.attention = AttentionNetwork(in_channels=im_channels + 1)

        self.im_channels = im_channels
        self.output_dims = output_dims

        init_scope = torch.zeros((1, 1, im_x_size, im_y_size))
        self.register_buffer('init_scope', init_scope)

        self.obj1_linear = nn.Linear(output_dims[1],output_dims[1])
        self.obj2_linear = nn.Linear(output_dims[1],output_dims[1])

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)

    def objects_to_pair_representations(self, object_representations_batched):
        num_objects = object_representations_batched.size(1)

        obj1_representations = self.obj1_linear(object_representations_batched)
        obj2_representations = self.obj2_linear(object_representations_batched)

        obj1_representations.unsqueeze_(-1)#now batch_size x num_objects x feature_dim x 1
        obj2_representations.unsqueeze_(-1)

        obj1_representations = obj1_representations.transpose(2,3)
        obj2_representations = obj2_representations.transpose(2,3).transpose(1,2)

        obj1_representations = obj1_representations.repeat(1,1,num_objects,1)  
        obj2_representations = obj2_representations.repeat(1,num_objects,1,1)

        object_pair_representations = obj1_representations+obj2_representations
        object_pair_representations = object_pair_representations

        return object_pair_representations

        
    def forward(self, x, objects, objects_length):

        x = torch.nn.functional.interpolate(x, size=(128,128), mode='bilinear')

        batch_size = x.shape[0]
        max_num_objects = max(objects_length)


        log_scope = self.init_scope.expand(batch_size, -1, -1, -1)

        object_representations = []

        for slot in range(max_num_objects):
            if slot < max_num_objects - 1:
                log_mask, log_scope = self.attention(x, log_scope)
            else:
                log_mask = log_scope

            inp = torch.cat((x, log_mask), dim=1)
            object_rep = self.object_encoder(inp)

            object_representations.append(object_rep)


        object_values_batched = torch.stack(object_representations,dim=1)

        object_representations_batched = self._norm(object_values_batched)
        object_pair_representations_batched = self.objects_to_pair_representations(object_representations_batched)


        outputs = []
        for i in range(batch_size):
            num_objects = objects_length[i]
            object_representations = torch.squeeze(object_representations_batched[i,0:num_objects,:],dim=0)
            object_pair_representations = torch.squeeze(object_pair_representations_batched[i,0:num_objects,0:num_objects,:],dim=0).contiguous()
            
            outputs.append([
                        None,
                        object_representations,
                        object_pair_representations
                    ])


        return outputs



