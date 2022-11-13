import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
    '''
    Optionaly seg decoder
    '''
    def __init__(self,
                 image_height,
                 image_width,
                 num_class,
                 prior_feat_channels=64,
                 refine_layers=3):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * refine_layers, num_class, 1)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        """
        Args:
            x: (B, refine_layers*C, fH, fW)
        Returns:
            x: (B, num_class, img_H, img_W)
        """
        x = self.dropout(x)
        # (B, refine_layers*C, fH, fW) --> (B, num_class, fH, fW)
        x = self.conv(x)
        # (B, num_class, fH, fW) --> (B, num_class, img_H, img_W)
        x = F.interpolate(x,
                          size=[self.image_height, self.image_width],
                          mode='bilinear',
                          align_corners=False)
        return x