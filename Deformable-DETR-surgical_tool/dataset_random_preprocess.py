import torch
from torchvision import transforms as T
import numpy as np

size_list = [(480, 640),
 (512, 682),
 (544, 725),
 (576, 768),
 (608, 810),
 (640, 853),
 (672, 896),
 (704, 938),
 (736, 981),
 (768, 1024),
 (800, 1066)]
def random_preprocess(img):
    """
        In this method, we provide a random image re-size preprocessing.
    """
    size_idx = np.random.choice(np.arange(0, len(size_list)))
    size = size_list[size_idx]
    return T.Compose([
                        T.ToTensor(),
                        T.Resize(size),
                        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # we cannot include this line at here. The current input image is only
                                                                                    # gray scale image, i.e. one channel. We can only apply this normalization
                                                                                    # later when we cast from 1-channel to 3-channel sample.
                    ])(img)
        
    