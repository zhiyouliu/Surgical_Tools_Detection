# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

#from .coco import CocoDetection
from .coco_with_change_of_class import CocoDetection  # modified by StK: use this when using 
                                                      # the tag: --coco_re_mapped_class
