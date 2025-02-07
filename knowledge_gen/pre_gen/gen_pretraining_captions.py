import sys
import os
sys.path.append(os.path.abspath("PATH FOR CLIP-ViP"))

from src.utils.load_save import (ModelSaver,
                                 BestModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.modeling.VidCLIP import VidCLIP
args = {
    "clip_config": "YOUR CLIP PATH CONFIG",
    "clip_vision_additional_config": {'type': "ViP", 
                                      "logit_scale_init_value": 4.6,
                                      "temporal_size": 12,
                                        "if_use_temporal_embed": 1,
                                        "logit_scale_init_value": 4.60,
                                        "add_cls_num": 3
                                      },
    "clip_weights": "YOUR CLIP WEIGHT",
    "e2e_weights_path": "YOUR CLIP-ViP WEIGHT PATH"

}
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

args = DictToObject(args)
model = VidCLIP(args)
load_state_dict_with_mismatch(model, args.e2e_weights_path)

from transformers import CLIPTokenizer, CLIPModel
tokenizer = CLIPTokenizer.from_pretrained("YOUR CLIP-ViP WEIGHT PATH")
inputs = tokenizer(["a photo of a cat", "a photo of a dog", "this is video is "], padding=True, return_tensors="pt")
print(inputs)
text_features = model.forward_text(inputs['input_ids'], inputs['attention_mask'])


