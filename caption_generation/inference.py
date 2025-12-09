import torch
import os
import sys
from PIL import Image


sys.path.insert(
    0, "/scratch/zt1/project/mqzhu-prj/user/yog/CHD/data/external/changechat/ChangeChat"
)

from changechat.model.builder import load_pretrained_model
from changechat.mm_utils import get_model_name_from_path, tokenizer_image_token
from changechat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from changechat.conversation import conv_templates
from changechat.utils import disable_torch_init


class ChangeChatInference:
    def __init__(self, model_path, model_base, vision_tower=None, num_gpus=4):
        self.model_path = model_path
        self.model_base = model_base
        self.vision_tower = vision_tower
        self.num_gpus = num_gpus
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Loading ChangeChat model...")
        self._load_model()

    def _load_model(self):
        original_dir = os.getcwd()
        changechat_dir = "/scratch/zt1/project/mqzhu-prj/user/yog/CHD/data/external/changechat/ChangeChat"
        os.chdir(changechat_dir)

        try:
            disable_torch_init()
            model_name = get_model_name_from_path(self.model_base)

            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=self.model_path,
                    model_base=self.model_base,
                    model_name="changechat_lora" + model_name,
                    load_8bit=False,
                    load_4bit=False,
                    device_map=None,
                )
            )
        finally:
            os.chdir(original_dir)

        if self.num_gpus > 1:
            from torch.nn import DataParallel

            device_ids = list(range(self.num_gpus))
            self.model = DataParallel(self.model, device_ids=device_ids)
            self.device = torch.device(f"cuda:{device_ids[0]}")
            self.is_data_parallel = True
        else:
            self.device = torch.device("cuda:0")
            self.is_data_parallel = False

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")

    def generate_caption(
        self, image_a, image_b, prompt=None, temperature=0.2, max_tokens=512
    ):
        if prompt is None:
            prompt = "Please briefly describe the changes in these two images."

        conv = conv_templates["v1"].copy()
        inp = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} {prompt}"
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        images = [image_a, image_b]
        images_tensor = self.image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ]
        images_tensor = images_tensor.half().to(self.device)

        model_to_use = self.model.module if self.is_data_parallel else self.model

        with torch.inference_mode():
            output_ids = model_to_use.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        vocab_size = len(self.tokenizer)
        filtered_ids = []
        for ids in output_ids:
            valid_ids = [int(id) for id in ids.tolist() if 0 <= id < vocab_size]
            filtered_ids.append(torch.tensor(valid_ids, device=ids.device))

        output = self.tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)[
            0
        ].strip()

        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:")[-1].strip()

        return output

    def generate_multiple_captions(self, image_a, image_b, num_captions=5):
        prompts = [
            "Please briefly describe the changes in these two images.",
            "What are the main differences between these two remote sensing images?",
            "Describe what has changed between the first and second image.",
            "What changes can you observe in these two images?",
            "Please describe the alterations visible in these two images.",
        ]

        temperatures = [0.2, 0.3, 0.4, 0.5, 0.6]

        captions = []
        for i in range(num_captions):
            prompt = prompts[i % len(prompts)]
            temp = temperatures[i % len(temperatures)]
            caption = self.generate_caption(
                image_a, image_b, prompt=prompt, temperature=temp
            )
            captions.append(caption)

        return captions
