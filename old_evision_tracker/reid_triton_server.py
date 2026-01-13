# This code base on Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import cv2
import numpy as np
import torch
import torch.nn as nn

from configs.autocfg import cfg
from modules.triton_.tritonclient_ import TritonInfer
from tracker.utils import logger as LOGGER
from tracker.utils.checks import TestRequirements

tr = TestRequirements()


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights=cfg.MODEL.REID_TRT.M,
        device=torch.device("cpu"),
        fp16=False,
        model_version=cfg.MODEL.REID_TRT.V,
    ):
        super().__init__()

        self.fp16 = fp16
        self.device = device

        try:
            self.reid_model = TritonInfer(rec_name=weights, model_version=model_version)
            LOGGER.info(f"🤖 Loading reid model inference...")
        except:
            LOGGER.error("⛔ This model framework is not supported yet!")
            exit()

    def preprocess(self, xyxys, img):
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            box[box < 0] = 0  # FIXED BUG!!!
            x1, y1, x2, y2 = box.astype("int")
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                (128, 256),  # from (x, y) to (128, 256) | (w, h)
                interpolation=cv2.INTER_LINEAR,
            )

            # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_imgs = crop

            # normalization
            crop = crop / 255

            # standardization (RGB channel order)
            crop = crop - np.array([0.485, 0.456, 0.406])
            crop = crop / np.array([0.229, 0.224, 0.225])

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        crops = torch.stack(crops, dim=0)
        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(dtype=torch.half if self.fp16 else torch.float)

        return crops

    def forward(self, im_batch):
        # batch processing
        features = []

        # try:
        im_batch = im_batch.cpu().numpy()  # torch to numpy
        features = self.reid_model.forward(im_batch)[0]
        # except:
        #     LOGGER.error("Framework not supported at the moment, leave an enhancement suggestion")
        #     exit()

        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0])
                if len(features) == 1
                else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            im = self.preprocess(xyxys=np.array([[0, 0, 128, 256]]), img=im)
            self.forward(im)  # warmup

    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.preprocess(xyxys, img)
            features = self.forward(crops)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        return features