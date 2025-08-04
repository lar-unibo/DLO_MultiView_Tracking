import cv2
import torch
import dlo_python.dlo_segmentation.model as network

# Set up model
MODEL_MAP = {
    "resnet50": network.deeplabv3plus_resnet50,
    "resnet101": network.deeplabv3plus_resnet101,
    "swinT": network.deeplabv3plus_swinT,
    "swinS": network.deeplabv3plus_swinS,
    "swinB": network.deeplabv3plus_swinB,
    "swinL": network.deeplabv3plus_swinL,
}


class DloSegmentationNetwork:

    def __init__(self, checkpoint_path) -> None:

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)

        self.img_h, self.img_w = checkpoint["img_h"], checkpoint["img_w"]
        print("Image size SEG: ", self.img_h, self.img_w)

        print("model selected: ", MODEL_MAP[checkpoint["backbone"]])
        self.model = MODEL_MAP[checkpoint["backbone"]](num_classes=1, output_stride=16, pretrained_backbone=False)
        network.convert_to_separable_conv(self.model.classifier)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded !")

    @torch.inference_mode()
    def predict(self, img):
        img = self.prepare_img(img)
        mask = self.model(img)
        mask = torch.sigmoid(mask)
        mask = mask.squeeze().detach().cpu().numpy()
        return mask

    @torch.inference_mode()
    def predict_batch(self, imgs):
        imgs = [self.prepare_img(img) for img in imgs]
        imgs = torch.cat(imgs, dim=0)
        masks = self.model(imgs)
        masks = torch.sigmoid(masks)
        masks = masks.squeeze().detach().cpu().numpy()
        return masks

    def prepare_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(self.device)
        return img
