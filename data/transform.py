from torchvision import transforms


class ImageNetNorm(object):
    def __init__(self, image_keys):
        self.image_keys = image_keys
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image):
        for key in self.image_keys:
            image[key] /= 255.0
            image[key] = self.normalize(image[key])
        return image


class SDTransform(object):
    def __init__(self, image_keys, img_size: tuple[int, int]=(768, 768), ensemble_size: int = 8):
        self.image_keys = image_keys
        self.ensemble_size = ensemble_size
        self.resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

    def __call__(self, image):
        for key in self.image_keys:
            image[key] = self.resize(image[key])
            image[key] = (image[key] / 255.0 - 0.5) * 2.0
            image[key] = image[key].unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)    # ensem, c, h, w

        return image