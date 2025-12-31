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