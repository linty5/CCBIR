import albumentations as A
from albumentations.pytorch import ToTensorV2

class CompositionTrainTransform_A:
    def __init__(self, opt, total_mean=None, total_std=None):
        self.opt = opt
        self.mean_value = total_mean
        self.std_value = total_std

    def get_transform(self):
        transform = A.Compose([
            A.Resize(self.opt.size_h, self.opt.size_w),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5), 
                A.VerticalFlip(p=0.5)
                ], p=0.8),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.MotionBlur(p=0.5),
                A.GlassBlur(p=0.5),
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.5), 
            A.GaussNoise(var_limit=(0.0, 0.1), p=0.5),
            A.Normalize(mean=self.mean_value, std=self.std_value),
            ToTensorV2()
        ])
        return transform

class CompositionTestTransform_A:
    def __init__(self, opt, total_mean=None, total_std=None):
        self.opt = opt
        self.mean_value = total_mean
        self.std_value = total_std

    def get_transform(self):
        transform = A.Compose([
            A.Resize(self.opt.size_h, self.opt.size_w),
            A.Normalize(mean=self.mean_value, std=self.std_value),
            ToTensorV2()
        ])
        return transform
