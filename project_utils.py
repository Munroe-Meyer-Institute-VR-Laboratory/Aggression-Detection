import albumentations as A

# define the transforms
transform = A.Compose([
    A.Resize(224, 224, always_apply=True),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True)
])
