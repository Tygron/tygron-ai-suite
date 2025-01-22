from urllib.request import urlretrieve

def retrievePytorchDetectionLibs():
    urlretrieve("https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py", "engine.py")
    urlretrieve("https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py", "utils.py")
    urlretrieve("https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py", "coco_utils.py")
    urlretrieve("https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py", "coco_eval.py")
    urlretrieve("https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py", "transforms.py")