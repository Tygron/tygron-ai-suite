import os
import logging
import onnx
import torch
from pathlib import Path
import matplotlib.pyplot as pyplot
import numpy as np
import torchvision
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
import torch.nn as nn
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import utils
import datetime
from engine import train_one_epoch, evaluate

# from torch.nn import Sequential,ModuleList
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.ops import misc as misc_nn_ops

logger = logging.getLogger(__name__)

def initCudaEnvironment(numCudaDevices: int = 1,
                        visibleCudaDevices: str = "0",
                        clearCudaDeviceCount: bool = False):
    os.environ["CUDA_VISIBLE_DEVICES"] = visibleCudaDevices
    os.environ["WORLD_SIZE"] = str(numCudaDevices)
    if clearCudaDeviceCount:
        torch.cuda.device_count.cache_clear()


def seperateMasks(mask):
    # instances are encoded as different colors
    obj_ids = torch.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    return (mask == obj_ids[:, None, None])


def showMasks(image, masks, labels):
    drawn_masks = []
    maskIdx = 0
    for mask in masks:
        maskColor = "blue"
        if labels[maskIdx] == 2:
            maskColor = "red"
        elif labels[maskIdx] == 3:
            maskColor = "purple"

        drawn_masks.append(draw_segmentation_masks(image, mask, alpha=0.8, colors=maskColor))
        maskIdx += 1

    show(drawn_masks)


def showBBoxes(image, masks):
    # get bounding box coordinates for each mask
    boxes = masks_to_boxes(masks)
    drawn_boxes = draw_bounding_boxes(image, boxes, colors="red")
    show(drawn_boxes)


def show(imgs, cols=3):

    rows = len(imgs) // cols;
    if len(imgs) % cols > 0:
        rows += 1

    if not isinstance(imgs, list):
        imgs = [imgs]
    
    fig, axs = pyplot.subplots(nrows=rows, ncols=cols, squeeze=False,layout='compressed',
                               figsize=(100,100))

    for i, img in enumerate(imgs):
        img = img.detach()
        img = v2.functional.to_pil_image(img)
        axs[i//cols, i % cols].imshow(np.asarray(img))
        axs[i//cols, i % cols].set(xticklabels=[], yticklabels=[], 
                                   xticks=[], yticks=[])


def createTransforms(train: bool):
    transforms = []

    # Disable, random is evil, and colorjitter only on input images
    # if train:
        # transforms.append(v2.RandomHorizontalFlip(0.5))
        # transforms.append(v2.ColorJitter(brightness=.5, saturation=0.05))

    transforms.append(v2.ToDtype(torch.float, scale=True))
    transforms.append(v2.ToPureTensor())
    return v2.Compose(transforms)


class LegendEntry:
    def __init__(self, name: str, value: float,
                 color: str = "#00ffbf"):
        self.name = name
        self.value = value
        self.color = color


class Configuration:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.filePrefix = ""
        self.modelPrefix = "inference"
        self.epochs = 20
        self.legendEntries = []
        self.imagePredicate = lambda f: str(f).endswith("image.png")
        self.labelPredicate = lambda f: str(f).endswith("labels.txt")
        self.maskPredicate = lambda f: str(f).endswith("mask.png")
        self.bboxPredicate = lambda f: str(f).endswith("bboxs.txt")

        self.inputWidth = 250
        self.inputHeight = 250
        self.numClasses = 1+1
        self.cellSizeM = 0.25
        self.bboxPerImage = 250
        self.inferenceMode = 1
        self.MAX_ERROR_COUNT = 10
        self.MAX_CLASSES = 250
        self.autoLimitLabel = False
        self.isCrowd = False

        self.version = datetime.date.today().strftime("%Y%m%d")

    def setModelName(self, modelName: str):
        self._modelName = modelName

    def getPytorchModelFileName(self):
        return self.getModelName() + ".pt"

    def getOnnxFileName(self):
        return self.getModelName() + ".onnx"

    def getModelName(self):
        if hasattr(self, '_modelName'):
            return self._modelName

        modelname = self.modelPrefix + "_" + str( self.numClasses-1) + "c_" + self.getCellSizeCM() + "_" + str(self.epochs) + "e_" + str(self.bboxPerImage) +"_maxgt"
        return modelname
    
    def setDatasetPaths(self, trainPath: str, testPath: str):
        self.trainPath = Path(trainPath)
        self.testPath = Path(testPath)

    def getTrainPath(self):
        return self.trainPath

    def getTestPath(self):
        return self.testPath

    def setFilePrefix(self, prefix: str):
        self.filePrefix = prefix

    def setInputSizes(self,
                      inputWidth: int = 250,
                      inputHeight: int = 250):
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight

    def setInputCellSize(self, cellSizeM: float = 0.25,
                         minCellSizeM: float = None,
                         maxCellSizeM: float = None):
        self.cellSizeM = cellSizeM
        if minCellSizeM is not None:
            minCellSizeM = self.cellSizeM
        self.minCellSizeM = min(minCellSizeM, self.cellSizeM)

        if maxCellSizeM is not None:
            maxCellSizeM = self.cellSizeM
        self.maxCellSizeM = max(maxCellSizeM, self.cellSizeM)

    def getCellSizeCM(self):
        return str(int(self.cellSizeM*100)) + "cm"

    def setVersion(self, version: int):
        self.version = str(version)

    def setEpochs(self, epochs: int):
        self.epochs = int(epochs)

    def setAutoLimitLabel(self, autoLimit: bool):
        self.autoLimitLabel = autoLimit

    def setIsCrowd(self, isCrowd: bool):
        self.isCrowd = isCrowd

    def setTensorInfo(self, tensorName: str = "input_A:RGB_normalized",
                      batchAmount: int = 1):
        self.tensorName = str(tensorName)
        self.batchAmount = int(batchAmount)

    def setModelInfo(self,
                     channels: int = 3,
                     numClasses: int = 2,
                     bboxOverlap: bool = True,
                     bboxPerImage: int = 250,
                     reuseModel: bool = False):
        self.channels = channels
        self.numClasses = numClasses
        self.bboxOverlap = bboxOverlap
        self.bboxPerImage = bboxPerImage
        self.reuseModel = reuseModel

    def clearLegendEntries(self):
        self.legendEntries = []

    def addLegendEntry(self, name: str, value: int,
                       color: str = "#00ffbf"):
        self.legendEntries.append(LegendEntry(name, value, color))

    def getLegendEntries(self):
        return self.legendEntries

    def setOnnxInfo(self,
                    producer: str,
                    description: str):
        self.producer = producer
        self.description = description

    def setOnnxMetaData(self, scoreThreshold=0.2,
                        maskThreshold=0.3,
                        strideFraction=0.5):
        self.scoreThreshold = scoreThreshold
        self.maskThreshold = maskThreshold
        self.strideFraction = strideFraction
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, configuration: Configuration,
                 isTraining: bool = True,
                 transforms=None, imageTransforms=None):

        self.config = configuration
        self.transforms = transforms
        self.imageTransforms = imageTransforms
        self.isTraining = isTraining
        self.images = []
        self.masks = []
        self.labels = []

        try:
            files = self.getFiles()
            self.images = [f.path for f in files if configuration.imagePredicate(f.path)]
            self.masks = [f.path for f in files if configuration.maskPredicate(f.path)]
            self.labels = [f.path for f in files if configuration.labelPredicate(f.path)]
        except FileNotFoundError:
            logger.warning("The datasets directory does not exist: "+str(self.getDirectory()))
            logger.warning("Please adjust the configured dataset directory in the configuration file!")


    def getDirectory(self):
        return self.config.getTrainPath() if self.isTraining else self.config.getTestPath()
    
    def getFiles(self):
        files = listFilesRecursive(self.getDirectory(), files=[])
        return sorted(files, key=lambda f: f.path)
        
    def getAssetNumber(self, text: str):
        prefix = self.config.filePrefix
        if len(prefix) > 0 and text.startswith(prefix):
            text = text[:len(prefix)]

        return text.split("_")[0]

    def getImage(self, idx):
        return read_image(self.images[idx], mode=ImageReadMode.RGB)[:3]

    def getImages(self, idx):
        return [self.getImage(idx)]

    def getMask(self, idx):
        return read_image(self.masks[idx], mode=ImageReadMode.GRAY)[0]

    def getLabels(self, idx):

        lines = open(self.labels[idx], "r").read().splitlines()
        result = []

        for line in lines:
            for value in line.split():

                intValue = int(value)
                if self.config.autoLimitLabel:
                    intValue = min(intValue, self.config.numClasses-1)

                result.append(intValue)
        return result

    def getMaxLabel(self):
        maxLabel = 0
        for i in range(self.__len__()):
            if len(self.getLabels(i)) > 0:
                maxLabel = max(maxLabel, max(self.getLabels(i)))
        return maxLabel

    def __getitem__(self, idx):
        # load images and masks
        imgs = self.getImages(idx)
        mask = self.getMask(idx)

        for i in range(0, len(imgs)):
            imgs[i] = v2.functional.convert_image_dtype(imgs[i], 
                                                        dtype=torch.float)

        mask = v2.functional.convert_image_dtype(mask, dtype=torch.float)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None])

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        labels = self.getLabels(idx)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxArea = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(boxArea, dtype=torch.int64)

        # suppose all instances are crowd
        iscrowd = torch.ones(num_objs, dtype=torch.int64) if self.config.isCrowd else torch.zeros(num_objs, dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        for i in range(0, len(imgs)):
            imgs[i] = tv_tensors.Image(imgs[i])

        tensorMasks = tv_tensors.Mask(masks)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                                   canvas_size=v2.functional.get_size(imgs[0]))
        target["masks"] = tensorMasks

        if self.imageTransforms is not None:
            for i in range(0, len(imgs)):
                imgs[i] = self.imageTransforms(imgs[i])

        if self.transforms is not None:
            for i in range(0, len(imgs)):
                imgs[i] = self.transforms(imgs[i])
            target = self.transforms(target)

        target["labels"] = labels
        target["area"] = area
        target["image_id"] = idx
        target["iscrowd"] = iscrowd

        # https://pytorch.org/docs/stable/generated/torch.cat.html
        img = torch.cat(imgs, 0)

        return img, target

    def __len__(self):
        return min(len(self.images), len(self.masks), len(self.labels))

    def getName(self):
        return "Training dataset" if self.isTraining else "Test dataset"
		
	# get transforms applied to both image and target (masks etc)
    def getTransforms(self):
        return self.transforms
	
	# get transforms only applied to images (not masks)
    def getImageTransforms(self):
        return self.imageTransforms	

	# set transforms applied to both image and target (masks etc)
    def setTransforms(self, transforms):
        self.transforms = transforms
	
	# set transforms only applied to images (not masks)
    def setImageTransforms(self, imageTransforms):
        self.imageTransforms = imageTransforms

    def validate(self):
        
        if self.__len__() <= 0:
            logger.warning("No files were found in the " + self.getName())
            return False    
        
        validFiles = self.validateFiles()
        if validFiles:
            logger.info("File check passed for " + self.getName())

        validLabels = self.validateLabels()
        if validLabels:
            logger.info("Validation check passed for " + self.getName())

        return validFiles and validLabels

    def validateFiles(self):
        errorCount = 0
        for i in range(0, self.__len__()):

            index = self.getAssetNumber(self.images[i])
            maskIndex = self.getAssetNumber(self.masks[i])
            if index != maskIndex:
                errorCount += 1
                if errorCount < self.config.MAX_ERROR_COUNT:
                    logger.warning("mask is missing for image " + self.images[i])

            labelIndex = self.getAssetNumber(self.labels[i])
            if index != labelIndex:
                errorCount += 1
                if errorCount < self.config.MAX_ERROR_COUNT:
                    logger.warning("labels are missing for image " + self.images[i])

        return errorCount == 0

    def validateLabels(self):

        maxLabel = self.getMaxLabel()
        valid = maxLabel <= self.config.numClasses

        if not valid:
            logger.warning("More classes found than expected: " +
                                       str(maxLabel) + " vs " +
                                       str(self.config.numClasses))

            if maxLabel <= self.config.MAX_CLASSES:
                logger.warning("Please adjust the Configuration parameter numClasses to: " + str(maxLabel))
                logger.warning("Or set the autoLimitLabel option of Configuration to True")

            else:
                logger.warning("More classes are defined in the label files than the trained model can handle.")
                logger.warning("Please set the autoLimitLabel option of Configuration to True")

        return valid


def listFilesRecursive(path, files=[]):

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry)
            elif entry.is_dir():
                listFilesRecursive(entry.path, files)
    return files


def createSGDOptimizer(model):

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,    lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    return optimizer


def trainModel(config: Configuration,
               trainingDataset: ImageDataset,
               testDataset: ImageDataset,
               model=None,
               trainingDataLoader=None,
               testDataLoader=None,
               optimizer=None,
               lrSceduler=None,
               evaluateModel: bool = True):

    if trainingDataLoader is None:
        if trainingDataset is None:
            logger.warning("Please provide a training dataset or dataset loader")
            return

        elif not trainingDataset.validate():
            logger.warning("Training dataset is invalid, please inspect the logs.")
            return

        else:
            trainingDataLoader = torch.utils.data.DataLoader(
                trainingDataset,
                batch_size=2,
                shuffle=True,
                collate_fn=utils.collate_fn
            )

    if testDataLoader is None:
        if testDataset is None:
            logger.warning("Please provide a test dataset or dataset loader")
            return
        elif not testDataset.validate():
            logger.warning("Test dataset is invalid, please inspect the logs.")
            return

        else:
            testDataLoader = torch.utils.data.DataLoader(
                testDataset,
                batch_size=1,
                shuffle=False,
                collate_fn=utils.collate_fn
            )

    if model is None:
        # get the model using our helper function
        model = createModelInstance(config)

    if optimizer is None:
        optimizer = createSGDOptimizer(model)

    # and a learning rate scheduler
    if lrSceduler is None:
        lrScheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

    hasModel = os.path.exists(config.getPytorchModelFileName())

    if hasModel and config.reuseModel:
        checkpoint = torch.load(config.getPytorchModelFileName(),
                                weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()

    for epoch in range(config.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, trainingDataLoader, config.device, 
                        epoch, print_freq=10)
        # update the learning rate
        lrScheduler.step()
        # evaluate on the test dataset
        evaluate(model, testDataLoader, device=config.device)

    if evaluateModel:
        model.eval()

    return model


def createModelInstance(config: Configuration):

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT", 
                                                               box_detections_per_img=config.bboxPerImage)

    logger.info("Detections per image " + str(model.roi_heads.detections_per_img))
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.numClasses)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    logger.info("in features mask: " + str(in_features_mask))

    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        config.numClasses
    )

    # move model to the right device
    model = model.to(config.device)

    return model


def get_resnet50(config: Configuration, weights=None):
    model = resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=config.numClasses)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(config.numChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model


def createLabelNamesArray(config: Configuration):
    label_names = ["0"]
    for i in range(1, config.numClasses+1):
        label_names.append(str(i))


def get_faster_rcnn(config: Configuration, weights=None):

    weights_backbone = None
    trainable_backbone_layers = None

    is_trained = weights is not None or weights_backbone is not None

    trainable_backbone_layers = _validate_trainable_layers(
        is_trained,
        trainable_backbone_layers,
        5, 3)

    #norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = get_resnet50(config.numClasses, config.channels)
    # resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    # ResNet50 output channels is 2048
    # backbone.out_channels = 2048
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    label_names = createLabelNamesArray(config)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=label_names,
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = MaskRCNN(backbone=backbone,
                     num_classes=config.numClasses,
                     box_roi_pool=roi_pooler,
                     image_mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                     image_std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
                     box_detections_per_img=config.bboxPerImage)

    return model


def createMultichannelModelInstance(config: Configuration,
                                    weights=None):

    # load an instance segmentation model pre-trained on COCO
    model = get_faster_rcnn(config, weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.numClasses)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        config.numClasses
    )

    return model


def drawImageAndFeatureMasks(config: Configuration,
                             dataset: ImageDataset,
                             imageNumber: int):

    image = dataset.getImages(imageNumber)[0]
    mask = dataset.getMask(imageNumber)
    labels = dataset.getLabels(imageNumber)

    masks = seperateMasks(mask)

    showMasks(image, masks, labels)
    showBBoxes(image, masks)
    
def drawTransformedImageAndFeatureMasks(config: Configuration,
                             dataset: ImageDataset,
                             imageNumber: int):
    itemTuple = dataset.__getitem__(imageNumber)
    image = itemTuple[0]
    mask = itemTuple[1]["masks"]
    labels = dataset.getLabels(imageNumber)

    masks = seperateMasks(mask)

    showMasks(image, masks, labels)
    showBBoxes(image, masks)


def testInference(config: Configuration,
                  dataset: ImageDataset, model,
                  imageNumber: int):

    imgs = dataset.getImages(imageNumber)
    for i in range(0, len(imgs)):
        imgs[i] = v2.functional.convert_image_dtype(imgs[i], dtype=torch.float)
        imgs[i] = tv_tensors.Image(imgs[i])

    # https://pytorch.org/docs/stable/generated/torch.cat.html
    img = torch.cat(imgs, 0) 
    eval_transform = createTransforms(train=False)

    with torch.no_grad():
        x = eval_transform(img)
        x = x.to(config.device)
        predictions = model([x, ])
        pred = predictions[0]

    image = imgs[0]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"numbers: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()

    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > config.maskThreshold).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    pyplot.figure(figsize=(64, 64))
    pyplot.subplot(121)
    pyplot.imshow(output_image.permute(1, 2, 0))

    pyplot.subplot(122)
    pyplot.title("Image")
    pyplot.imshow(image.permute(1, 2, 0))
    return pred


def createOnnxInput(config):
    return torch.randn(config.batchAmount,
                       config.channels,
                       config.inputWidth,
                       config.inputHeight)


def toNumpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def exportOnnxModel(config: Configuration, model, opsetVersion=11):
    device = torch.device('cpu')
    onnx_input = createOnnxInput(config)
    model.to(device)
    model.eval()

    # Export the model
    torch.onnx.export(model,                 # model being run
                  onnx_input,                # model input (or a tuple for multiple inputs)
                  config.getOnnxFileName(),  # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=opsetVersion,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = [config.tensorName],   # the model's input names
                  output_names = ['boxes', 'labels','scores','masks'],) # the model's output names)

    
    model.to(config.device)
    model.eval()


def loadONNX(config: Configuration):
    return onnx.load(config.getOnnxFileName())


def writeONNXMeta(config: Configuration, onnxModel=None):

    if onnxModel is None:
        onnxModel = loadONNX(config)
    del onnxModel.metadata_props[:]

    addMeta(onnxModel, "VERSION", str(config.version))
    addMeta(onnxModel, "DESCRIPTION", config.description)
    addMeta(onnxModel, "PRODUCER", config.producer)

    addMeta(onnxModel, "MIN_CELL_SIZE_M", str(config.minCellSizeM))
    addMeta(onnxModel, "MAX_CELL_SIZE_M", str(config.maxCellSizeM))
    addMeta(onnxModel, "MASK_THRESHOLD", str(config.maskThreshold))
    addMeta(onnxModel, "SCORE_THRESHOLD", str(config.scoreThreshold))
    addMeta(onnxModel, "INFERENCE_MODE", str(config.inferenceMode))
    addMeta(onnxModel, "BOX_OVERLAP", "1" if config.bboxOverlap else "0")
    addMeta(onnxModel, "STRIDE_FRACTION",  str(config.strideFraction))
    addMeta(onnxModel, "MAX_GT_INSTANCES", str(config.bboxPerImage))

    legendNames = ', '.join(e.name for e in config.legendEntries)
    legendValues = ', '.join(str(e.value) for e in config.legendEntries)
    legendColors = ', '.join(e.color for e in config.legendEntries)

    addMeta(onnxModel, "LEGEND_NAMES", legendNames)
    addMeta(onnxModel, "LEGEND_VALUES", legendValues)
    addMeta(onnxModel, "LEGEND_COLORS", legendColors)

    setProducer(onnxModel, config.producer)
    setModelVersion(onnxModel, int(config.version))
    setDocString(onnxModel, config.description)

    with open(config.getOnnxFileName(), "wb") as f:
        f.write(onnxModel.SerializeToString())


def getModelProducer(onnxModel):
    return onnxModel.producer_name


def getModelVersion(onnxModel):
    return onnxModel.model_version


def getModelDescription(onnxModel):
    return onnxModel.doc_string


def setProducer(onnxModel, producer):
    onnxModel.producer_name = producer


def setModelVersion(onnxModel, modelVersion):
    onnxModel.model_version = modelVersion


def setDocString(onnxModel, description):
    onnxModel.doc_string = description


def addMeta(onnx_model, key, value):

    for entry in onnx_model.metadata_props:
        if entry.key == key:
            entry.value = value
            return

    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value


def saveModel(config: Configuration, model, path: str = None):
    if path is None:
        path = config.getPytorchModelFileName()

    torch.save(model.state_dict(), path)


def loadModel(config: Configuration, model, path: str = None):
    if path is None:
        path = config.getPytorchModelFileName()
    model.load_state_dict(torch.load(path, weights_only=True))
