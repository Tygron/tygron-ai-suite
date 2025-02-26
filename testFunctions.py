import unittest
import logging
from inference_training import Configuration, ImageDataset

logger = logging.getLogger(__name__)

class TestFunction(unittest.TestCase):

    def testInvalidDatasetPath(self):

        config = createTestConfiguration()
        config.setDatasetPaths("./test_assets/nonexisting/train/",
                               "./test_assets/nonexisting/test/")
        trainingSet = ImageDataset(config, isTraining=True)
        testSet = ImageDataset(config, isTraining=False)

        self.assertFalse(trainingSet.validate())
        self.assertFalse(testSet.validate())

    def testValidDatasets(self):

        config = createTestConfiguration()
        config.setDatasetPaths("./test_assets/dataset_a/train/",
                               "./test_assets/dataset_a/test/")
        trainingSet = ImageDataset(config, isTraining=True)
        testSet = ImageDataset(config, isTraining=False)

        self.assertTrue(trainingSet.validate())
        self.assertTrue(testSet.validate())

    def testMissingFiles(self):

        config = createTestConfiguration()

        paths = ["missing_all/", "missing_mask/", "missing_labels/", "missing_image/" ]
        sizes = [0, 1,1 ,1]
        for i in range(0,len(paths)):
            
            config.setDatasetPaths("./test_assets/dataset_invalid/"+paths[i], "")
            trainingSet = ImageDataset(config, isTraining=True)

            self.assertEquals(sizes[i]>0 , trainingSet.validate())
            self.assertEquals(sizes[i],trainingSet.__len__(), "Failed on path: " + str(config.getTrainPath()))
            
        

    def testAutolimitLabelInDataset(self):

        config = createTestConfiguration()
        config.setDatasetPaths("./test_assets/dataset_label/train/",
                               "./test_assets/dataset_label/test/")
        trainingSet = ImageDataset(config, isTraining=True)
        testSet = ImageDataset(config, isTraining=False)

        logger.info("Test autolimit False")
        config.setAutoLimitLabel(False)

        self.assertFalse(trainingSet.validate())
        self.assertFalse(testSet.validate())

        logger.info("Test autolimit True")
        config.setAutoLimitLabel(True)
        
        self.assertTrue(trainingSet.validate())
        self.assertTrue(testSet.validate())


if __name__ == '__main__':
    unittest.main()


def createTestConfiguration() -> Configuration:

    config = Configuration()
    config.setFilePrefix("")
    config.setModelName("configmodel")
    config.setInputSizes(inputWidth=250, inputHeight=250)
    config.setInputCellSize(cellSizeM=0.25, minCellSizeM=0.1, maxCellSizeM=0.5)

    config.setModelInfo(channels=3, numClasses=5+1,  # (1 + background)
                        bboxOverlap=True, bboxPerImage=250, reuseModel=False)
    config.setEpochs(1)

    description = "Model description"
    config.setOnnxInfo(producer="Tygron", description=description)

    config.addLegendEntry("Background", 0, "#00000000")
    config.addLegendEntry("Label name 1", 1, "#00ffbf")
    config.addLegendEntry("Label name 2", 2, "#12d900")

    config.setOnnxMetaData(scoreThreshold=0.2,
                           maskThreshold=0.3,
                           strideFraction=0.5)
    
    config.setTensorInfo(tensorName='input_A:RGB_normalized', batchAmount=1)

    return config