import os
import numpy as np
import radii as radi
import transformTools as tr
import re
import time


class RadiiPipeline:
    def __init__(self, amInputPathList, maxZPathList, hocFile, outputFolder, amWithRad="default"):
        self.amInputPathList = amInputPathList
        self.maxZPathList = maxZPathList
        self.hocFile = hocFile
        self.amWithRad = amWithRad

        self.outputDirectory = self.initOutputDirectory(outputFolder)

        self.amWithErrorsDirectory = self.outputDirectory + "/amWithErrors/"
        self.amOutput025 = self.outputDirectory + "/am025/"
        self.amOutput050 = self.outputDirectory + "/am050/"
        self.amOutput075 = self.outputDirectory + "/am075/"
        self.hocFileOutput = self.outputDirectory + "hocFileWithRad.hoc"
        self.amWithUcrs = {}
        self.points025 = {}
        self.points050 = {}
        self.points075 = {}

    def runRayBurstOnSlices(self, tr025=True, tr050=True, tr075=True):
        self.createOutputDirectories()
        self.extractRadii(tr025, tr050, tr075)
        res = self.hocFile
        return res


    def initOutputDirectory(self, folder):
        cellPath = os.path.dirname(os.path.dirname(self.amInputPathList[0]))
        cellFolderName = os.path.basename(cellPath)
        outputDirectory = folder + cellFolderName
        if not (os.path.isdir(outputDirectory)):
            os.mkdir(outputDirectory)
        return outputDirectory


    def createOutputDirectories(self):


        if not os.path.isdir(self.amWithErrorsDirectory):
            os.mkdir(self.amWithErrorsDirectory)

        if not os.path.isdir(self.amOutput025):
            os.mkdir(self.amOutput025)

        if not os.path.isdir(self.amOutput050):
            os.mkdir(self.amOutput050)

        if not os.path.isdir(self.amOutput075):
            os.mkdir(self.amOutput075)


    def extractRadii(self, tr025=True, tr050=True, tr075=True):

        if not os.path.isdir(self.amWithErrorsDirectory):
            os.mkdir(self.amWithErrorsDirectory)

        if (tr025):
            radi025=radi.exRadSets.RadiusCalculatorForManyFiles(tresholdPercentage=0.25)
            for idx, amPth in enumerate(self.amInputPathList):
                am = os.path.basename(amPth)
                spatialGraphName = re.findall(r'[sS]\d+', am)[0]
                for imageFilePath in self.maxZPathList:
                    imageName = os.path.basename(imageFilePath)
                    if imageName.startswith(spatialGraphName):
                        radi025.exRadSets(amPth, imageFilePath, self.amOutput025, postMeasurment='yes')
                        break
        if (tr050):
            radi050=radi.exRadSets.RadiusCalculatorForManyFiles(tresholdPercentage=0.50)
            for idx, amPth in enumerate(self.amInputPathList):
                am = os.path.basename(amPth)
                spatialGraphName = re.findall(r'[sS]\d+', am)[0]
                for imageFilePath in self.maxZPathList:
                    imageName = os.path.basename(imageFilePath)
                    if imageName.startswith(spatialGraphName):
                        radi050.exRadSets(amPth, imageFilePath, self.amOutput050, postMeasurment='yes')
                        break

        if (tr075):
            radi075=radi.exRadSets.RadiusCalculatorForManyFiles(tresholdPercentage=0.75)
            for idx, amPth in enumerate(self.amInputPathList):
                am = os.path.basename(amPth)
                spatialGraphName = re.findall(r'[sS]\d+', am)[0]
                for imageFilePath in self.maxZPathList:
                    imageName = os.path.basename(imageFilePath)
                    if imageName.startswith(spatialGraphName):
                        radi075.exRadSets(amPth, imageFilePath, self.amOutput075, postMeasurment='yes')
                        break

    def extractUncertainties(self):
        self.readExtractedRadii()
        self.amWithUcrs = radi.calcError.addUncertainties(self.points050, self.points025, self.points075)
        self.writeUncertainties()

    def readExtractedRadii(self):
        print(self.amOutput025)
        self.allAmPointsWithRadius025, self.points025 = tr.read.multipleAmFiles(self.amOutput025)
        self.allAmPointsWithRadius050, self.points050 = tr.read.multipleAmFiles(self.amOutput050)
        self.allAmPointsWithRadius075, self.points075 = tr.read.multipleAmFiles(self.amOutput075)

    def writeUncertainties(self):
        for amInputPath in self.amInputPathList:
            tr.write.multipleAmFilesWithRadiusAndUncertainty(amInputPath, self.amWithErrorsDirectory, self.amWithUcrs)

    def getAmFileWithRad(self):

        assert self.amWithRad != "Default"
        amFile = self.amWithRad
        return amFile

    def findTransformation(self, amWithRad):

        assert self.amWithRad != "Default"

        self.amWithRad = amWithRad

        pointsWithRadius = tr.read.hocFileReduced(self.hocFile)
        set1 = []

        for el in pointsWithRadius:
            set1.append([el[0], el[1], el[2]])

        amFile = self.amWithRad

        set2 = radi.spacialGraph.getSpatialGraphPoints(amFile)

        numberOfEdges = 2

        matchedSet = tr.getDistance.matchEdges(set1, set2, numberOfEdges)

        src = []
        dst = []
        for point in matchedSet:
            src.append(point[0].start)
            dst.append(point[1].start)

            src.append(point[0].end)
            dst.append(point[1].end)

        print("In the calculations of the transofrmation matrix")
        trMatrix2 = tr.exTrMatrix.getTransformation(dst, src)


        amPoints4D = tr.read.amFile(amFile)

        print("Applying the transofrmation matrix to the initial am points")
        trAmPoints4D = []
        for point4D in amPoints4D:
            point = point4D[:3]
            point.append(1)
            p = np.dot(trMatrix2, np.array(point))
            p_listed = p.tolist()[0]
            trAmPoints4D.append([p_listed[0], p_listed[1], p_listed[2], point4D[3]])

        hocPointsComplete = tr.read.hocFileComplete(self.hocFile)
        hocSet = []
        for el in hocPointsComplete:
            hocSet.append([el[0], el[1], el[2]])

        trAmPoints4DList = trAmPoints4D

        print("In the process of finding pairs in between hoc file and the transoformed points to add radi to hocpoint")
        startTime = time.time()
        pairs = radi.addRadii.findPairs(trAmPoints4DList, hocSet)
        endTime = time.time()
        print(endTime - startTime)

        startTime = time.time()
        hocWithRad = []
        newHPoint = []

        for pair in pairs:
            newHPoint = pair[1]
            dual = pair[0]
            newHPoint.append(dual[3])
            hocWithRad.append(newHPoint)
            newHPoint = []

        endTime = time.time()
        print(endTime - startTime)

        print("writing the final result in the output hocFile")
        tr.write.hocFile(self.hocFile, self.hocFileOutput, hocWithRad)

        return 0
