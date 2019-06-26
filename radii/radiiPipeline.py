import os
import numpy as np
import radii as radi
import transformTools as tr
import re
import time


class RadiiPipeline:
    """
    Inputs:
    1. amInputPathList: an array contains the paths of am files
    2. maxZPathList: an array contains list of path of maxZ projection image files.
    3. outputFolder: Is a path to the outputFolder that the final result will be written.
    4. amWithRad: if provided then the class can be use to run the findTransformation method.

    """
    def __init__(self, amInputPathList, maxZPathList, hocFile, outputFolder, amWithRad="default"):
        self.amInputPathList = amInputPathList
        self.maxZPathList = maxZPathList
        self.hocFile = hocFile
        self.amWithRad = amWithRad
        self.spanFactor = 10.0

        self.outputDirectory = self.initOutputDirectory(outputFolder)

        self.amWithErrorsDirectory = self.outputDirectory + "/amWithErrors/"
        self.amOutput025 = self.outputDirectory + "/am025/"
        self.amOutput050 = self.outputDirectory + "/am050/"
        self.amOutput075 = self.outputDirectory + "/am075/"
        self.hocFileOutput = self.outputDirectory + "/hocFileWithRad.hoc"
        self.amWithUcrs = {}
        self.points025 = {}
        self.points050 = {}
        self.points075 = {}

    def runRayBurstOnSlices(self, tr025=True, tr050=True, tr075=True):
        """
        Will call the extrationRadii method for different Treshhold.
        each tresholdPercentage can be optionally removed by making its parameter false:
        like tr025 = Fales

        """
        self.createOutputDirectories()
        self.extractRadii(tr025, tr050, tr075)
        res = self.hocFile
        return res


    def initOutputDirectory(self, folder):
        """
        The ouptut directory need to be initilize to have different folders for each tresholdPercentage.
        if the folders are exist the function will not touch them
        """
        cellPath = os.path.dirname(os.path.dirname(self.amInputPathList[0]))
        cellFolderName = os.path.basename(cellPath)
        outputDirectory = folder + cellFolderName
        if not (os.path.isdir(outputDirectory)):
            os.mkdir(outputDirectory)
        return outputDirectory


    def createOutputDirectories(self):
     '''will creat the output directory with the name of the cell'''

        if not os.path.isdir(self.amWithErrorsDirectory):
            os.mkdir(self.amWithErrorsDirectory)

        if not os.path.isdir(self.amOutput025):
            os.mkdir(self.amOutput025)

        if not os.path.isdir(self.amOutput050):
            os.mkdir(self.amOutput050)

        if not os.path.isdir(self.amOutput075):
            os.mkdir(self.amOutput075)


    def extractRadii(self, tr025=True, tr050=True, tr075=True):
    '''will handle the calling of exRadSets function for different tresholdPercentages '''

        if not os.path.isdir(self.amWithErrorsDirectory):
            os.mkdir(self.amWithErrorsDirectory)

        if (tr025):
            radi025=radi.exRadSets.RadiusCalculatorForManyFiles(tresholdPercentage=0.25)
            for idx, amPth in enumerate(self.amInputPathList):
                am = os.path.basename(amPth)
                spatialGraphName = re.findall(r'[sS]\d+', am)[0]
                for imageFilePath in self.maxZPathList:
                    imageName = os.path.basename(imageFilePath)
                    if spatialGraphName in imageName:
                        radi025.exRadSets(amPth, imageFilePath, self.amOutput025, postMeasurment='yes')
                        break
        if (tr050):
            radi050=radi.exRadSets.RadiusCalculatorForManyFiles(tresholdPercentage=0.50)
            for idx, amPth in enumerate(self.amInputPathList):
                am = os.path.basename(amPth)
                spatialGraphName = re.findall(r'[sS]\d+', am)[0]
                for imageFilePath in self.maxZPathList:
                    imageName = os.path.basename(imageFilePath)
                    if spatialGraphName in imageName:
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
        '''extract uncertainties for diff. tresholds and add them to file by calling addUncertainties'''
        self.readExtractedRadii()
        self.amWithUcrs = radi.calcError.addUncertainties(self.points050, self.points025, self.points075)
        self.writeUncertainties()

    def readExtractedRadii(self):
        '''will hanld reading ampoints with radii and uncertainties for diff. tresholds'''
        print(self.amOutput025)
        self.allAmPointsWithRadius025, self.points025 = tr.read.multipleAmFiles(self.amOutput025)
        self.allAmPointsWithRadius050, self.points050 = tr.read.multipleAmFiles(self.amOutput050)
        self.allAmPointsWithRadius075, self.points075 = tr.read.multipleAmFiles(self.amOutput075)

    def writeUncertainties(self):
        '''will write uncertainties in the output files'''
        for amInputPath in self.amInputPathList:
            tr.write.multipleAmFilesWithRadiusAndUncertainty(amInputPath, self.amWithErrorsDirectory, self.amWithUcrs)

    def getAmFileWithRad(self):
        '''check if it need to ger amFile or not.'''
        assert self.amWithRad != "Default"
        amFile = self.amWithRad
        return amFile

    def findTransformation(self, amWithRad, spanFactor = 10.0, addRadii = True):

        """
        find the transformation between amFile and HocFile.
        inputs:
        1. amWithRad: final Provided amFile which contains the radii of points.
        2. spanFactor: Default value is 10.0, it will adjust the choosing of the edges by a span.
        3. addRadii: if it provided False it will not go to the step finding the pairs bbetween transformedPoints
        and HocPoints, this will lead to faster run for experimenting the transformion quality.

        outputs:
        1. provide the final hocPointsWithRad if addRadii is true.
        2. it will write the amTransformed and egAm and egHoc in text format files. These are by their order contain
        the transformed am points, choosed am edges, and choosed hoc edges.

        """

        assert self.amWithRad != "Default"

        self.amWithRad = amWithRad
        self.spanFactor = spanFactor

        pointsWithRadius = tr.read.hocFileComplete(self.hocFile)
        #  pointsWithRadius = tr.read.hocFileReduced(self.hocFile)
        hocPointsSet = []
        pairs = []

        for el in pointsWithRadius:
            hocPointsSet.append([el[0], el[1], el[2]])

        amFile = self.amWithRad

        amSet = radi.spacialGraph.getSpatialGraphPoints(amFile)

        numberOfEdges = 2

        matchedSet = tr.getDistance.matchEdges(hocPointsSet, amSet, numberOfEdges, spanFactor)

        src_hoc = []
        dst_am = []
        for point in matchedSet:
            src_hoc.append(point[0].start)
            src_hoc.append(point[0].end)

            dst_am.append(point[1].start)
            dst_am.append(point[1].end)

        print("In the calculations of the transofrmation matrix")
        trMatrix2 = tr.exTrMatrix.getTransformation(dst_am, src_hoc)

        amPoints4D = tr.read.amFile(amFile)

        print("Applying the transofrmation matrix to the initial am points")
        trAmPoints4D = []
        for point4D in amPoints4D:
            point = point4D[:3]
            mPoint = np.matrix(point)
            mTrPoint = mPoint.T

            p = trMatrix2*np.matrix(np.vstack((mTrPoint, 1.0)))
            p = np.array(p.T)
            p_listed = p.tolist()[0]
            # raw_input("somet")
            trAmPoints4D.append([p_listed[0], p_listed[1], p_listed[2], point4D[3]])

        hocPointsComplete = tr.read.hocFileComplete(self.hocFile)
        hocSet = []
        for el in hocPointsComplete:
            hocSet.append([el[0], el[1], el[2]])

        trAmPoints4DList = trAmPoints4D

        print("In the process of finding pairs in between hoc file and the transoformed points to add radi to hocpoint")
        startTime = time.time()
        if addRadii:
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

        if (addRadii):
            print("writing the final result in the output hocFile")
            tr.write.hocFile(self.hocFile, self.hocFileOutput, hocWithRad)

        egHocFile = self.outputDirectory + '/egHoc.txt'
        egAmFile = self.outputDirectory + '/egAm.txt'

        with open(egHocFile, 'w') as f:
            for item in matchedSet:
                startPoint = item[0].start
                endPoint = item[0].end
                f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
                f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))


        with open(egAmFile, 'w') as f:
            for item in matchedSet:
                startPoint = item[1].start
                endPoint = item[1].end
                f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
                f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))


        amTrFile = self.outputDirectory + '/amTransformed.txt'
        with open(amTrFile, 'w') as f:
            for it in trAmPoints4DList:
                f.write('{:f}\t{:f}\t{:f} \n'.format(it[0], it[1], it[2]))
        return 0
