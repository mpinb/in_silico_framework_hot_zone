import os

class radii:
    def __init__(self, amInputPathList, maxZPathList, hocFile, outputDirectory="non"):
        self.amInputPathList = amInputPathList
        self.maxZPathList = maxZPathList
        self.hocFile = hocFile

        if outputDirectory == "non":
            self.createOutputDirectory()
        else:
            self.outputDirectory = outputDirectory
            if not (os.path.isdir(self.outputDirectory)):
                os.mkdir(self.outputDirectory)

        self.amOutput025 = self.outputDirectory + "/am025"
        self.amOutput050 = self.outputDirectory + "/am050"
        self.amOutput075 = self.outputDirectory + "/am075"

    def runRayBurstOnSlices(self):
        self.createOutputDirectories()
        self.readPoints()

        res = self.hocFile
        return res


    def createOutputDirectory(self):
        cellPath = os.path.dirname(os.path.dirname(self.amInputPathList[0]))
        self.outputDirectory = cellPath + "/output"
        os.mkdir(self.outputDirectory)

    def createOutputDirectories(self):
        if not os.path.isdir(self.amOutput025):
            os.mkdir(self.amOutput025)

        if not os.path.isdir(self.amOutput050):
            os.mkdir(self.amOutput050)

        if not os.path.isdir(self.amOutput075):
            os.mkdir(self.amOutput075)
