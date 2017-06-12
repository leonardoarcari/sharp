#!/usr/bin/env python3
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib
import re
import subprocess
from time import sleep

MAGIC_NUMBER = 10

def plotExecutionTime(shapePath, xValues, yValues, numThreads):
	newX, newY = zip(*sorted(zip(xValues, yValues)))
	maxX = newX[-1]
	minY = newY[-1]
	maxY = newY[0]

	print(newY)

	figName = shapePath.stem + '_plot' + '_' + str(numThreads) + '.png'

	fig, ax = plt.subplots()
	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_major_locator(ticker.MaxNLocator(integer=True))

	plt.plot(newX, newY, 'b-.')
	plt.axis([1-0.15, maxX+0.15, minY - 10, maxY + 10])
	plt.title(shapePath.name)
	plt.xlabel('Number of threads')
	plt.ylabel('Execution time (ms)')
	fig.savefig(figName)

def runScalabilityTest(refShapes, testShapes, maxThreads, shapeSize, minTheta, maxTheta, thetaStep, lenThresh):
	refPath = pathlib.Path(refShapes).resolve()
	testPath = pathlib.Path(testShapes).resolve()
	testShapes = [s for s in testPath.iterdir() if s.is_file() and s.suffix.lower() in [".jpg", ".jpeg", ".png"]]

	args = ["./sharp"]
	args.extend(["-r", str(refPath)])
	args.extend(["--shape-size", str(shapeSize)])
	args.extend(["--min-theta", str(minTheta)])
	args.extend(["--max-theta", str(maxTheta)])
	args.extend(["--length-thresh", str(lenThresh)])

	execTimeRE = '\[Thread \d+\] Time spent in shape recognition: \d+ms'

	statistics = dict()

	for ts in testShapes:
		testList = list(args)
		testList.extend(["--test-shape", str(ts.resolve())])

		statistics[ts.name] = dict()

		threadRange = [2**x for x in range(int(math.log2(maxThreads)) + 1)]
		# print("ThreadRange =", threadRange)
		for t in threadRange:
			sum_time = 0.0
			threadList = list(testList)
			threadList.extend(["--threads", str(t)])
			for i in range(MAGIC_NUMBER):
				sleep(0.01)
				# print("Threadlist:", threadList)
				cp = subprocess.run(threadList, stdout=subprocess.PIPE, check=True)
				stdout = cp.stdout.decode("utf-8")
				print("SHARP running with test image:", ts.name, "on", t, "threads")
				print(stdout)
				target = re.search(execTimeRE, stdout)
				if target:
					time = re.search('\d+ms', target.group())
					if time:
						time = time.group()[:-2]
						sum_time += float(time)
			statistics[ts.name][t] = sum_time/MAGIC_NUMBER
		plotExecutionTime(ts, list(statistics[ts.name].keys()), list(statistics[ts.name].values()), maxThreads)

	print(statistics)

# def runRecognitionTest(refShapes, testShapes, maxThreads, shapeSize, minTheta, maxTheta, thetaStep, lenThresh):
# 	refPath = pathlib.Path(refShapes).resolve()
# 	testPath = pathlib.Path(testShapes).resolve()
# 	testShapes = [s for s in testPath.iterdir() if s.is_file() and s.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# 	args = ["./sharp"]
# 	args.extend(["-r", str(refPath)])
# 	args.extend(["--shape-size", str(shapeSize)])
# 	args.extend(["--min-theta", str(minTheta)])
# 	args.extend(["--max-theta", str(maxTheta)])
# 	args.extend(["--length-thresh", str(lenThresh)])
# 	args.extend(["--threads", str(maxThreads)])

# 	finalScoreRE = '\[Thread \d+\] Final score: \[(\d+(\.\d+)?, )+\d+(\.\d+)?\]'

# 	statistics = dict()
# 	max_values = dict()

# 	for ts in testShapes:
# 		testList = list(args)
# 		testList.extend(["--test-shape", str(ts.resolve())])

# 		statistics[ts.name] = dict()
# 		max_values[ts.name] = dict()

# 		cp = subprocess.run(threadList, stdout=subprocess.PIPE, check=True)
# 		stdout = cp.stdout.decode("utf-8")
# 		print("SHARP running with test image:", ts.name, "on", t, "threads")
# 		print(stdout)

# 		# Find finalScoreRE in stdout
# 		scoreIter = re.finditer(execTimeRE, stdout)
# 		for match in scoreIter:
# 			scoreString = re.sub('\[Thread \d+\] Final score: ', '', match.group())[1:-1]
# 			score = scoreString.split(', ')
			
# 			#for i in range(len(score)):
# 				#statistics[ts.name][i*thetaStep] = score[i]
				
# 		plotExecutionTime(ts, list(statistics[ts.name].keys()), list(statistics[ts.name].values()))

# 	print(statistics)


def buildParser():
	parser = argparse.ArgumentParser(description='A scalability test driver for SHARP algortihm')

	refShapeDesc = 'Path of reference images (Default: working directory)'
	testShapeDesc = 'Path of test images'
	maxThreadsDesc = 'Max number of threads to test SHARP with. Must be a power of 2.'
	shapeSizeDesc = 'Test shape width in pixels'
	minThetaDesc = 'Minimum angle to consider for shape rotation in degrees'
	maxThetaDesc = 'Maximum angle to consider for shape rotation in degrees'
	thetaStepDesc = 'Angular distance between two consecutive angles'
	lenThreshDesc = 'Minimum accepted length of a shape-tangent segment'

	parser.add_argument('--ref-shapes', '-r', dest='refShapes', type=str, help=refShapeDesc)
	parser.add_argument('--test-shapes', '-t', dest='testShapes', type=str, help=testShapeDesc)
	parser.add_argument('--max-threads', dest='maxThreads', type=int, help=maxThreadsDesc)
	parser.add_argument('--shape-size', dest='shapeSize', default=256, type=int, help=shapeSizeDesc)
	parser.add_argument('--min-theta', dest='minTheta', default=0.0, type=float, help=minThetaDesc)
	parser.add_argument('--max-theta', dest='maxTheta', default=180.0, type=float, help=maxThetaDesc)
	parser.add_argument('--theta-step', dest='thetaStep', default=5, type=int, help=thetaStepDesc)
	parser.add_argument('--length-thresh', dest='lenThresh', default=2.0, type=float, help=lenThreshDesc)

	return parser

def main():
	parser = buildParser()
	args = parser.parse_args()

	runScalabilityTest(args.refShapes, args.testShapes, args.maxThreads, args.shapeSize, args.minTheta, args.maxTheta, args.thetaStep, args.lenThresh)

if __name__ == "__main__":
    main()
