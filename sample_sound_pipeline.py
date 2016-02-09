import argparse
from os import listdir
from os.path import isfile, join, isdir
import re
import subprocess

parser = argparse.ArgumentParser()

#optional arguments for torch 
parser.add_argument("-inputSize", default='61', help="number of input dimension")
parser.add_argument("-hiddenSize", default='800', help="number of hidden units in lstms")
parser.add_argument("-dimSize", default='2', help="number of hidden units in lstms")
parser.add_argument("-maxlen", default='500', help="max sequence length")
parser.add_argument("-numMixture", default='5', help="number of mixture components in output layer")
parser.add_argument("-modelFilename", default="relumodel.t7", help="model filename")
parser.add_argument("-testString", default="which was, wow, somebodys screwed!", help="string for testing")
parser.add_argument("-straightScale", default='0.7', help="scaling components for synthesis")

#optional arguments for STRAIGHT
parser.add_argument("-straightPath", default='../TEDNET_toy/strght.mat', help="path to matlab file for STRAIGHT")
parser.add_argument("-wavePath", default='../TEDNET_toy/sampleSound.wav', help="path to file to save sample sound from STRAIGHT")
parser.add_argument("-straightDir", default='../STRAIGHTV40_007d/', help="path to STRAIGHT library")

args = parser.parse_args()

# get STRAIGHT sample vector
command = 'th soundMaker3.lua '
command += '-inputSize ' + args.inputSize + ' '
command += '-hiddenSize ' + args.hiddenSize + ' '
command += '-dimSize ' + args.dimSize + ' '
command += '-maxlen ' + args.maxlen + ' '
command += '-numMixture ' + args.numMixture + ' '
command += '-modelFilename ' + args.modelFilename + ' '
command += '-testString ' + '"' + args.testString + '"' + ' '
command += '-straightScale ' + args.straightScale

subprocess.call(command, shell=True)

command = 'matlab -r "cd(\'' + args.straightDir + '\'); getSound61('
command += "'" + args.straightPath + "',"
command += "'" + args.wavePath + "'"
command += '); quit;"'

print(command)
subprocess.call(command, shell=True)

# reset terminal as matlab is doing something strange to STDOUT
subprocess.call('reset', shell=True)
