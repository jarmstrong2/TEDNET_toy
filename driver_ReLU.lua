local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-inputSize' , 61, 'number of input dimension')
cmd:option('-hiddenSize' , 600, 'number of hidden units in lstms')
cmd:option('-dimSize' , 2, 'dim size for U')
cmd:option('-lr' , 3e-3, 'learning rate')
cmd:option('-maxlen', 500, 'max sequence length')
cmd:option('-batchSize' , 4, 'mini batch size')
cmd:option('-numPasses' , 1, 'number of passes')
cmd:option('-useAveragedLoss' , false, 'divide the loss and its gradient by the batch size.')
cmd:option('-isCovarianceFull' , true, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 10, 'number of mixture components in output layer') 
cmd:option('-lossImageFN' , 'plot_reluexlayer_ronson.png', 'filename for plot file')
cmd:option('-evalEvery' , 5, 'number of iterations to record training/validation losses ')
cmd:option('-modelFilename' , 'reluexlayer_ronson.t7', 'model filename')
cmd:option('-reluSize', 400, 'number of ReLU units')

cmd:text()
opt = cmd:parse(arg)

print(opt)

dofile('model_ReLU.lua')
dofile('train_ReLU.lua')
