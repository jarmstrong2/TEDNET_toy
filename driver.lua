local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-inputSize' , 35, 'number of input dimension')
cmd:option('-hiddenSize' , 500, 'number of hidden units in lstms')
cmd:option('-dimSize' , 2, 'dim size for U')
cmd:option('-lr' , 3e-3, 'learning rate')
cmd:option('-maxlen', 500, 'max sequence length')
cmd:option('-batchSize' , 2, 'mini batch size')
cmd:option('-numPasses' , 1, 'number of passes')
cmd:option('-isCovarianceFull' , true, 'true if full covariance, o.w. diagonal covariance')
cmd:option('-numMixture' , 20, 'number of mixture components in output layer') 
cmd:option('-lossImageFN' , 'plot_exlayer.png', 'filename for plot file')
cmd:option('-evalEvery' , 5, 'number of iterations to record training/validation losses ')
cmd:option('-modelFilename' , 'exlayer.t7', 'model filename')

cmd:text()
opt = cmd:parse(arg)

dofile('model.lua')
dofile('train.lua')
