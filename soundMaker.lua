require 'gnuplot'
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'getvocab'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'window'
require 'yHat'
local mixture = require 'mixtureGauss'
local model_utils=require 'model_utils'
require 'cunn'
require 'distributions'
local matio = require 'matio'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for Sound Sampling.')

cmd:option('-inputSize' , 61, 'number of input dimension')
cmd:option('-hiddenSize' ,800, 'number of hidden units in lstms')
cmd:option('-dimSize' , 2, 'number of hidden units in lstms')
cmd:option('-maxlen' , 500, 'max sequence length')
cmd:option('-numMixture' , 5, 'number of mixture components in output layer')
cmd:option('-modelFilename' , 'relumodel.t7', 'model filename')
cmd:option('-testString' , 'somebodys life is about to get terrible!', 'string for testing')
cmd:option('-straightScale' , '0.7', 'scaling components for synthesis')
cmd:option('-isCovarianceFull' , true, 'true if full covariance, o.w. diagonal covariance')

cmd:text()
opt = cmd:parse(arg)

function getX(output)

    pi_t, mu_t, u_t = unpack(output)     

    local maxPi, indMaxPi = torch.max(torch.exp(pi_t:float()),2)

    local chosenPi = torch.multinomial(torch.exp(pi_t:float()), 1):squeeze()

       chosenPi = indMaxPi:squeeze()
	mu_t:resize(opt.numMixture, 1, opt.inputSize) 

    mu_t = mu_t[chosenPi]

   return mu_t:double()
end

cuMat = getOneHotStrs({[1]=opt.testString})

model = torch.load(opt.modelFilename)
param, grad = model.rnn_core:getParameters()
-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(1, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()
initstate_h4_c = initstate_h1_c:clone()
initstate_h4_h = initstate_h1_c:clone()

-- initialize input
x = torch.zeros(1,opt.inputSize)

function getInitW(cuMat)
	cuMatClone = cuMat:clone()
    return torch.zero(cuMatClone[{{},{1},{}}]:squeeze(2))
end

-- initialize window to first char in all elements of the batch
local w = {[0]=getInitW(cuMat:cuda())}

local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
local lstm_c_h4 = {[0]=initstate_h3_c} -- internal cell states of LSTM
local lstm_h_h4 = {[0]=initstate_h3_h} -- output values of LSTM

local kappa_prev = {[0]=torch.zeros(1,10):cuda()}

local output_h1_w = {}
local input_h3_y = {}
local output_h3_y = {}

-- FORWARD

for t = 1, opt.maxlen - 1 do
    -- model 
    output_y, kappa_prev[t], w[t], phi, lstm_c_h1[t], lstm_h_h1[t],
    lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
	= unpack(model.rnn_core:forward({x:cuda(), cuMat:cuda(), 
         kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
         lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))

	-- perform op on x
	x = getX(output_y)
	if straightMat then
		straightMat = torch.cat(straightMat, x, 1)
	else
		straightMat = x
	end
	
	if t == 1 then
                phiMat = phi:double()
        else
                phiMat = torch.cat(phiMat, phi:double(), 1)
        end
end


straightMat:mul(opt.straightScale)

std = torch.load('toy_std_61_sing.t7')
mean = torch.load('toy_mean_61_sing.t7')
--rs_std = torch.expand(std, opt.maxlen -1, opt.inputSize)
--rs_mean = torch.expand(mean1, opt.maxlen-1, opt.inputSize)
newin = straightMat:float() * std
newin = newin:float() + mean
matio.save('strght.mat', newin)
gnuplot.pngfigure('attentionmat.png')
gnuplot.imagesc(phiMat:squeeze(2),'color')
gnuplot.plotflush()
