require 'getbatch'
require 'gnuplot'

function getXBatch(output)
                local pi_t, mu_t, u_t = unpack(output_y[t])
                pi_t = pi_t:clone():float():reshape(opt.batchSize, opt.numMixture)
                mu_t = mu_t:clone():float():reshape(opt.batchSize, opt.numMixture, opt.inputSize)
                local maxPi, indMaxPi = torch.max((pi_t:float()),2)
                --local chosenPi = torch.multinomial(torch.exp(pi_t:float()), 1):squeeze()

                chosenPi = indMaxPi:squeeze()
                mu_t:resize(opt.numMixture, 1, opt.inputSize) 
                mu_t = mu_t[chosenPi]

                return mu_t:double()
            end



-- get training dataset
if opt.useLargeDataset == true then
    STRAIGHTdata = torch.load('toy_set_large.t7')
else
    STRAIGHTdata = torch.load('toy_set.t7')
end
dataSize = #STRAIGHTdata

print('uploaded training data')

-- get validation dataset
if opt.useLargeDataset == true then
    valSTRAIGHTdata = torch.load('toy_set_large.t7')
else
    valSTRAIGHTdata = torch.load('toy_set.t7')
end
valdataSize = #valSTRAIGHTdata

print('uploaded validation data')

print('start training')

--params:uniform(-0.08, 0.08)
sampleSize = opt.batchSize
numberOfPasses = opt.numPasses

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(sampleSize, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()
initstate_h4_c = initstate_h1_c:clone()
initstate_h4_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()
dfinalstate_h3_c = initstate_h1_c:clone()
dfinalstate_h3_h = initstate_h1_c:clone()
dfinalstate_h4_c = initstate_h1_c:clone()
dfinalstate_h4_h = initstate_h1_c:clone()
initkappa = torch.randn(sampleSize,10)
dinitkappa = torch.zeros(sampleSize,10)

count = 1

batchCount = nil

function getEps()
   eps = torch.eye(opt.inputSize,opt.inputSize)*1e-4
   eps:resize(1,opt.inputSize,opt.inputSize)
   fulleps = eps:clone()
   for i = 2, opt.batchSize do
       fulleps = torch.cat(fulleps,eps,1)
    end
    return fulleps:cuda()
end

function getInitW(cuMat)
    cuMatClone = cuMat:clone()
    return torch.zero(cuMatClone[{{},{1},{}}]:squeeze(2))
end

function getValLoss()
    local valnumberOfPasses = opt.numPasses
    local valcount = 1
    local valsampleSize = opt.batchSize
    local loss = 0
    local elems = 0
    local eps = getEps()
    -- add for loop to increase mini-batch size
    for i=1, valnumberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, cmaskMat, elementCount, valcount
        = getBatch(valcount, valSTRAIGHTdata, valsampleSize)
        ------------------------------------------------------------

  --      if valcount > 100 then
            valcount = 1
    --    end

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        local lstm_c_h4 = {[0]=initstate_h4_c} -- internal cell states of LSTM
        local lstm_h_h4 = {[0]=initstate_h4_h} -- output values of LSTM        

        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- forward
        
        for t = 1, maxLen - 1 do
            -- set training mode off
            --clones.rnn_core[t]:evaluate()
            clones.rnn_core[t]:training()

            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze(3)
            if opt.feedPrediction == true then 
                if t > 1 then
                    x_in = getXBatch(output_y[t-1])
                else
                    x_in:fill(0.)
                end
            end

            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze(3)

            -- model 
            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))
       
            pi, mu, u = unpack(output_y[t])

            --loss = clones.criterion[t]:forward({pi:float(), mu:float(), u:float(),
            --    cmaskMat[{{},{},{t}}]:float(), x_target:float()}):sum() + loss  
            loss = clones.criterion[t]:forward({pi:cuda(), mu:cuda(), u:cuda(),
                cmaskMat[{{},{},{t}}]:cuda(), x_target:cuda(), eps}):sum() + loss      

            -- get the window attention and model output
            function getX(output, idx)
                local pi_t, mu_t, u_t = unpack(output_y[t])
                pi_t = pi_t:clone():float():reshape(opt.batchSize, opt.numMixture)[idx] 
                mu_t = mu_t:clone():float():reshape(opt.batchSize, opt.numMixture, opt.inputSize)[idx] 
                local maxPi, indMaxPi = torch.max(torch.exp(pi_t:float()),1)
                local chosenPi = torch.multinomial(torch.exp(pi_t:float()), 1):squeeze()

                chosenPi = indMaxPi:squeeze()
                mu_t:resize(opt.numMixture, 1, opt.inputSize) 
                mu_t = mu_t[chosenPi]

                return mu_t:double()
            end

            local x = getX(output_y[t], 1)
            if t == 1 then
                phiMat = _:clone():double()
                straightMat = x
            elseif t < 300 then
                phiMat = torch.cat(phiMat, _:clone():double(), 2)
                straightMat = torch.cat(straightMat, x, 1)
            end

        end

        loss = loss/(valsampleSize * valnumberOfPasses)
        pi = nil
        mu = nil 
        u = nil
        --maxLen = nil
        --strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        lstm_c_h4 = nil -- internal cell states of LSTM
        lstm_h_h4 = nil -- output values of LSTM
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local loss = 0
    local elems = 0
    local eps = getEps()
    
    -- add for loop to increase mini-batch size
    for i=1, numberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, cmaskMat, elementCount, count
        = getBatch(count, STRAIGHTdata, sampleSize)
        ------------------------------------------------------------
        if count > (#STRAIGHTdata-opt.batchSize) then
            count = 1
        end

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        local lstm_c_h4 = {[0]=initstate_h4_c} -- internal cell states of LSTM
        local lstm_h_h4 = {[0]=initstate_h4_h} -- output values of LSTM
        
        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        local input_crit = {}
        
        -- FORWARD
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze(3)

            -- model 
            -- turn on training mode
            clones.rnn_core[t]:training()

            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t], lstm_c_h4[t], lstm_h_h4[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]}))
       
            pi, mu, u = unpack(output_y[t])

            --input_crit[t] = {pi:float(), mu:float(), u:float(),
            --cmaskMat[{{},{},{t}}]:float(), x_target:float()}

            input_crit[t] = {pi:cuda(), mu:cuda(), u:cuda(),
            cmaskMat[{{},{},{t}}]:cuda(), x_target:cuda(), eps}

            loss = clones.criterion[t]:forward(input_crit[t]):sum() + loss 
        end

        loss = loss/(sampleSize*numberOfPasses)
        --print('current pass ',loss)        
        elems = (elementCount - sampleSize) + elems
        
        -- BACKWARD
                
        local dlstm_c_h1 = dfinalstate_h1_c
        local dlstm_h_h1 = dfinalstate_h1_h
        local dlstm_c_h2 = dfinalstate_h2_c
        local dlstm_h_h2 = dfinalstate_h2_h
        local dlstm_c_h3 = dfinalstate_h3_c
        local dlstm_h_h3 = dfinalstate_h3_h
        local dlstm_c_h4 = dfinalstate_h4_c
        local dlstm_h_h4 = dfinalstate_h4_h
        
        local dh1_w = torch.zeros(sampleSize, 32):cuda()
        local dkappa = torch.zeros(sampleSize, 10):cuda()
        
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{1,opt.inputSize},{t}}]:squeeze()
            local x_target = inputMat[{{},{1,opt.inputSize},{t+1}}]:squeeze()
            
            -- criterion
            --local grad_crit = clones.criterion[t]:backward(input_crit[t], torch.ones(sampleSize,1):float())            
            local grad_crit = clones.criterion[t]:backward(input_crit[t], torch.ones(sampleSize,1):cuda())

            d_pi, d_mu, d_u = unpack(grad_crit)

            -- model
            _x, _c, dkappa, dh1_w, dlstm_c_h1, dlstm_h_h1,
            dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3, dlstm_c_h4, dlstm_h_h4 = unpack(clones.rnn_core[t]:backward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1], lstm_c_h4[t-1], lstm_h_h4[t-1]},
                 {{d_pi:cuda(), d_mu:cuda(), d_u:cuda()}, dkappa, dh1_w, _, dlstm_c_h1, dlstm_h_h1, 
                  dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3, dlstm_c_h4, dlstm_h_h4}))
        end
        input_crit = nil
        grad_crit = nil
        d_pi = nil
        d_mu = nil
        d_u = nil
        dh2_w = nil
        dh2_h1 = nil
        dh3_w = nil
        dh3_h2 = nil
        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        lstm_c_h4 = nil -- internal cell states of LSTM
        lstm_h_h4 = nil -- output values of LSTM
        dlstm_c_h1 = nil -- internal cell states of LSTM
        dlstm_h_h1 = nil -- internal cell states of LSTM
        dlstm_c_h2 = nil -- internal cell states of LSTM
        dlstm_h_h2 = nil -- internal cell states of LSTM
        dlstm_c_h3 = nil -- internal cell states of LSTM
        dlstm_h_h3 = nil -- internal cell states of LSTM
        dlstm_c_h4 = nil -- internal cell states of LSTM
        dlstm_h_h4 = nil -- internal cell states of LSTM
        dkappaNext = nil
        dh1_w_next = nil
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        --collectgarbage()
    end
    
    grad_params:div(numberOfPasses)
    if opt.useAveragedLoss then
        grad_params:div(sampleSize)
    end
    
    -- clip gradient element-wise
    grad_params:clamp(-10, 10)
     
    return loss, grad_params
end

vallosses = nil
losses = nil 
local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-8}
local iterations = 800000
local minValLoss = 1/0
for i = 1, iterations do
    batchCount = i

    local _, loss = optim.adam(feval, params, optim_state)
    if i%2 == 0 then
        collectgarbage()
    end
    print(string.format("update param, loss = %6.8f, gradnorm = %6.4e", loss[1], grad_params:clone():norm()))
    if i % opt.evalEvery == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
        valLoss = getValLoss()

        print(string.format("validation loss = %6.8f", valLoss))
        if minValLoss > valLoss then
            minValLoss = valLoss
            torch.save(opt.modelFilename, model)
            print("------- Model Saved --------")
        end
        
        if valLoss ~= 1/0 then
        if not vallosses or (#vallosses)[1] >= 900 then
            vallosses = torch.Tensor(1)
            vallosses[1] = valLoss
            valiter = torch.Tensor(1)
            valiter[1] = i
        else
            vallossesaddition = torch.Tensor(1)
                    vallossesaddition[1] = valLoss
                    valiteraddition = torch.Tensor(1)
                    valiteraddition[1] = i
            vallosses = torch.cat(vallosses:float(), vallossesaddition:float(), 1)
            valiter = torch.cat(valiter, valiteraddition, 1)
        end
        end
        
        if loss[1] ~= 1/0 then
        if not losses or (#losses)[1] >= 900 then
                    losses = torch.Tensor(1)
                losses[1] = loss[1]
                    iter = torch.Tensor(1)
                    iter[1] = i
            else
                    lossesaddition = torch.Tensor(1,1)
                    lossesaddition[1] = loss[1]
                    iteraddition = torch.Tensor(1)
                    iteraddition[1] = i
                    losses = torch.cat(losses:float(), lossesaddition:float(),1)
                    iter = torch.cat(iter, iteraddition, 1)
            end
        end
        function plotCost()
            gnuplot.pngfigure(opt.lossImageFN)
            gnuplot.plot({iter, losses},{valiter, vallosses})
            gnuplot.plotflush()
        end
        if not pcall(plotCost) then
            print('had problem plot cost figure') 
        end
        print(phiMat:size())
        
        function plotAttn()
            gnuplot.pngfigure(opt.lossImageFN..'attn.png')
            gnuplot.imagesc(phiMat[1]:squeeze(2),'color')
            
            local xtics_str = ''
            xtics_str = xtics_str .. '"'..strs[1]:sub(1,1)..'" '.. 1
            for j = 2, #strs[1] do
                xtics_str = xtics_str .. ', "'..strs[1]:sub(j,j)..'" '.. j
            end
            gnuplot.raw('set xtics ('..xtics_str..')')
            gnuplot.plotflush()
        end
        
        if not pcall(plotAttn) then
            print('had problem plot attention figure') 
        end

        local matio = require 'matio'
        straightMat:mul(0.95)
        if opt.useLargeDataset == true then
            toy_std = toy_std or torch.load('toy_std_large.t7')
            toy_mean1 = toy_mean1 or torch.load('toy_mean_large.t7')
        else
            toy_std = toy_std or torch.load('toy_std.t7')
            toy_mean1 = toy_mean1 or torch.load('toy_mean.t7')
        end
        local newin = straightMat:float() * toy_std
        newin = newin:float() + toy_mean1
        matio.save(opt.modelFilename .. '_straight.mat', newin)
    end
end
