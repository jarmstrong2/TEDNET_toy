-- adapted from: wojciechz/learning_to_execute on github

local LSTMHN = {}

-- Creates one timestep of one LSTM
function LSTMHN.lstm(inputSize, hiddenSize)
    local x = nn.Identity()()
    local w = nn.Identity()()
    local below_h = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.BatchNormalization(hiddenSize)(nn.Linear(inputSize, hiddenSize)(x))
        -- transforms window
        local w2h            = nn.BatchNormalization(hiddenSize)(nn.Linear(32, hiddenSize)(w))
        -- transforms hidden output from below current hidden layer
        local bh2h            = nn.BatchNormalization(hiddenSize)(nn.Linear(hiddenSize, hiddenSize)(below_h))
        -- transforms previous timestep's output
        local h2h            = nn.Linear(hiddenSize, hiddenSize)(prev_h)
        return nn.CAddTable()({i2h, w2h, bh2h, h2h})
    end
    function new_input_sum_bias()
        -- transforms input
        local bn1 = nn.BatchNormalization(hiddenSize)
        bn1.bias:fill(4)
        local i2h            = bn1(nn.Linear(inputSize, hiddenSize)(x))
        -- transforms window
        local bn2 = nn.BatchNormalization(hiddenSize)
        bn2.bias:fill(4)
        local w2h            = bn2(nn.Linear(32, hiddenSize)(w))
        -- transforms hidden output from below current hidden layer
        local bn3 = nn.BatchNormalization(hiddenSize)
        bn3.bias:fill(4)
        local bh2h            = bn3(nn.Linear(hiddenSize, hiddenSize)(below_h))

        -- transforms previous timestep's output
        local hh = nn.Linear(hiddenSize, hiddenSize)
        hh.bias:fill(4)
        local h2h            = hh(prev_h)
        return nn.CAddTable()({i2h, w2h, bh2h, h2h})
    end


    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum_bias())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({x, w, below_h, prev_c, prev_h}, {next_c, next_h})
end

return LSTMHN

