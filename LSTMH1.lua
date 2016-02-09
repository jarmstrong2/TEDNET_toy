-- adapted from: wojciechz/learning_to_execute on github
--
local LSTMH1 = {}

-- Creates one timestep of one LSTM
function LSTMH1.lstm(inputSize, hiddenSize)
    local x = nn.Identity()()
    local w = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(inputSize, hiddenSize)(x)
        -- transforms window
        local w2h            = nn.Linear(32, hiddenSize)(w)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(hiddenSize, hiddenSize)(prev_h)
        return nn.CAddTable()({i2h, w2h, h2h})
    end
    function new_input_sum_bias()
        -- transforms input
        local bn1 = nn.Linear(inputSize, hiddenSize)
        bn1.bias:fill(4)
        local i2h            = bn1((x))
        -- transforms window
        local bn2 = nn.Linear(32, hiddenSize)
        bn2.bias:fill(4)
        local w2h            = bn2((w))
        -- transforms previous timestep's output
        local hh = nn.Linear(hiddenSize, hiddenSize)
        hh.bias:fill(4)
        local h2h            = hh(prev_h)
        return nn.CAddTable()({i2h, w2h, h2h})
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

    return nn.gModule({x, w, prev_c, prev_h}, {next_c, next_h})
end

return LSTMH1

