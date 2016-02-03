require 'nn'
require 'utils'
require 'image'
require 'optim'
require 'DistanceRatioCriterion'
require 'cudnn'
require 'cutorch'
require 'cunn'

-- number of threads
torch.setnumthreads(13)

-- read training data, save mu and sigma & normalize
name = 'notredame'
traind = read_brown_data(name)
stats = get_stats(traind)
print(stats)
torch.save('stats.'..name..'.t7',stats)
norm_data(traind,stats)
print("==> read the dataset")

-- generate random triplets for training data
num_triplets = 1280000
training_triplets = generate_triplets(traind, num_triplets)
print("==> created the tests")

-- setup the CNN
model1 = nn.Sequential() 
model1:add(cudnn.SpatialConvolution(1, 32, 7, 7))
model1:add(cudnn.Tanh(true))
model1:add(cudnn.SpatialMaxPooling(2,2,2,2)) 
model1:add(cudnn.SpatialConvolution(32, 64, 6, 6))
model1:add(cudnn.Tanh(true))
model1:add(nn.View(64*8*8))
model1:add(nn.Linear(64*8*8, 256))
model1:add(cudnn.Tanh(true))

--  clone the other two networks in the triplet
model2 = model1:clone('weight', 'bias','gradWeight','gradBias')
model3 = model1:clone('weight', 'bias','gradWeight','gradBias')

-- add them to a parallel table
prl = nn.ParallelTable()
prl:add(model1)
prl:add(model2)
prl:add(model3)
prl:cuda()

mlp= nn.Sequential()
mlp:add(prl)

-- get feature distances 
cc = nn.ConcatTable()

-- feats 1 with 2 
cnn_left = nn.Sequential()
cnnpos_dist = nn.ConcatTable()
cnnpos_dist:add(nn.SelectTable(1))
cnnpos_dist:add(nn.SelectTable(2))
cnn_left:add(cnnpos_dist)
cnn_left:add(nn.PairwiseDistance(2))
cnn_left:add(nn.View(128,1))
cnn_left:cuda()
cc:add(cnn_left)

-- feats 2 with 3 
cnn_left2 = nn.Sequential()
cnnpos_dist2 = nn.ConcatTable()
cnnpos_dist2:add(nn.SelectTable(2))
cnnpos_dist2:add(nn.SelectTable(3))
cnn_left2:add(cnnpos_dist2)
cnn_left2:add(nn.PairwiseDistance(2))
cnn_left2:add(nn.View(128,1))
cnn_left2:cuda()
cc:add(cnn_left2)

-- feats 1 with 3 
cnn_right = nn.Sequential()
cnnneg_dist = nn.ConcatTable()
cnnneg_dist:add(nn.SelectTable(1))
cnnneg_dist:add(nn.SelectTable(3))
cnn_right:add(cnnneg_dist)
cnn_right:add(nn.PairwiseDistance(2))
cnn_right:add(nn.View(128,1))
cnn_right:cuda()
cc:add(cnn_right)
cc:cuda()

mlp:add(cc)

last_layer = nn.ConcatTable()

-- select min negative distance inside the triplet
mined_neg = nn.Sequential()
mining_layer = nn.ConcatTable()
mining_layer:add(nn.SelectTable(1))
mining_layer:add(nn.SelectTable(2))
mined_neg:add(mining_layer)
mined_neg:add(nn.JoinTable(2))
mined_neg:add(nn.Min(2))
mined_neg:add(nn.View(128,1))
last_layer:add(mined_neg)
-- add positive distance
pos_layer = nn.Sequential()
pos_layer:add(nn.SelectTable(3))
pos_layer:add(nn.View(128,1))
last_layer:add(pos_layer)

mlp:add(last_layer)

mlp:add(nn.JoinTable(2))
mlp:cuda()

-- setup the criterion: ratio of min-negative to positive
epoch = 1
crit=nn.DistanceRatioCriterion():cuda()

batch_size = 128
x=torch.zeros(batch_size,1,32,32):cuda()
y=torch.zeros(batch_size,1,32,32):cuda()
z=torch.zeros(batch_size,1,32,32):cuda()

-- optim parameters
optimState = {
  learningRate = 0.1,
  weightDecay = 1e-4,
  momentum = 0.9,
  learningRateDecay = 1e-6
}

parameters, gradParameters = mlp:getParameters()

-- main training loop
for epoch=epoch,1000 do
   Gerr = 0
   shuffle = torch.randperm(num_triplets)   
   nbatches = num_triplets/batch_size

   for k=1,nbatches-1 do
      xlua.progress(k+1, nbatches)

      s = shuffle[{ {k*batch_size,k*batch_size+batch_size} }]
      for i=1,batch_size do 
      	 x[i] = traind.patches32[training_triplets[s[i]][1]]
      	 y[i] = traind.patches32[training_triplets[s[i]][2]]
	 z[i] = traind.patches32[training_triplets[s[i]][3]]
      end

      local feval = function(f)
	 if f ~= parameters then parameters:copy(f) end
	 gradParameters:zero()
	 inputs = {x,y,z}
	 local outputs = mlp:forward(inputs)
	 local f = crit:forward(outputs, 1)
	 Gerr = Gerr+f
	 local df_do = crit:backward(outputs)
	 mlp:backward(inputs, df_do)
	 return f,gradParameters
      end
      optim.sgd(feval, parameters, optimState)


   end

   print('==> epoch '..epoch)
   print(Gerr/nbatches)
   print('')

   torch.save(name..'_pnnet_full_epoch'.. epoch .. '.t7',mlp)
end

