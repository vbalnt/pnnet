require 'cutorch'
require 'xlua'
require 'trepl'
require 'cunn'
require 'cudnn'
require 'image'
require 'nn'
require 'torch'


-- load default 128 out tanh-maxpooling network trained on liberty dataset
-- for more details http://phototour.cs.washington.edu/patches/default.htm
network = 'liberty'
local net = torch.load('pnnet-'..network..'.t7'):cuda()
print(net)

-- load mu and sigma 
stats = torch.load('stats-'..network..'.t7')

--download from https://github.com/vbalnt/UBC-Phototour-Patches-Torch
eval_data = 'notredame'

dataset = torch.load(eval_data..'.t7')
npatches =  (dataset.patches32:size(1))

print(npatches)

-- normalize data
patches32 = dataset.patches32:cuda()
patches32:add(-stats.mi):div(stats.sigma)

-- split the patches in batches to avoid memory problems
BatchSize = 128
Descrs = torch.CudaTensor(npatches,128)
DescrsSplit = Descrs:split(BatchSize)
for i,v in ipairs(patches32:split(BatchSize)) do
	DescrsSplit[i]:copy(net:forward(v))
end

-- test on the testing gt (100k pairs from Brown's dataset)
ntest = 100000
for i=1,ntest do
   l = dataset.gt100k[i]
   lbl = l[2]==l[5] and 1 or 0
   id1 = l[1]+1
   id2 = l[4]+1
   dl = Descrs[{ {id1},{} }]
   dr = Descrs[{ {id2},{} }]

   d = torch.dist(dl,dr)

   io.write(string.format("%d %.4f \n", lbl,d))
end
