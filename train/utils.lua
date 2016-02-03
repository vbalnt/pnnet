-- read dataset in the wanted format
-- example here: http://vbalnt.io/notredame-torch.tar.gz
function read_brown_data(name)
   local d = torch.load('../'..name..'.t7') 
   d.patches32 = d.patches32:float()
   -- labels in data are zero-indexed
   d.labels:add(1)
   return d
end

-- get the stats
function get_stats(d)
   local mi = d.patches32:mean()
   local sigma = d.patches32:std()
   local stats = {}
   stats.mi = mi
   stats.sigma = sigma
   return stats
end

-- norm data based on stats
function norm_data(d,stats)
   d.patches32:add(-stats.mi):div(stats.sigma)
end

-- following functions taken from Elad Hoffer's
-- TripletNet https://github.com/eladhoffer
function ArrangeByLabel(traind)
   local numClasses = traind.labels:max()
   local Ordered = {}
   for i=1,traind.labels:size(1) do
      -- print(i)
      if Ordered[traind.labels[i]] == nil then
	 Ordered[traind.labels[i]] = {}
      end
      table.insert(Ordered[traind.labels[i]], i)
   end
   return Ordered
end


function generate_pairs(traind, num_pairs)
   local list = torch.IntTensor(num_pairs,3)
   local pairs = torch.IntTensor(num_pairs,3)

   local Ordered = ArrangeByLabel(traind)
   local nClasses = #Ordered
   for i=1, num_pairs do
      -- print(i)
      local c1 = math.random(nClasses)
      local c2 = math.random(nClasses)
      while c2 == c1 do
	 c2 = math.random(nClasses)
      end
      local n1 = math.random(#Ordered[c1])
      local n2 = math.random(#Ordered[c2])
      local n3 = math.random(#Ordered[c1])
      while n3 == n1 do
	 n3 = math.random(#Ordered[c1])
      end

      list[i][1] = Ordered[c1][n1]
      list[i][2] = Ordered[c2][n2]
      list[i][3] = Ordered[c1][n3]
      
      lbl = math.random(0,2)
      if ((lbl==0) or (lbl==1)) then
	 pairs[i][1] = list[i][1]
	 pairs[i][2] = list[i][2]
	 pairs[i][3] = -1
      else
	 pairs[i][1] = list[i][1]
	 pairs[i][2] = list[i][3]
	 pairs[i][3] = 1
      end
   end

   return pairs

end



function generate_triplets(traind, num_pairs)
   local list = torch.IntTensor(num_pairs,3)

   local Ordered = ArrangeByLabel(traind)
   local nClasses = #Ordered
   for i=1, num_pairs do
      -- print(i)
      local c1 = math.random(nClasses)
      local c2 = math.random(nClasses)
      while c2 == c1 do
	 c2 = math.random(nClasses)
      end
      local n1 = math.random(#Ordered[c1])
      local n2 = math.random(#Ordered[c2])
      local n3 = math.random(#Ordered[c1])
      while n3 == n1 do
	 n3 = math.random(#Ordered[c1])
      end
      list[i][1] = Ordered[c1][n1]
      list[i][2] = Ordered[c2][n2]
      list[i][3] = Ordered[c1][n3]
   end

   return list
end

