import sampling_methods
import numpy as np

#create random data matrix 10 x 6-dim datapoint
X = np.random.rand(10,6)

#sample a subset of 4 datapoints using kDPP
S1 = sampling_methods.kDPPGreedySample(X,4)

print(S1)

#randomly distribute 10 datapoints to 2 classes 
classes = np.random.randint(0,2,10)

#sample a subset of 4 datapoints 2+2 using PartitionDPP
S2 = sampling_methods.PartitionDPPGreedySample(X,[2,2],classes)


print(S2)