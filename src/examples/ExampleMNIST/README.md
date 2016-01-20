# Example MNIST

Here an MLP classifier is implemented to classify handwritten digits from the famous MNIST dataset. The code takes care of importing the raw data files provided on the MNIST website. A little-endian processor it is assumed. To use this code simply change the path to the dataset according to your environment.

## Results 

I tried various shallow configuration using different activation functions. Sincerely, I don't remember all the details for each case. Most probably, for the hidden layer I used the logistic sigmoid function or the hyper tangent. In all cases I used linear activations on the output layer and L2 loss.  

Best results:

* Single Hidden Layer (130 units): 5.7%
* Single Hidden Layer (100 units): 5.5%	
* Single Hidden Layer (175 units): 3.65%