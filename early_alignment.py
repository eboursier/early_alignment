import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

import numpy as np
import imageio
from tqdm import tqdm
import copy
import _pickle as pickle # _pickle is the newer updated version (cpickle) I believe, with improved C-backend
import argparse


class Net(torch.nn.Module):
    """
    1 hidden layer Relu network
    """
    def __init__(self, n_feature, n_hidden, n_output, init_scale=1, bias_hidden=True, initialisation='balanced', **kwargs):
        """
        n_feature: dimension of input
        n_hidden: number of hidden neurons
        n_output: dimension of output
        init_scale: all the weights are initialized ~ N(0, init_scale^2/m) where m is the input dimension of this layer
        bias_hidden: if True, use bias parameters in hidden layer. Use no bias otherwise
        bias_output: if True, use bias parameters in output layer. Use no bias otherwise
        intialisation: 'balanced', 'unbalanced' or 'dominated'
                            - balanced ensures ||w_j|| = |a_j|
                            - unbalanced ensures no relation and independently initialise gaussian weights
                            - dominated ensures |a_j| > ||w_j||
        """
        super(Net, self).__init__()
        self.init_scale = init_scale/np.sqrt(n_hidden) # normalisation by sqrt(m)
        self.initialisation_ = initialisation
        
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=bias_hidden)   # hidden layer with rescaled init

        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)   # output layer with rescaled init
        
        if initialisation=='balanced': # balanced initialisation
            torch.nn.init.normal_(self.hidden.weight.data, std=self.init_scale)
            if bias_hidden:
                torch.nn.init.normal_(self.hidden.bias.data, std=self.init_scale)
                neuron_norms = (self.hidden.weight.data.norm(dim=1).square()+self.hidden.bias.data.square()).sqrt()
            else:
                neuron_norms = (self.hidden.weight.data.norm(dim=1).square()).sqrt()
            self.predict.weight.data = 2*torch.bernoulli(0.5*torch.ones_like(self.predict.weight.data)) -1
            self.predict.weight.data *= neuron_norms
            
        if initialisation=='unbalanced':
            torch.nn.init.normal_(self.hidden.weight.data, std=self.init_scale)
            if bias_hidden:
                torch.nn.init.normal_(self.hidden.bias.data, std=self.init_scale)
            torch.nn.init.normal_(self.predict.weight.data, std=self.init_scale)
            
        if initialisation=='dominated':
            torch.nn.init.uniform_(self.hidden.weight.data, a=-self.init_scale, b=self.init_scale)
            self.predict.weight.data = 2*torch.bernoulli(0.5*torch.ones_like(self.predict.weight.data)) -1
            self.predict.weight.data *= self.init_scale
            if bias_hidden:
                torch.nn.init.uniform_(self.hidden.bias.data, a=-self.init_scale, b=self.init_scale)
                self.predict.weight.data *= np.sqrt(2)
            
        self.activation = kwargs.get('activation', torch.nn.ReLU()) # activation of hidden layer
        
        if kwargs.get('zero_output', False):
            # ensure that the estimated function is 0 at initialisation
            # useful when initialising in lazy regime
            half_n = int(n_hidden/2)
            self.hidden.weight.data[half_n:] = self.hidden.weight.data[:half_n]
            if bias_hidden:
                self.hidden.bias.data[half_n:] = self.hidden.bias.data[:half_n]
            self.predict.weight.data[0, half_n:] = -self.predict.weight.data[0, :half_n]
            

    def forward(self, z):
        z = self.activation(self.hidden(z))     
        z = self.predict(z)             # linear output
        return z


if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--data", nargs='?', const='load', default="stewart", type=str, help="3 points, stewart or load")
	parser.add_argument("-init", "--initialisation", nargs='?', const='unbalanced', default='balanced', help="balanced, unbalanced or dominated")
	parser.add_argument("--n_hidden", default=20000, type=int, help="width of network")
	parser.add_argument("--n_iterations", default=2000000, type=int, help="number of GD iterations")
	parser.add_argument("--init_scale", default=1e-3, type=float, help="init scale lambda")
	parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate for GD")
	parser.add_argument("-warm", "--warm_restart", nargs='?', const=True, default=False, type=bool, help="if True, start from last saved iteration state")
	parser.add_argument("-v", "--verbose", nargs='?', const=True, default=False, type=bool)


	args = parser.parse_args()
	params = vars(args)
	print("Running with")
	for arg in params:
		print('\t\t' + arg + ' : ' + str(params[arg]))

	##Data
	## no convergence example
	if params['data']=='3 points':
		x = torch.Tensor(np.array([-0.75,-0.5,0.125]).reshape(-1,1)) 
		y = torch.Tensor(np.array([1.1,0.1,0.8]).reshape(-1,1)) 

	## Generate Stewart et al. data
	elif params['data']=='stewart':
		if params['warm_restart']:
			raise ValueError('Warm restart is not possible with data = stewart')
		x_stew = np.array([-1, -0.7, -0.55, -0.4, 0, 0.5, 0.6, 0.7, 1])
		y_stew = np.array([0, 0.7, 0.5, 1, 0, 1, 0.6, 0.7, 0])
		x = torch.Tensor(np.random.uniform(-1, 1, 40).reshape(-1,1))
		y = torch.Tensor(np.interp(x, x_stew, y_stew))

	elif params['data']=='load':
		x = torch.load('saves/x.pth')
		y = torch.load('saves/y.pth')

	else:
		raise ValueError('Unavailable argument for --data.')

	#############################

	# torch can only train on Variable, so convert them to Variable
	x, y = Variable(x), Variable(y)

	if params['data']!='load':
		torch.save(x,'saves/x.pth')
		torch.save(y,'saves/y.pth')

	# init network
	net = Net(n_feature=1, n_output=1, bias_hidden=True, **params)     # define the network
	 
	optimizer = torch.optim.SGD(net.parameters(), lr=params['learning_rate']) #Gradient descent
	loss_func = torch.nn.MSELoss(reduction='mean')  # mean squared error

	n_samples = x.shape[0]

	# plot parameters
	iter_geom = 1.1 #saved frames correspond to steps t=\lceil k^{iter_geom} \rceil for all integers k 

	if params['warm_restart']:
		last_iter = np.loadtxt('saves/iters.txt', dtype="int")[-1]
		with open("saves/nets.pth","rb") as netfile:
			while True:
				try:
					net.load_state_dict(pickle.load(netfile))
				except EOFError:
					break
		if params['verbose']:
			print('Loaded state from iteration ' + str(last_iter))
		iterfile = open('saves/iters.txt','a')
		netfile = open("saves/nets.pth","ab") 
		lossfile = open('saves/losses.txt','a')

	else:
		last_iter = -1
		iterfile = open('saves/iters.txt','w')
		netfile =  open("saves/nets.pth","wb") 
		lossfile = open('saves/losses.txt','w')

	# train the network
	for it in tqdm(range(last_iter+1, params['n_iterations']), initial=last_iter+1, total=params['n_iterations'], desc="Training Network"):
		prediction = net(x)
		loss = loss_func(prediction, y) 
		if (it<2 or it==int(last_iter*iter_geom)+1): # save net weights
			pickle.dump(net.state_dict(), netfile)
			iterfile.write("%i\n" % it)
			lossfile.write(f"{loss.data.numpy()}\n")
			last_iter = it
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # descent step
		
	# save last iterate
	pickle.dump(net.state_dict(), netfile)
	iterfile.write("%i\n" % it)

	lossfile.close()
	iterfile.close()
	netfile.close()	
	if params['verbose']:	
		print(f'Experimental run completed.')
