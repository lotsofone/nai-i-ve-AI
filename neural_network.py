from sklearn.neural_network import MLPClassifier

def neural(data, typ):
	mlp = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
	mlp.fit(data, typ)
	return mlp
