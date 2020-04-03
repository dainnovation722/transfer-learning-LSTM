from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras import initializers,regularizers

	
class regressor():

	def __init__(self, input_shape, gpu, pre_model=None):
		self.model = self.build_model(input_shape, gpu, pre_model=pre_model)
		self.hist = None
	
	def build_model(self, input_shape:tuple, gpu, pre_model=None, verbose=True) -> list:			
		if gpu:
			from keras.layers import CuDNNLSTM as LSTM
		else:
			from keras.layers import LSTM	 

		# construct the model
		input_layer = Input(input_shape)

		dense = TimeDistributed(
				Dense(
					50, 
					kernel_regularizer=regularizers.l2(0.01),
					kernel_initializer=initializers.glorot_uniform(seed=0), 
					bias_initializer=initializers.Zeros()
					)
								)(input_layer)

		lstm1 = LSTM(
					25, 
					return_sequences=True, 
					kernel_regularizer=regularizers.l2(0.01),
					kernel_initializer=initializers.glorot_uniform(seed=0),
					recurrent_initializer=initializers.Orthogonal(seed=0), 
					bias_initializer=initializers.Zeros()
					)(dense)
		lstm1 = BatchNormalization()(lstm1)

		lstm2 = LSTM(
					50, 
					return_sequences=True,
					kernel_regularizer=regularizers.l2(0.01),
					kernel_initializer=initializers.glorot_uniform(seed=0),
					recurrent_initializer=initializers.Orthogonal(seed=0),
					bias_initializer=initializers.Zeros()
					)(lstm1)
		lstm2 = BatchNormalization()(lstm2)

		lstm3 = LSTM(
					25, 
					return_sequences=False, 
					kernel_regularizer=regularizers.l2(0.01),
					kernel_initializer=initializers.glorot_uniform(seed=0),
					recurrent_initializer=initializers.Orthogonal(seed=0),
					bias_initializer=initializers.Zeros()
					)(lstm2)
		lstm3 = BatchNormalization()(lstm3)

		output_layer = Dense(
					1, 
					activation='sigmoid', 
					kernel_regularizer=regularizers.l2(0.01),
					kernel_initializer=initializers.glorot_uniform(seed=0), 
					bias_initializer=initializers.Zeros()
						)(lstm3)

		model = Model(inputs=input_layer, outputs=output_layer)

		# transfer weights from pre-trained model
		if pre_model:
			for i in range(2,len(model.layers)-1): 
				model.layers[i].set_weights(pre_model.layers[i].get_weights())

		model.compile(optimizer = Adam(), loss='mse', metrics=['accuracy'])
		if verbose: print(model.summary())

		return model
		

	def fit(self, X_train, y_train, X_valid, y_valid, nb_batch, nb_epochs, callbacks, verbose):
		mini_batch_size = X_train.shape[0]//nb_batch
		self.hist = self.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,\
			verbose=verbose, validation_data=(X_valid, y_valid), callbacks=callbacks)
	
	def predict(self, X_test):
		prediction = self.model.predict(X_test)
		return prediction

	