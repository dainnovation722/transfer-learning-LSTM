from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import CuDNNLSTM as LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam


class regressor():

	def __init__(self, input_shape, pre_model=None):
		self.model = self.build_model(input_shape, pre_model=pre_model)
		self.hist = None

	def build_model(self, input_shape:tuple, pre_model=None, verbose=True) -> list:
		'''学習モデルを構築'''
		input_layer = Input(input_shape)

		dense = TimeDistributed(Dense(50))(input_layer)

		lstm1 = LSTM(25, return_sequences=True)(dense)
		lstm1 = BatchNormalization()(lstm1)

		lstm2 = LSTM(50, return_sequences=True)(lstm1)
		lstm2 = BatchNormalization()(lstm2)

		lstm3 = LSTM(25, return_sequences=False)(lstm2)
		lstm3 = BatchNormalization()(lstm3)

		output_layer = Dense(1, activation='sigmoid')(lstm3)

		model = Model(inputs=input_layer, outputs=output_layer)

		if pre_model:
			for i in range(2,len(model.layers)): 
				model.layers[i].set_weights(pre_model.layers[i].get_weights())

		model.compile(loss='mse', optimizer = Adam(), metrics=['accuracy'])
		if verbose: print(model.summary())
		return model
		

	def fit(self, X_train, y_train, X_valid, y_valid, nb_batch, nb_epochs, callbacks, verbose):
		mini_batch_size = X_train.shape[0]//nb_batch
		self.hist = self.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,\
			verbose=verbose, validation_data=(X_valid, y_valid), callbacks=callbacks)
	
	def predict(self, X_test):
		prediction = self.model.predict(X_test)
		return prediction

	