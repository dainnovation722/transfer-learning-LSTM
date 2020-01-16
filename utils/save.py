import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.rcParams["font.size"] = 13

def save_lr_curve(model):
	# 学習曲線の保存
	hist = model.hist
	plt.figure(figsize=(18,5))
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	return plt 
	
def save_plot(y_test_time, y_pred_test_time):
	# テストデータの予測結果プロットの保存
	plt.figure(figsize=(18,5))
	plt.plot([ i for i in range(1, 1+len(y_pred_test_time))], y_pred_test_time, 'r',label="predicted")
	plt.plot([ i for i in range(1, 1+len(y_test_time))], y_test_time, 'b',label="measured", lw=1, alpha=0.3)
	plt.ylim(0,1)
	plt.legend(loc="best")
	return plt