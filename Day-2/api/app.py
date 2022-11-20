from flask import Flask,request
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
	return "<p>Hello, World!</p>"

model_path = '/models/svm_gamma=0.001_C=0.2.joblib'

@app.route("/predict",methods=['POST'])
def predict_digit():
	image = request.json['image']
	model_path = request.json['model_name']
	model = load(model_path)
	print('done......loading')
	return model.predict([image])[0]

if __name__=="__main__":
	app.run(debug=True, host='0.0.0.0',port=5000)
