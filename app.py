import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('admit.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	University_Rating = int(request.form['University Rating'])
	SOP = float(request.form['SOP'])
	LOR = float(request.form['LOR'])
	CGPA = float(request.form['CGPA'])
	Research = int(request.form['Research'])
	
	#final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
	
	#predict = model.predict(final_features)
	
	#output = predict[0]
	input_data = {"GRE Score": GRE_Score,
              "TOEFL Score": TOEFL_Score,
              "University Rating": University_Rating,
              "SOP": SOP, 
              "LOR ": LOR,
              "CGPA": CGPA,
              "Research": Research}
	input_df = pd.DataFrame(input_data, index=[0])
	prediction = model.predict(input_df)[0]
        
	return render_template('admit.html', prediction_text='Your chance of admit is {:.2f}%'.format(prediction * 100))
	
if __name__ == "__main__":
	app.run(debug=True)