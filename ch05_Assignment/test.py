# Question 4)
from flask import Flask
from flask import request
from flask import jsonify
import pickle

model_file = 'model1.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    
dv_file = 'dv.bin'
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('churn')

@app.route('/prediction', methods=["POST"])
def pred():
    customer = request.get_json()
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[0, 1]
    churn = y_pred >= .5
    result = {
        'churn_proba': float(y_pred),
        'churn': bool(churn) 
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9696')







