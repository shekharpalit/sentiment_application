from flask import Flask, render_template, url_for, request
from sentiment import predict_sentiment


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        predect_n = predect_sentiment(comment)
    return render_template('result.html', predection = predict_n)

if __name__ == '__main__':
    app.run(debug = True)
