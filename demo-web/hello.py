from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/uploader', methods=['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('uploaded_file')
      
      return 'success'

@app.route('/results')
def results():
    return

if __name__ == '__main__':
    app.run(debug=True)
