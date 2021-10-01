from flask import Flask,render_template,request
from flask_cors import cross_origin
from detection import mobilenet, Decode

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        image = request.json['image']
        img = Decode(image).copy()
        #img = cv2.resize(img, (480, 320),interpolation=cv2.INTER_NEAREST)
        mobilenet(img)
        return render_template('index.html')
    return render_template('index.html')



if __name__=="__main__":
    app.run(host="0.0.0.0",port="5000")

# Author: Swati Sinha & Maitry Sinha
# Date : 2-09-2021