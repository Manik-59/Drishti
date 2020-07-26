from flask import *
import caption_predict

app=Flask(__name__)

@app.route('/')

def home():
	return render_template("index.html")

@app.route('/',methods=['POST'])

def accept():
	
	if request.method == 'POST':

		f = request.files['userfile']

		path = "./static/{}".format(f.filename)

		f.save(path)

		caption = caption_predict.caption_it(path)
		

	return render_template("index.html",image_caption=caption,path=path) 

if __name__=="__main__":
	app.run(debug=False,threaded=False)