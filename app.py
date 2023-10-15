from flask import Flask, render_template, session, redirect,request, jsonify,url_for
from bson import ObjectId
from chat import get_response
import json
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from functools import wraps
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pymongo
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'

# Database
client=pymongo.MongoClient("mongodb+srv://SRISAILA:saila1410@cluster0.jnllygb.mongodb.net/?retryWrites=true&w=majority")
db=client.get_database('AgroSage')
collection = db["questions"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model.load_state_dict(checkpoint['model_state_dict'])
print("analyzing")

model_path = "my_modell.h5"
model1 = tf.keras.models.load_model(model_path, compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')
data = pd.read_csv("Dataset.csv", index_col='Disease')

filename = "XGBoost.pkl"

ffn="xgb_pipeline.pkl"
fertilizer_model=pickle.load(open(ffn, "rb"))
pricedata=pd.read_csv("cost15.csv")

indtoferti={0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}

crops=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

croptocost={'apple':'Apple','watermelon':'Water+Melon','mungbean':'Gram+Raw(Chholia)','banana':'Banana','blackgram': 'Alasande+Gram','coconut': 'Coconut','coffee' :'Coffee','grapes' :'Grapes','jute': 'Jute','maize':'Maize','mango ':'Mango','orange': 'Orange','papaya':'Papaya','pomegranate': 'Pomegranate','rice' :'Rice'}

soiltoind={'Black':0, 'Clayey':1,'Loamy':2, 'Red':3,'Sandy':4}

croptoind={'Barley':0,'Cotton':1,'Ground Nuts':2, 'Maize':3, 'Millets':4,'Oil seeds':5,'Paddy':6,'Pulses':7, 'Sugarcane':8, 'Tobacco':9, 'Wheat':10}

def toxicityPredict(st):
    input_ids = tokenizer.encode(st, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        model.eval()
        input_ids = input_ids.to(device) 
        outputs = model(input_ids)
    logits = outputs.logits
    threshold = 0.5 
    predictions = (torch.sigmoid(logits) > threshold).cpu().numpy()
    tox=0
    predicted_labels = [label for label, prediction in zip(labels, predictions[0]) if prediction]
    output_string = f"Input Text: '{st}'\nPredicted Labels: {', '.join(predicted_labels)}"
    print(output_string)
    if(len(predicted_labels)>1):
        tox=1
    else:
        tox=0
    return tox

def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap

# Routes
from user import routes

@app.route('/register/')
def register():
    return render_template('register.html')

@app.route('/dashboard/')
@login_required
def dashboard():
  return render_template('dashboard.html')

@app.route("/input", methods=["GET", "POST"])
def input():
    if request.method == "POST":
        with open("Rainfall.json", "r") as file:
            data = json.load(file)
        user_input=request.form["states"]+"_"+request.form["districts"]
        season=request.form["seasons"]
        print(data[user_input][season])
        with open("temphum.json","r") as file1:
            data1=json.load(file1)
        
        states=request.form["states"]
        with open("NPK.json","r") as file2:
            data2=json.load(file2)
    
        predictions=[
            {'Temperature' : data1[states]["Temperature"]},
            {'Humidity':data1[states]["Humidity"]},
            {'Rainfall':data[user_input][season]},
            {'ph':data2[states]["pH"]},
            {'Nitrogen content':data2[states]['N']},
            {'Phosphorous content':data2[states]['P']},
            {'Potassium Content':data2[states]['K']}
        ]
    return render_template('suggestions.html', predictions=predictions)

@app.route("/base")
def index():
    return render_template("base.html")

@app.route("/")    
def home():
    return render_template("index.html")  

@app.route("/about")
def about():
    return render_template("aboutus.html")

@app.route("/predictpage")
def predict():
    return render_template("predictpage.html")

@app.route("/fertilizer")
def predfer():
    return render_template("fertilizer.html")

@app.route("/fertipredict",methods = ['POST'])
def fertipredict():
    msg=""""""
    if request.method=="POST":
        fdata=[[]]
        fdata[0].append(int(request.form["temperature"]))
        fdata[0].append(int(request.form["humidity"]))
        fdata[0].append(int(request.form["Moisture"]))
        fdata[0].append(soiltoind[request.form["soils"]])
        fdata[0].append(croptoind[request.form["crops"]])
        fdata[0].append(int(request.form["N"]))
        fdata[0].append(int(request.form["K"]))
        fdata[0].append(int(request.form["P"]))
        fdata=np.array(fdata)
        pred=fertilizer_model.predict(fdata)[0]
        msg+="<b>"+"suggested fertilizer is "+str(indtoferti[pred])+"</b>"
        return msg
@app.route("/exactpredict",methods = ['POST'])
def predictpredicts():
    loaded_model = pickle.load(open(filename, "rb"))
    message = """"""
    if request.method == 'POST':
        rdata={}
        rdata["N"]=[int(request.form["N"])]
        rdata["P"]=[int(request.form["P"])]
        rdata["K"]=[int(request.form["K"])]
        rdata["temperature"]=[float(request.form["temperature"])]
        rdata["humidity"]=[float(request.form["humidity"])]
        rdata["ph"]=[float(request.form["ph"])]
        rdata["rainfall"]=[float(request.form["rainfall"])]
        state=request.form["states"]
        district=request.form["districts"]
        district=district[0]+district[1:].lower()
        df=pd.DataFrame.from_dict(rdata)
        predprob=loaded_model.predict_proba(df)[0]
        top3=np.flip(np.argsort(predprob))[:3]
        predprob=sorted(predprob,reverse=True)
        best_crop=crops[top3[0]]
        crop2=crops[top3[1]]
        crop3=crops[top3[2]]
        ind=0
        for prob in predprob[:3]:
            if prob>0.1:
                prob=prob*100
                message+="<b>"+"Best crop:"+str(crops[top3[ind]])+"</b>"+str(prob)[:4]+"%"+"<br>"
            else:
                message+="Average crops:"+str(crops[top3[ind]])+"  "
            ind+=1
        for ci in [best_crop]:
            if ci in croptocost.keys():
                cicrop=croptocost[ci]
                if pricedata[pricedata["district"]==district].shape[0]>0 and pricedata[pricedata["commodity_name"]==cicrop].shape[0]>0:
                    message+="<br>"+"<b>"+"District:"+district+"</b>"+"<br>"
                    datadest=pricedata[(pricedata["commodity_name"]==cicrop)&(pricedata["district"]==district)]
                    datadest["date"].str[-2:]
                    yearset=set(datadest["date"].str[-2:])
                    for y in yearset:
                        datay=datadest[datadest["date"].str[-2:]==y]
                        monthset=set(datay["date"].str[:2])
                        message+="20"+str(y)+":<br>"
                        for m in monthset:
                            cost=datay[datay["date"].str[:2]==m]
                            if m[1]=="/":
                                m=m[:1]
                            ma=cost["modal_price"].max()
                            mi=cost["modal_price"].min()
                            av=cost["modal_price"].mean()
                            if ma<500 and mi<500 and av<500:
                                message+=str(m)+"th month prices"+"::"+" Max:"+str(ma)+" Mean:"+str(av)[:4]+" Min:"+str(mi)+"<br>"
                                print(ma,mi,av)


          
        return message
@app.post("/predicts")
def predicts():
    text = request.get_json().get("message")
    response=get_response(text)
    message={"answer" :response}
    return jsonify(message)

@app.route('/help')
def help():
    questions = collection.find()
    return render_template('help.html', questions=questions)

@app.route('/ask', methods=['POST'])
def ask_question():
    question_text = request.form['question']
    res_toxic = toxicityPredict(question_text)
    
    if res_toxic == 0:
        user_name = session['user']['name'] if 'user' in session else None  # Get the user's name from the session
        profession=session['user']['profession'] if 'user' in session else None
    # Store the question along with the username in the database
        collection.insert_one({'question': question_text, 'user_name': user_name, 'profession':profession,'answers': []})
        session['message'] = 'Your Question is Posted!'
    else:
        session['message'] = 'Toxic Question Detected!'
    return redirect(url_for('help'))

@app.route('/answer/<question_id>', methods=['POST'])
def answer_question(question_id):
    answer_text = request.form['answer']
    res_tox = toxicityPredict(answer_text)
    if res_tox == 0:
        user_name = session['user']['name'] if 'user' in session else None  # Get the user's name from the session
        profession=session['user']['profession'] if 'user' in session else None
        upvotes=0
        collection.update_one({'_id': ObjectId(question_id)}, {'$push': {'answers': {'text': answer_text, 'user_name': user_name,'profession':profession,'upvotes':upvotes}}})
        session['message'] = 'Your Answer is posted!'
    else:
        session['message'] = 'Toxic Answer Detected!'
   
    return redirect(url_for('help'))

@app.route('/upvote/<question_id>/<answer_index>')
def upvote_answer(question_id, answer_index):
    # Convert answer_index to an integer
    answer_index = int(answer_index)
    
    # Find the question and answer
    question = collection.find_one({'_id': ObjectId(question_id)})
    answer = question['answers'][answer_index]

    # Increment the upvotes for the answer
    answer['upvotes'] += 1

    # Update the document in the database
    collection.update_one({'_id': ObjectId(question_id)}, {'$set': {'answers': question['answers']}})

    return redirect(url_for('help'))
def model_predict(img_path, model1):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model1.predict(x)
    return preds

def get_treatment_for_disease(disease_name):
    try:
        treatment = data.loc[disease_name, 'Treatment']
        return treatment
    except KeyError:
        return "Disease not found in the dataset"
    
@app.route('/plantdisease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return jsonify({'error': 'Unsupported file format'})

        # Ensure that the uploads directory exists
        upload_dir = "/uploads/"
        os.makedirs(upload_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        preds = model_predict(file_path, model1)
        # Replace 'disease_class' with your list of disease classes
        disease_class = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
                         

        predicted_class_index = np.argmax(preds)
        if predicted_class_index < len(disease_class):
            predicted_class = disease_class[predicted_class_index]
            print(f"Predicted Class Index: {predicted_class_index}")
            print(f"Predicted Class: {predicted_class}")
            treatment = get_treatment_for_disease(predicted_class)
            return render_template('result.html', result1=predicted_class, result2=treatment)
        else:
            print("Disease not found")
            return jsonify({'error': 'Disease not found'})

    return render_template('ind.html')
    
if __name__ == "__main__":
    app.run(debug=True)