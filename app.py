from flask import Flask, request, jsonify, render_template
#import pickle 
import sigma_cabs
import numpy as np


app=Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')

@app.route("/",methods=["POST","Get"])
def le_ip():
    if request.method=="POST":
        v1=int(request.form["Trip_Distance"])
        v2=request.form["Type_of_Cab"]
        v3=int(request.form["Customer_Since_Months"])
        v4=request.form["Destination_Type"]
        v5=float(request.form["Customer_Rating"])
        v6=int(request.form["Var3"])

        final_features=[v1,v2,v3,v4,v5,v6]
        
        w=sigma_cabs.Loaded_model(final_features)
        
    return render_template("index.html",prediction=w)
if __name__=="__main__":
    app.run(debug=True)
