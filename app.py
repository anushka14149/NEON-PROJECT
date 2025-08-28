
# html_template = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Soil Predictor</title>
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
#     <style>
#         body {
#             background: linear-gradient(120deg, #d4fc79, #96e6a1);
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         }
#         .container {
#             margin-top: 50px;
#             max-width: 700px;
#         }
#         .card {
#             border-radius: 15px;
#             box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
#         }
#         h2 {
#             color: #2d572c;
#             text-align: center;
#             margin-bottom: 20px;
#         }
#         .btn-custom {
#             background: #2d572c;
#             color: white;
#             font-weight: bold;
#         }
#         .btn-custom:hover {
#             background: #244422;
#             color: #fff;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="card p-4">
#             <h2>Soil Parameter Prediction</h2>
#             <form method="POST">
#                 <div class="mb-3">
#                     <label class="form-label">Select Target Parameter:</label>
#                     <select class="form-select" name="target" onchange="this.form.submit()">
#                         <option value="">--Choose--</option>
#                         {% for t in targets %}
#                           <option value="{{ t }}" {% if selected_target == t %}selected{% endif %}>{{ t }}</option>
#                         {% endfor %}
#                     </select>
#                 </div>
#             </form>

#             {% if selected_target %}
#             <form method="POST">
#                 <input type="hidden" name="target" value="{{ selected_target }}">
#                 <div class="row">
#                 {% for feature in features %}
#                     <div class="col-md-6 mb-3">
#                         <label class="form-label">{{ feature }}</label>
#                         <input type="text" class="form-control" name="{{ feature }}" placeholder="Enter {{ feature }}">
#                     </div>
#                 {% endfor %}
#                 </div>
#                 <button type="submit" class="btn btn-custom w-100">Predict</button>
#             </form>
#             {% endif %}

#             {% if prediction %}
#               <div class="alert alert-success mt-4 text-center">
#                 <h4> Prediction for <b>{{ selected_target }}</b>: <span style="color:#2d572c;">{{ prediction }}</span></h4>
#               </div>
#             {% endif %}
#         </div>
#     </div>
# </body>
# </html>
# """

# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = None
#     selected_target = request.form.get("target")
#     features = []

#     if selected_target in targets_features:
#         features = targets_features[selected_target]
#         if request.method == "POST" and all(f in request.form for f in features):
#             try:
#                 model = joblib.load(f"{selected_target}_model.pkl")
#                 inputs = [float(request.form[f]) for f in features]
#                 prediction = round(model.predict([inputs])[0], 3)
#             except Exception as e:
#                 prediction = f"Error: {str(e)}"

#     return render_template_string(html_template,
#                                   targets=targets_features.keys(),
#                                   selected_target=selected_target,
#                                   features=features,
#                                   prediction=prediction)

# if __name__ == "__main__":
#     print(" Flask app starting... Visit http://127.0.0.1:5000")
#     app.run(debug=True, port=5000)





from flask import Flask, request, render_template_string, send_file
import joblib
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

targets_features = {
    'soilInWaterpH': [ 'dryMassFraction','pHSoilInWaterMass', 'sampleTopDepth', 'elevation'],
    'soilTemp': ['soilMoisture', 'dryMassFraction', 'elevation', 'sampleBottomDepth', 'pHSoilInCaClMass'],
    'soilMoisture': ['dryMassBoatMass', 'pHCaClVol', 'pHSoilInCaClMass','waterpHRatio'],
    'pHCaClVol': ['waterpHRatio', 'pHWaterVol', 'soilMoisture', 'dryMassFraction'],
    'boatMass': ['dryMassBoatMass', 'sampleTopDepth', 'elevation', 'soilInCaClpH'],
    'sampleBottomDepth': ['dryMassBoatMass', 'dryMassFraction', 'pHSoilInCaClMass', 'waterpHRatio', 'caclpHRatio'],
    'dryMassFraction': ['pHSoilInWaterMass', 'waterpHRatio', 'caclpHRatio', 'sampleBottomDepth']
}

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soil Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: url('/static/bg.png') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            margin-top: 60px;
            max-width: 750px;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
            background: rgba(255, 255, 255, 0.95);
        }
        h2 {
            color: #2d572c;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .btn-custom {
            background: #2d572c;
            color: white;
            font-weight: bold;
        }
        .btn-custom:hover {
            background: #244422;
            color: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card p-4">
            <h2> Soil Parameter Prediction</h2>
            <form method="POST">
                <div class="mb-3">
                    <label class="form-label">Select Target Parameter:</label>
                    <select class="form-select" name="target" onchange="this.form.submit()">
                        <option value="">--Choose--</option>
                        {% for t in targets %}
                          <option value="{{ t }}" {% if selected_target == t %}selected{% endif %}>{{ t }}</option>
                        {% endfor %}
                    </select>
                </div>
            </form>

            {% if selected_target %}
            <form method="POST">
                <input type="hidden" name="target" value="{{ selected_target }}">
                <div class="row">
                {% for feature in features %}
                    <div class="col-md-6 mb-3">
                        <label class="form-label">{{ feature }}</label>
                        <input type="text" class="form-control" name="{{ feature }}" placeholder="Enter {{ feature }}">
                    </div>
                {% endfor %}
                </div>
                <button type="submit" class="btn btn-custom w-100"> Predict</button>
            </form>
            {% endif %}

            {% if prediction %}
              <div class="alert alert-success mt-4 text-center">
                <h4> Prediction for <b>{{ selected_target }}</b>: <span style="color:#2d572c;">{{ prediction }}</span></h4>
              </div>
              <div class="text-center mt-3">
                <a href="/plot?target={{selected_target}}" class="btn btn-outline-success"> Show Chart</a>
              </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    selected_target = request.form.get("target")
    features = []

    if selected_target in targets_features:
        features = targets_features[selected_target]
        if request.method == "POST" and all(f in request.form for f in features):
            try:
                model = joblib.load(f"{selected_target}_model.pkl")
                inputs = [float(request.form[f]) for f in features]
                prediction = round(model.predict([inputs])[0], 3)

                # Store inputs + prediction in session for chart
                app.config['last_inputs'] = inputs
                app.config['last_features'] = features
                app.config['last_prediction'] = prediction

            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template_string(html_template,
                                  targets=targets_features.keys(),
                                  selected_target=selected_target,
                                  features=features,
                                  prediction=prediction)

@app.route("/plot")
def plot():
    """Generate chart comparing input features vs predicted value"""
    features = app.config.get('last_features', [])
    inputs = app.config.get('last_inputs', [])
    prediction = app.config.get('last_prediction', None)

    if not features or not inputs:
        return "No prediction data available!"

    plt.figure(figsize=(8, 5))
    plt.bar(features, inputs, color="skyblue", label="Input Values")
    plt.axhline(y=prediction, color="green", linestyle="--", label=f"Predicted {prediction}")
    plt.title(" Feature Inputs & Predicted Val")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Values")
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    print(" Flask app starting... Visit http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
