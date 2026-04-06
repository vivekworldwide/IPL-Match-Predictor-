from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#load data
model = joblib.load('model.pkl')
encoder_team = joblib.load('encoder_team.pkl')
encoder_venue = joblib.load('encoder_venue.pkl')
encoder_toss = joblib.load('encoder_toss.pkl')

@app.route('/')
def home():
    teams = list(encoder_team.classes_)
    venues = list(encoder_venue.classes_)
    toss_decisions = list(encoder_toss.classes_)
    return render_template('index.html', teams=teams, venues=venues, toss_decisions=toss_decisions)

@app.route('/predict', methods=['POST'])
def predict():

    team1 = request.form['team1']
    team2 = request.form['team2']
    venue = request.form['venue']
    toss_winner = request.form['toss_winner']
    toss_decision = request.form['toss_decision']

    #encode catogorical data
    team1_encoded = encoder_team.transform([team1])[0]
    team2_encoded = encoder_team.transform([team2])[0]
    venue_encoded = encoder_venue.transform([venue])[0]
    toss_winner_encoded = encoder_team.transform([toss_winner])[0]
    toss_decision_encoded = encoder_toss.transform([toss_decision])[0]

    #feature vector 
    features = np.array([[team1_encoded, team2_encoded, venue_encoded, toss_winner_encoded, toss_decision_encoded]])
    prediction = model.predict(features)[0]
    probablities = model.predict_proba(features)[0]

    winner = encoder_team.inverse_transform([prediction])[0]

    win_prob = round(max(probablities) *100, 2)
    
    teams = list(encoder_team.classes_)
    venues = list(encoder_venue.classes_)
    toss_decisions = list(encoder_toss.classes_)

    return render_template('index.html', teams=teams, venues=venues, toss_decisions=toss_decisions,
                           winner=winner, win_prob=win_prob)

if __name__ == '__main__':
    app.run(debug = True)