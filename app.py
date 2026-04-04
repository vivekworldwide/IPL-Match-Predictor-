from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#load data
model = joblib.load('model.pkl')
encoder_team = joblib.load('encoder_team.pkl')
encoder_venue = joblib.load('encoder_venue.pkl')
encoder_toss = joblib.load('encoder_toss.pkl')

