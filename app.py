import os

script_directory = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_directory)

import base64
import io
import sqlite3
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_file
import torch
import datetime
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model_path = 'best.pt'
device = torch.device('cpu')  # Change to 'cuda' if you have a GPU
model = torch.hub.load('"ultralytics/yolov5"', 'custom', path=model_path)

class_labels = ['M.Beer', 'MD.Diet', 'MD.Orig', 'P.Cherry', 'P.Orig', 'P.Rsugar', 'P.Zero', 'P.Diet']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            files = request.files.getlist('file')
            labels = [x.strip() for x in request.form.getlist('label')[0].split(",")]

            if len(files) != len(labels):
                return render_template('predict.html', error='Number of files and labels must match')

            if not files or all(file.filename == '' for file in files):
                return render_template('predict.html', error='No files provided')

            predictions_list = []

            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()

            for file, true_label in zip(files, labels):
                if file.filename.split('.')[-1].lower() not in {'jpg', 'jpeg', 'png'}:
                    return render_template('predict.html', error='Invalid file format')

                image = Image.open(file.stream).convert('RGB')
                input_tensor = [image]

                with torch.no_grad():
                    output = model(input_tensor)
                    output.render()
                    image = Image.fromarray(output.ims[0])

                predicted_class = output.pandas().xyxy[0].iloc[0]["name"]

                cursor.execute("INSERT INTO predictions (true_label, predicted_label) VALUES (?, ?)",
                               (true_label, predicted_class))

                conn.commit()

                correctness = "correct" if predicted_class == true_label else "incorrect"

                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

                probabilities_str = []

                predictions = {'predicted_class': f'{predicted_class}, {correctness}', 'probabilities': probabilities_str, 'image': image_data}
                predictions_list.append(predictions)

            conn.close()

            return render_template('predict.html', predictions_list=predictions_list)

        return render_template('predict.html')

    except Exception as e:
        return render_template('predict.html', error=f'Error: {e}')

@app.route('/stats', methods=['GET'])
def stats():
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()

        query = """
            SELECT 
                true_label,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END) as correct_predictions
            FROM predictions
            WHERE timestamp BETWEEN datetime('now', '-10 days') AND datetime('now')
            GROUP BY true_label
            ORDER BY true_label;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        conn.close()

        stats_data = []
        for row in rows:
            true_label, total_predictions, correct_predictions = row
            accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0
            stats_data.append({
                'class': true_label,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy
            })

        return jsonify(stats_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats_visualize', methods=['GET'])
def stats_visualize():
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()

        query = """
            SELECT 
                true_label,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END) as correct_predictions
            FROM predictions
            WHERE timestamp BETWEEN datetime('now', '-10 days') AND datetime('now')
            GROUP BY true_label
            ORDER BY true_label;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        labels = []
        accuracies = []

        for row in rows:
            true_label, total_predictions, correct_predictions = row
            accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0
            labels.append(true_label)
            accuracies.append(accuracy)
        conn.close()

        plt.figure(figsize=(10, 6))
        plt.bar(labels, accuracies, color='blue')
        plt.xlabel('Label')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for each label')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')

        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        plt.close()

        return send_file(image_stream, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({'classes': class_labels})


if __name__ == '__main__':
    app.run(debug=True)
