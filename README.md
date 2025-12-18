# 🌤️ Weather & Scene Classification App

A Deep Learning application built with **PyTorch** and **Streamlit** that classifies images into **Weather conditions** and **Natural Scenes**. The project features custom implementations of CNN architectures (**ResNet** and **DenseNet**) trained from scratch.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

## Link app

https://weather-scene-classification.streamlit.app/

## 📌 Features

* **Dual Functionality:** Switch easily between two modes:
    * 🌤️ **Weather Classification:** Detects 11 types of weather (dew, fog, rain, snow, etc.).
    * 🏞️ **Scene Classification:** Identifies 6 types of scenery (buildings, forest, mountain, etc.).
* **Custom Architectures:**
    * **ResNet:** Implemented from scratch using Residual Blocks.
    * **DenseNet:** Implemented from scratch using Dense Blocks and Transition Layers.
* **Interactive UI:** User-friendly web interface powered by Streamlit.
* **Real-time Prediction:** Upload an image and get instant results with confidence scores.

## 📂 Project Structure

```bash
Weather-Scene-Classification/
├── app/
│   ├── app.py                # Main Streamlit application
├── models/
│   ├── model_weather.pth     # Trained model for weather (PyTorch full model)
│   └── model_scenes.pth      # Trained model for scenes (PyTorch full model)
├── requirements.txt          # Python dependencies
├── src                       
│   ├── Weather-Scene.ipynb   
└── README.md                 # Project documentation
