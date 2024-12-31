# Emotion-Classification-Angry-Happy
# AHFace

AHFace is a Django web application that allows users to upload a facial image and detect emotions such as **anger** (angry) and **happiness** (happy).

---

## Features

- Upload facial images for emotion detection.
- Detects emotions like **angry** and **happy** using deep learning models.
- Simple and user-friendly web interface.

---

## Requirements

Ensure you have the following installed on your system:

- **Python 3**
- **Django 5**

---

## Installation

### Step 1: Clone the Repository


git clone [https://github.com/yourusername/ahface.git](https://github.com/DHarley22/Emotion-Classification-Angry-Happy.git)
cd ahface

### Step 2: Set Up a Virtual Environment

Create and activate a Python virtual environment:

python3 -m venv env
source env/bin/activate

### Step 3: Install Dependencies

Install the required Python modules:

pip install torch torchvision
pip install opencv-python
pip install django

### Step 4: Run the Application

    Navigate to the root directory of the project:

cd ahface

Run the migrations:

python manage.py migrate

Start the development server:

python manage.py runserver

Open your web browser and go to:

http://127.0.0.1:8000/face

## App Preview

![App Screenshot](https://github.com/DHarley22/Emotion-Classification-Angry-Happy/raw/main/images/app.png)
![App Detection](https://github.com/DHarley22/Emotion-Classification-Angry-Happy/raw/main/images/app2.png)











