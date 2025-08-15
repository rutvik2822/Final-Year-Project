# Behavioral CAPTCHA – ML Based Bot Detection

##  Project Overview
This project implements a **Behavioral CAPTCHA** system using a machine learning model to detect whether a drawing input is from a **human** or a **bot**.  
Unlike traditional image/text-based CAPTCHAs, this approach analyzes **user behavior patterns** such as speed, pauses, and total drawing time, making it more secure against automated attacks while remaining user-friendly.

---

##  Features
- Detects **bots** based on drawing behavior rather than static puzzles.
- Extracts behavioral features:
  - Average speed
  - Speed variance
  - Number of pauses
  - Total drawing time
- Frontend: Interactive HTML Canvas for drawing.
- Backend: Python Flask API with ML prediction.
- Hosting:
  - **Frontend:** Cloudflare Pages
  - **Backend:** Render

---

##  Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python Flask  
- **Machine Learning:** Scikit-learn (RandomForest Classifier)  
- **Hosting:** Cloudflare Pages & Render  

---

##  Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/rutvik2822/Final-Year-Project
   cd behavioral-captcha

2. **Install dependencies**
   pip install -r requirements.txt

3. **Run the backend**
   python app.py

4. **Run the frontend**
   Open index.html in your browser
   OR
   Deploy to a static hosting service like Cloudflare Pages.

## Usage
    Open the web interface.
    Draw on the canvas using your mouse or touch.
    Click Verify.
    The backend will:
    Extract behavioral features from your drawing data.
    Predict Human or Bot based on trained ML model.
    Result will be displayed instantly.

## Results

    Model Accuracy: 100% on test dataset.

    Classification Report:

                precision    recall  f1-score   support
            0       1.00      1.00      1.00
            1       1.00      1.00      1.00


## Future Scope
    Support for touch gestures on mobile.
    Adaptive difficulty levels for CAPTCHA tasks.
    More behavioral parameters for stronger detection.
    Integration with biometric authentication.

## Contributors
    Rutvik Devdare 
    Pranil More 
    Akshata Kuldev 
    Rajatkumar Moolya 
    Hardik Sonawane 

## License
    This project is developed for academic purposes at DY Patil University, School of Engineering & Technology.




