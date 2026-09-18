import gradio as gr
import joblib

model = joblib.load("logistic_regression_student_studyhours_model.pkl")

def predict_result(hours):
    input_data = [[hours]]

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    pass_probability = probabilities[0][1] * 100
    fail_probability = probabilities[0][0] * 100

    if prediction[0] == 1:
        result = "✅ Student is predicted to PASS"
    else:
        result = "❌ Student is predicted to FAIL"

    return (
        result,
        f"Probability of Passing: {pass_probability:.2f}%",
        f"Probability of Failing: {fail_probability:.2f}%"
    )

demo = gr.Interface(
    fn=predict_result,
    inputs=gr.Number(
        label="Enter Study Hours",
        minimum=0,
        maximum=15,
        value=5
    ),
    outputs=[
        gr.Textbox(label="📊 Prediction Result"),
        gr.Textbox(label="Probability of Passing"),
        gr.Textbox(label="Probability of Failing")
    ],
    title="🎓 Student Pass/Fail Prediction",
    description="Enter the number of study hours to predict whether the student will pass or fail."
)

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
