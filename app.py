import gradio as gr
import joblib
import numpy as np

model = joblib.load("loan_risk_model.pkl")

def predict_risk(balance, duration, payment_status):
    try:
    
        input_data = np.zeros(20)

        input_data[0] = balance
        input_data[1] = duration
        input_data[2] = payment_status

        prediction = model.predict([input_data])[0]

        if prediction == 1:
            return "Low Risk", "Customer is likely to repay the loan."
        else:
            return "High Risk", "Customer may default."

    except Exception as e:
        return "Hata", str(e)

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Loan Risk Agent")

    balance = gr.Number(label="Account Balance")
    duration = gr.Number(label="Credit Duration")
    payment_status = gr.Number(label="Previous Payment Status")

    btn = gr.Button("Predict")

    risk = gr.Textbox(label="Risk Level")
    explanation = gr.Textbox(label="Explanation")

    btn.click(
        predict_risk,
        inputs=[balance, duration, payment_status],
        outputs=[risk, explanation]
    )

demo.launch()
