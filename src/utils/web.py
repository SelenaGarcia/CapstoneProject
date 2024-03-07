import gradio as gr

class WebApp:
    def classify_emergency(*argv):
        for arg in argv:
            print(arg)
        return 'Classifier Working'
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                chest_pain = gr.Checkbox(label="Does the patient have chest pain?")
                dificult_breathing = gr.Checkbox(label="Is the patient having difficulty breathing?")
                fainted = gr.Checkbox(label="Has the patient experienced fainting (syncope)?")
                unresponsive = gr.Checkbox(label="Is the patient unconscious or unresponsive?")
                recent_seizure = gr.Checkbox(label="Has the patient had a recent seizure?")
                history_seizure = gr.Checkbox(label="Does the patient have a history of seizures?")
                more_than_1_seizure = gr.Checkbox(label="Has the patient had more than one seizure?")
                bleeding = gr.Checkbox(label="Is the patient experiencing bleeding or bruising?")
            with gr.Column():
                hight_glucose = gr.Checkbox(label="Does the patient have elevated blood glucose levels?")
                hypertension = gr.Checkbox(label="Does the patient have hypertension?")
                hypotension = gr.Checkbox(label="Does the patient have hypotension?")
                stroke_alerts = gr.Checkbox(label="Is the patient a possible stroke alert?")
                accidental_overdose = gr.Checkbox(label="Has the patient had an accidental overdose?")
                intentional_overdose = gr.Checkbox(label="Has the patient had an intentional overdose?")
                suicidal_thoughts = gr.Checkbox(label="Does the patient have suicidal thoughts?")
            with gr.Column():
                is_emergency = gr.Textbox(label="Classification")
                classify_btn = gr.Button(value="Is Emergency?")
        
        inputs = [chest_pain, dificult_breathing, fainted, unresponsive, recent_seizure, history_seizure, more_than_1_seizure, bleeding, hight_glucose, hypertension, hypotension, stroke_alerts, accidental_overdose, intentional_overdose, suicidal_thoughts]
        classify_btn.click(classify_emergency, inputs=inputs, outputs=is_emergency, api_name="Emergency Classificator")

    def launch(self) -> None:
        self.demo.launch()

if __name__ == '__main__':
    app = WebApp()
    app.launch()