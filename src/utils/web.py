import gradio as gr

class WebApp:
    def __init__(self, model) -> None:
        self.model = model
        self.genres = ["Female", "Male"]
        self.emergency_factors = [
            "Accidental Overdose",
            "Bleeding or Bruising",
            "Chest Pain",
            "Difficulty Breathing",
            "Elevated Blood Glucose Levels",
            "Fainting (Syncope)",
            "History of Seizures",
            "Hypertension",
            "Hypotension",
            "Intentional Overdose",
            "More than one Seizure",
            "Possible Stroke Alert",
            "Recent Seizure",
            "Suicidal Thoughts",
            "Unconscious or Unresponsive"
        ]
        self.other_factors = [
            "Fever",
            "Rash"
        ]
    
    def classify_emergency(self, *argv):
        is_emergency = "Emergency" if len(argv[3]) else "Consult"
        emergency_factors = f'Emergency Factors: {", ".join(argv[3]) if argv[3] else "None"}.'
        other_factors = f'Other Factors: {", ".join(argv[4]) if argv[4] else "None"}.'

        return f'Patiend: {argv[0]}, Age: {argv[1]}, Genre: {argv[2]}, Should go to: {is_emergency}. {emergency_factors} {other_factors}'
    
    def get_demo(self):
        inputs = []

        inputs.append(gr.Textbox(label="Patient Name"))
        inputs.append(gr.Number(label="Patient Age", minimum=0, maximum=150))
        inputs.append(gr.Dropdown(choices=self.genres, label="Patient Sex"))
        inputs.append(gr.CheckboxGroup(choices=self.emergency_factors, label="Emergency Factors", info="Do Patient Has/Had..."))
        inputs.append(gr.Dropdown(choices=self.other_factors, multiselect=True, label="Other Factors", info="Do Patient Has/Had..."))

        is_emergency = gr.Textbox(label="Classification")

        return gr.Interface(fn=self.classify_emergency, inputs=inputs, outputs=is_emergency, allow_flagging='auto')
    
    def launch(self) -> None:
        self.demo = self.get_demo()
        self.demo.launch()

if __name__ == '__main__':
    app = WebApp()
    app.launch()