import gradio as gr
import pandas as pd

class WebApp:
    def __init__(self, model, priority_file_path: str, host: str = '0.0.0.0', port: int = 8080) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.genres = ["Female", "Male"]
        self.data = pd.read_excel(priority_file_path)
        self.features = self.get_features()
        self.features_questions = self.get_questions()
    
    def get_questions(self):
        return dict(zip(self.data['Label'], self.data['Name']))
    
    def get_features(self):
        return self.data['Name'].tolist()
    
    def get_types(self):
        return self.data['Type'].tolist()
    
    def classify_emergency(self, *argv):
        patient_name = argv[0]
        patient_age = argv[1]
        patient_sex = argv[2]
        patient_symptoms = argv[3]

        temp_features = [self.features_questions[patient_symptom] for patient_symptom in patient_symptoms]
        features = [(feature in temp_features) for feature in self.features]

        data = {self.features[i]: [features[i]] for i in range(len(self.features))}
        df = pd.DataFrame.from_dict(data)

        result = self.model.predict(df)

        return f'Patient Name: {patient_name}, Patient Age: {patient_age}, Patient Sex: {patient_sex}, Patient Symptoms: {patient_symptoms}, Prediction: {result[0]}'

    def get_demo(self):
        inputs = []

        inputs.append(gr.Textbox(label="Patient Name"))
        inputs.append(gr.Number(label="Patient Age", minimum=0, maximum=150))
        inputs.append(gr.Dropdown(choices=self.genres, label="Patient Sex"))
        inputs.append(gr.CheckboxGroup(choices=self.features_questions.keys(), label="Patient Symptoms | State", info="Patient Has | Had..."))

        is_emergency = gr.Textbox(label="Classification")

        return gr.Interface(fn=self.classify_emergency, inputs=inputs, outputs=is_emergency, title='Triage Sense', allow_flagging='auto')

    def launch(self) -> None:
        self.demo = self.get_demo()
        self.demo.launch(server_name=self.host, server_port=self.port)


if __name__ == '__main__':
    app = WebApp()
    app.launch()
