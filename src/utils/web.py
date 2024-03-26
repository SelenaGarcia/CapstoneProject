import gradio as gr
import pandas as pd

from itertools import chain

class WebApp:
    def __init__(self, model, priority_file_path: str, host: str = '0.0.0.0', port: int = 8080) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.genres = ["Female", "Male"]
        self.data = pd.read_excel(priority_file_path)
        self.features = self.get_features()
        self.features_questions = self.get_questions_by_features()
        self.priorities = self.get_priorities_by_features()
    
    def get_questions_by_features(self):
        return dict(zip(self.data['Label'], self.data['Name']))
    
    def get_features(self):
        return self.data['Name'].tolist()
    
    def get_types(self):
        return self.data['Type'].tolist()
    
    def get_priorities_by_features(self):
        return dict(zip(self.data['Name'], self.data['Priority']))
    
    def classify_emergency(self, *argv):
        patient_name = argv[0]
        patient_age = argv[1]
        patient_sex = argv[2]
        patient_symptoms = list(chain.from_iterable(argv[3:]))

        temp_features = [self.features_questions[patient_symptom] for patient_symptom in patient_symptoms]
        
        data = {'age': [int(patient_age)], 'gender': [0] if patient_sex == 'Female' else [1]}
        for feature in self.features:
            data[feature] = [self.priorities[feature]] if feature in temp_features else [0]
        
        df = pd.DataFrame.from_dict(data)

        prediction = self.model.predict(df)

        result = ''
        result += f'Patient Name: {patient_name}\n'
        result += f'Patient Age: {int(patient_age)}\n'
        result += f'Patient Sex: {patient_sex}\n'
        result += f'Patient Symptoms: {patient_symptoms}\n'
        result += f'Is Emergency: {"Yes" if prediction[0][0] == 1 else "No"}\n'
        result += f'Priority: {prediction[0][1]} / 10'

        return result

    def get_demo(self):
        inputs = []

        inputs.append(gr.Textbox(label="Patient Name"))
        inputs.append(gr.Number(label="Patient Age", minimum=0, maximum=150))
        inputs.append(gr.Dropdown(choices=self.genres, label="Patient Sex"))

        categories = sorted(list(set(self.get_types())))
        for category in categories:
            symptoms = self.data[self.data['Type'] == category]
            questions = dict(zip(symptoms['Label'], symptoms['Name']))
            inputs.append(gr.CheckboxGroup(choices=questions.keys(), label=category, info="Patient Has | Had..."))

        is_emergency = gr.Textbox(label="Triage")

        return gr.Interface(fn=self.classify_emergency, inputs=inputs, outputs=is_emergency, title='Triage Sense', allow_flagging='auto')

    def launch(self) -> None:
        self.demo = self.get_demo()
        self.demo.launch(server_name=self.host, server_port=self.port)


if __name__ == '__main__':
    app = WebApp()
    app.launch()
