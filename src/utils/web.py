import gradio as gr
import pandas as pd

class WebApp:
    def __init__(self, model, host: str = '0.0.0.0', port: int = 8080) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.genres = ["Female", "Male"]
        self.features = [
            "cc_addictionproblem", "cc_agitation", "cc_alcoholproblem", "cc_anxiety", "cc_depression", "cc_detoxevaluation", "cc_hallucinations", "cc_homicidal",
            "cc_panicattack", "cc_psychoticsymptoms", "cc_suicidal", "cc_alteredmentalstatus", "cc_abnormallab", "cc_abscess", "cc_alcoholintoxication",
            "cc_animalbite", "cc_asthma", "cc_cardiacarrest", "cc_chestpain", "cc_chesttightness", "cc_chills", "cc_coldlikesymptoms", "cc_confusion", "cc_cough",
            "cc_cyst", "cc_decreasedbloodsugar-symptomatic", "cc_dehydration", "cc_dentalpain", "cc_dizziness", "cc_drug/alcoholassessment", "cc_drugproblem",
            "cc_emesis", "cc_exposuretostd", "cc_fall", "cc_fatigue", "cc_fingerpain", "cc_fingerswelling", "cc_follow-upcellulitis", "cc_fulltrauma",
            "cc_generalizedbodyaches", "cc_gibleeding", "cc_giproblem", "cc_headache", "cc_headache-newonsetornewsymptoms", "cc_headache-recurrentorknowndxmigraines",
            "cc_headachere-evaluation", "cc_headinjury", "cc_headlaceration", "cc_hyperglycemia", "cc_hypertension", "cc_hypotension", "cc_influenza", "cc_ingestion",
            "cc_insectbite", "cc_irregularheartbeat", "cc_jawpain", "cc_jointswelling", "cc_lethargy", "cc_lossofconsciousness", "cc_mass", "cc_medicalproblem",
            "cc_medicalscreening", "cc_medicationproblem", "cc_medicationrefill", "cc_modifiedtrauma", "cc_nearsyncope", "cc_neurologicproblem", "cc_numbness",
            "cc_other", "cc_palpitations", "cc_poisoning", "cc_post-opproblem", "cc_psychiatricevaluation", "cc_rapidheartrate", "cc_respiratorydistress",
            "cc_ribinjury", "cc_ribpain", "cc_shortnessofbreath", "cc_sicklecellpain", "cc_stdcheck", "cc_suture/stapleremoval", "cc_syncope", "cc_tachycardia",
            "cc_testiclepain", "cc_tickremoval", "cc_toeinjury", "cc_toepain", "cc_trauma", "cc_uri", "cc_weakness", "cc_wheezing", "cc_withdrawal-alcohol",
            "cc_ankleinjury", "cc_arminjury", "cc_bleeding/bruising", "cc_blurredvision", "cc_bodyfluidexposure", "cc_breastpain", "cc_dyspnea", "cc_dysuria",
            "cc_groinpain", "cc_allergicreaction", "cc_burn", "cc_cellulitis", "cc_rash", "cc_skinirritation", "cc_skinproblem", "cc_woundcheck", "cc_woundinfection",
            "cc_woundre-evaluation", "cc_urinaryfrequency", "cc_urinaryretention", "cc_urinarytractinfection", "cc_vaginalbleeding", "cc_vaginaldischarge",
            "cc_vaginalpain", "cc_conjunctivitis", "cc_earpain", "cc_earproblem", "cc_epistaxis", "cc_eyeinjury", "cc_eyepain", "cc_eyeproblem", "cc_eyeredness",
            "cc_oralswelling", "cc_otalgia", "cc_sinusproblem", "cc_sorethroat", "cc_foreignbodyineye", "cc_nasalcongestion", "cc_abdominalcramping",
            "cc_abdominaldistention", "cc_abdominalpain", "cc_abdominalpainpregnant", "cc_constipation", "cc_diarrhea", "cc_epigastricpain", "cc_hematuria",
            "cc_hemoptysis", "cc_nausea", "cc_flankpain", "cc_rectalbleeding", "cc_rectalpain"
        ]
        self.features_questions = {
            "Abdominal Cramping": "cc_abdominalcramping", "Abdominal Distention": "cc_abdominaldistention", "Abdominal Pain Pregnant": "cc_abdominalpainpregnant",
            "Abdominal Pain": "cc_abdominalpain", "Abnormal Lab": "cc_abnormallab", "Abscess": "cc_abscess", "Adiction Problem": "cc_addictionproblem",
            "Agitation": "cc_agitation", "Alcohol Intoxication": "cc_alcoholintoxication", "Alcohol Problem": "cc_alcoholproblem",
            "Allergic Reaction": "cc_allergicreaction", "Altered Mental Status": "cc_alteredmentalstatus", "Animal Bite": "cc_animalbite",
            "Ankle Injury": "cc_ankleinjury", "Anxiety": "cc_anxiety", "Arm Injury": "cc_arminjury", "Asthma": "cc_asthma",
            "Bleeding | Bruising": "cc_bleeding/bruising", "Blurred Vision": "cc_blurredvision", "Body Fluid Exposure": "cc_bodyfluidexposure",
            "Breast Pain": "cc_breastpain", "Burn": "cc_burn", "Cardiac Arrest": "cc_cardiacarrest", "Cellulitis": "cc_cellulitis", "Chest Pain": "cc_chestpain",
            "Chest Tightness": "cc_chesttightness", "Chills": "cc_chills", "Cold Like Symptoms": "cc_coldlikesymptoms", "Confusion": "cc_confusion",
            "Conjunctivitis": "cc_conjunctivitis", "Constipation": "cc_constipation", "Cough": "cc_cough", "Cyst": "cc_cyst",
            "Decreased Blood Sugar-Symptomatic": "cc_decreasedbloodsugar-symptomatic", "Dehydration": "cc_dehydration", "Dental Pain": "cc_dentalpain",
            "Depression": "cc_depression", "Detox Evaluation": "cc_detoxevaluation", "Diarrhea": "cc_diarrhea", "Dizziness": "cc_dizziness",
            "Drug | Alcohol Assessment": "cc_drug/alcoholassessment", "Drug Problem": "cc_drugproblem", "Dyspnea": "cc_dyspnea", "Dysuria": "cc_dysuria",
            "Ear Pain": "cc_earpain", "Ear Problem": "cc_earproblem", "Emesis": "cc_emesis", "Epigastric Pain": "cc_epigastricpain", "Epistaxis": "cc_epistaxis",
            "Exposure To STD": "cc_exposuretostd", "Eye Injury": "cc_eyeinjury", "Eye Pain": "cc_eyepain", "Eye Problem": "cc_eyeproblem", "Eye Redness": "cc_eyeredness",
            "Fall": "cc_fall", "Fatigue": "cc_fatigue", "Finger Pain": "cc_fingerpain", "Finger Swelling": "cc_fingerswelling", "Flank Pain": "cc_flankpain",
            "Follow-Up Cellulitis": "cc_follow-upcellulitis", "Foreign Body In Eye": "cc_foreignbodyineye", "Full Trauma": "cc_fulltrauma",
            "Generalized Body Aches": "cc_generalizedbodyaches", "Gi Bleeding": "cc_gibleeding", "Gi Problem": "cc_giproblem", "Groin Pain": "cc_groinpain",
            "Hallucinations": "cc_hallucinations", "Head Injury": "cc_headinjury", "Head Laceration": "cc_headlaceration",
            "Headache - New Onset Or New Symptoms": "cc_headache-newonsetornewsymptoms",
            "Headache - Recurrent Or Known DX Migraines": "cc_headache-recurrentorknowndxmigraines", "Headache Re-Evaluation": "cc_headachere-evaluation",
            "Headache": "cc_headache", "Hematuria": "cc_hematuria", "Hemoptysis": "cc_hemoptysis", "Homicidal": "cc_homicidal", "Hyperglycemia": "cc_hyperglycemia",
            "Hypertension": "cc_hypertension", "Hypotension": "cc_hypotension", "Influenza": "cc_influenza", "Ingestion": "cc_ingestion", "Insect Bite": "cc_insectbite",
            "Irregular Heart Beat": "cc_irregularheartbeat", "Jaw Pain": "cc_jawpain", "Joint Swelling": "cc_jointswelling", "Lethargy": "cc_lethargy",
            "Loss Of Consciousness": "cc_lossofconsciousness", "Mass": "cc_mass", "Medical Problem": "cc_medicalproblem", "Medical Screening": "cc_medicalscreening",
            "Medication Problem": "cc_medicationproblem", "Medication Refill": "cc_medicationrefill", "Modified Trauma": "cc_modifiedtrauma",
            "Nasal Congestion": "cc_nasalcongestion", "Nausea": "cc_nausea", "Near Syncope": "cc_nearsyncope", "Neurologic Problem": "cc_neurologicproblem",
            "Numbness": "cc_numbness", "Oral Swelling": "cc_oralswelling", "Otalgia": "cc_otalgia", "Other": "cc_other", "Palpitations": "cc_palpitations",
            "Panic Attack": "cc_panicattack", "Phycotics Symptoms": "cc_psychoticsymptoms", "Poisoning": "cc_poisoning", "Post-Op Problem": "cc_post-opproblem",
            "Psychiatric Evaluation": "cc_psychiatricevaluation", "Rapid Heart Rate": "cc_rapidheartrate", "Rash": "cc_rash", "Rectal Bleeding": "cc_rectalbleeding",
            "Rectal Pain": "cc_rectalpain", "Respiratory Distress": "cc_respiratorydistress", "Rib Injury": "cc_ribinjury", "Rib Pain": "cc_ribpain",
            "Shortness Of Breath": "cc_shortnessofbreath", "Sickle Cell Pain": "cc_sicklecellpain", "Sinus Problem": "cc_sinusproblem",
            "Skin Irritation": "cc_skinirritation", "Skin Problem": "cc_skinproblem", "Sore Throat": "cc_sorethroat", "STD Check": "cc_stdcheck",
            "Suicidal": "cc_suicidal", "Suture | Staple Removal": "cc_suture/stapleremoval", "Syncope": "cc_syncope", "Tachycardia": "cc_tachycardia",
            "Testicle Pain": "cc_testiclepain", "Tick Removal": "cc_tickremoval", "Toe Injury": "cc_toeinjury", "Toe Pain": "cc_toepain", "Trauma": "cc_trauma",
            "Uri": "cc_uri", "Urinary Frequency": "cc_urinaryfrequency", "Urinary Retention": "cc_urinaryretention", "Urinary Tract Infection": "cc_urinarytractinfection",
            "Vaginal Bleeding": "cc_vaginalbleeding", "Vaginal Discharge": "cc_vaginaldischarge", "Vaginal Pain": "cc_vaginalpain", "Weakness": "cc_weakness",
            "Wheezing": "cc_wheezing", "Withdrawal - Alcohol": "cc_withdrawal-alcohol", "Wound Check": "cc_woundcheck", "Wound Infection": "cc_woundinfection",
            "Wound Re-Evaluation": "cc_woundre-evaluation"
        }
    
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
        inputs.append(gr.CheckboxGroup(choices=self.features_questions.keys(), label="Emergency Factors", info="Do Patient Has/Had..."))
        # inputs.append(gr.Dropdown(choices=self.other_factors, multiselect=True, label="Other Factors", info="Do Patient Has/Had..."))

        is_emergency = gr.Textbox(label="Classification")

        return gr.Interface(fn=self.classify_emergency, inputs=inputs, outputs=is_emergency, title='Triage Sense', allow_flagging='auto')

    def launch(self) -> None:
        self.demo = self.get_demo()
        self.demo.launch(server_name=self.host, server_port=self.port)


if __name__ == '__main__':
    app = WebApp()
    app.launch()
