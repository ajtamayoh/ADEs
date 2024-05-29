from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt
import spacy
import re
import time
from spacy.language import Language
#from spacy_language_detection import LanguageDetector
import sklearn
from joblib import load
#from transformers import pipeline
import requests
from huggingface_hub.inference_api import InferenceApi

#For Models from Hugging Face
@csrf_exempt
def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@csrf_exempt
def query2(payload, model_id, api_token, min_length, max_length):
        headers = {"Authorization": f"Bearer {api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        payload.update({
        "min_length": min_length,
        "max_length": max_length,
        "temperature": 0.9,
        "repetition_penalty": 1.2,
        "num_return_sequences": 1,
        })
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

#Post-processing for Models based on LM
@csrf_exempt
def grouping_entities(pred):
    import re
    output = []
    for e in pred:
        if "##" not in e['word']:
            output.append(e)
        else:
            try:
                if e['start'] == (output[-1]['end']):
                    output[-1]['word'] = output[-1]['word']+re.sub("##","",e['word'])
                    output[-1]['end'] = e['end']
            except:
                pass
    
        try:
            #if (e['entity_group'] == "B" or e['entity_group'] == "I") and (e['start'] == (output[-2]['end']+1)):
            if e['start'] == (output[-2]['end']+1):
                output[-2]['word'] = output[-2]['word']+" "+e['word']
                output[-2]['end'] = e['end']
                output.pop(-1)
        except:
            pass
    
        try:
            if e['start'] == (output[-2]['end']):
                output[-2]['word'] = output[-2]['word']+e['word']
                output[-2]['end'] = e['end']
                output.pop(-1)
        except:
            pass

    return output

@csrf_exempt
def index(request):

    return render(request, 'simple/index.html')

@csrf_exempt
def ades(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
            #elif request.POST['texto2'] != "": #if the option to upload a file is available, uncomment this and the following line.
            #    texto = request.FILES['texto2'].read().decode('utf-8')
                #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                    #texto = f.read()
                #print(texto)
    except:
        pass
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    s_original = texto # Applied the simple sentence tokenization
    ADEs_ents = []

    try:
        #ADEs
        ADEs_ents_ld = []
        #ADEs_model_id = "ajtamayoh/bert-finetuned-ADEs_model_1"
        #api_token = "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"    #token ajtamayoh
        ADEs_model_id = "Sonatafyai/scibert-finetuned_ADEs_SonatafyAI"
        api_token = "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"     #token Sonatafy AI
        s_original_tokenized = s_original.split('. ')
        for sent in s_original_tokenized:
            ADEs_entities = query(sent, ADEs_model_id, api_token)
            try:
                if ADEs_entities['error'] != '':
                    time.sleep(5)
                    ADEs_entities = query(sent, ADEs_model_id, api_token)
            except:
                ADEs_ents_ld.append(grouping_entities(ADEs_entities))

        #ADEs_entities = query(s_original, ADEs_model_id, api_token)
        #ADEs_ents_ld = grouping_entities(ADEs_entities)
        #for et in sum(ADEs_ents_ld, []):
        for e in ADEs_ents_ld:
           for et in e:
               print(et["word"])
               ADEs_ents.append(et["word"])
               if et['entity_group']== 'BD':
                   #texto = texto.lower().replace(et["word"], '<button class="btn btn-warning no-link">'+et["word"]+'</button>')
                   texto = texto.lower().replace(et["word"], '<span style="background-color: skyblue; color: white; padding: 5px; font-weight: bold;">'+et["word"]+'</span>')

               elif et['entity_group']=='BAE':
                   #texto = texto.lower().replace(et["word"], '<button class="btn btn-danger no-link">'+et["word"]+'</button>')
                   texto = texto.lower().replace(et["word"], '<span style="color: red; font-weight: bold;">'+et["word"]+'</span>')

        texto = re.sub(r'(<button class="btn btn-warning no-link">)+', '<button class="btn btn-warning no-link">', texto)
        texto = re.sub(r'(</button>)+', '</button>', texto)
        texto = re.sub(r'(<button class="btn btn-danger no-link">)+', '<button class="btn btn-danger no-link">', texto)

    except:
        print("Problems with ADEs")

    context = {
        'msg': texto,
        #'Diseases': SDN_ents,
        #'Procedures': PRO_ents,
        #'Organisms': LNER_ents,
        #'Neg_unc': NegNER_sco,
        #'Unc': Unc_sco,
        'ADEs': ADEs_ents,
    }
    
    return render(request, 'simple/ades.html', context)


@csrf_exempt
def diseases(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
            #elif request.POST['texto2'] != "": #if the option to upload a file is available, uncomment this and the following line.
            #    texto = request.FILES['texto2'].read().decode('utf-8')
                #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                    #texto = f.read()
                #print(texto)
    except:
        pass
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    s_original = texto # Applied the simple sentence tokenization
    Dis_ents = []

    try:
        #Diseases
        Dis_ents_ld = []
        Dis_model_id = "sonatafyai/Disease_Identification_SonatafyAI_BERT_v1"
        api_token = "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"     #token Sonatafy AI
        s_original_tokenized = s_original.split('. ')
        for sent in s_original_tokenized:
            Dis_entities = query(sent, Dis_model_id, api_token)
            try:
                if Dis_entities['error'] != '':
                    time.sleep(5)
                    Dis_entities = query(sent, Dis_model_id, api_token)
            except:
                Dis_ents_ld.append(grouping_entities(Dis_entities))

        for e in Dis_ents_ld:
           for et in e:
               print(et["word"])
               Dis_ents.append(et["word"])
               #texto = texto.lower().replace(et["word"].strip(), '<button class="btn btn-danger no-link">'+et["word"].strip()+'</button>')
               texto = texto.lower().replace(et["word"], '<span style="color: red; font-weight: bold;">'+et["word"]+'</span>')
        texto = re.sub(r'(<button class="btn btn-danger no-link">)+', '<button class="btn btn-danger no-link">', texto)
        texto = re.sub(r'(</button>)+', '</button>', texto)

    except:
        print("Problems with Diseases")

    context = {
        'msg': texto,
        'Dis': Dis_ents,
    }
    
    return render(request, 'simple/diseases.html', context)

@csrf_exempt
def symptoms2diagnosis(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
            #elif request.POST['texto2'] != "": #if the option to upload a file is available, uncomment this and the following line.
            #    texto = request.FILES['texto2'].read().decode('utf-8')
                #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                    #texto = f.read()
                #print(texto)
    except:
        pass
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    diagnosis = ""
    confidence = 0.0

    try:
        #Symptoms to diagnosis
        S2D_model_id = "sonatafyai/Symptoms_to_Diagnosis_SonatafyAI_BERT_v1"
        api_token = "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"     #token Sonatafy AI
        response = query(texto, S2D_model_id, api_token)
        try:
            if response['error'] != '':
                time.sleep(5)
                response = query(texto, S2D_model_id, api_token)
        except:
            diagnosis = 'symptoms are consistent with <b>' + response[0][0]['label'] + '</b>.'
            confidence = response[0][0]['score']

    except:
        print("Problems with symptoms to diagnosis")

    context = {
        'msg': texto,
        'diagnosis': diagnosis,
        'confidence': confidence,
    }
    
    return render(request, 'simple/symptoms2diagnosis.html', context)


@csrf_exempt
def sonatafyassistant(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
    except:
        pass

    response = ""

    if texto != "":
        #Eliminar saltos de linea y doble espacio
        texto = re.sub(r'\n+', ' ' ,texto)
        texto = re.sub(r'\s+', ' ', texto)

        response = ""
        #Preparing the response
        LLM_model_id = "meta-llama/Llama-2-7b-chat-hf"
        #LLM_model_id = "Sonatafyai/BioGPT_DocBot_SonatafyAI_V1"
        api_token = "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"
        payload = {"inputs": "Instructions: If you are a doctor, please give a medical concept based on the patient's description. (Do not leave uncomplete phrases) : \nPatient description: " + texto + "\nConcept: "}
        response = query2(payload, LLM_model_id, api_token, 100, 500)
        print(response)
        try:
            if response['error'] != '':
                time.sleep(5)
                response = query2(payload, LLM_model_id, api_token, 100, 500)
                #response = response[0]['generated_text']
        except:
            response = response[0]['generated_text']
            response = response.split(texto)
            #response = "<b>User:</b> " + texto + "<br>" + "<b>Sonatafy Assistant:</b> "+ response[1]
            
            try:
                response = response[1][:response[1].rfind('.')+1]
                response = re.sub("Concept:","<b>Concept:</b>",response)
            except:
                response = response[1]
                response = re.sub("Concept:","<b>Concept:</b>",response)
            
    context = {
        'msg': texto,
        'response': response,
    }
    
    return render(request, 'simple/sonatafyassistant.html', context)


@csrf_exempt
def summarization(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
    except:
        pass

    response = ""

    if texto != "":
        #Eliminar saltos de linea y doble espacio
        texto = re.sub(r'\n+', ' ' ,texto)
        texto = re.sub(r'\s+', ' ', texto)

        response = ""
        #Preparing the response
        LLM_model_id = "meta-llama/Llama-2-7b-chat-hf"
        #LLM_model_id = "Sonatafyai/BioGPT_DocBot_SonatafyAI_V1"
        api_token = "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"
        payload = {"inputs": "Instructions: Summarize the following clinical document (Do not leave uncomplete phrases) : \nClinical document: " + texto + "\n<b>Summary</b>: "}
        response = query2(payload, LLM_model_id, api_token, 100, 500)
        print(response)
        try:
            if response['error'] != '':
                time.sleep(5)
                response = query2(payload, LLM_model_id, api_token, 100, 500)
                #response = response[0]['generated_text']
        except:
            response = response[0]['generated_text']
            response = response.split(texto)
            #response = "<b>User:</b> " + texto + "<br>" + "<b>Sonatafy Assistant:</b> "+ response[1]
            
            try:
                response = response[1][:response[1].rfind('.')+1]
            except:
                response = response[1]
            
    context = {
        'msg': texto,
        'response': response,
    }
    
    return render(request, 'simple/summarization.html', context)


@csrf_exempt
def sentiment_analysis(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
            #elif request.POST['texto2'] != "": #if the option to upload a file is available, uncomment this and the following line.
            #    texto = request.FILES['texto2'].read().decode('utf-8')
                #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                    #texto = f.read()
                #print(texto)
    except:
        pass
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    sentiment  = ""
    confidence = 0.0

    try:
        #Symptoms to diagnosis
        SA_model_id = "sonatafyai/Sentiment_Analysis_in_Social_Media_SonatafyAI_BERT_v1"
        api_token = "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"     #token Sonatafy AI
        response = query(texto, SA_model_id, api_token)
        try:
            if response['error'] != '':
                time.sleep(5)
                response = query(texto, S2D_model_id, api_token)
        except:
            sentiment  = response[0][0]['label']
            confidence = response[0][0]['score']

    except:
        print("Problems with sentiment analysis")

    context = {
        'msg': texto,
        'sentiment': sentiment,
        'confidence': confidence,
    }
    
    return render(request, 'simple/sentiment-analysis.html', context)

@csrf_exempt
def fake_news_detection(request):

    texto = ""
    try:
        if request.method == "POST":
            if request.POST['texto1'] != "":
                texto = request.POST['texto1']
            #elif request.POST['texto2'] != "": #if the option to upload a file is available, uncomment this and the following line.
            #    texto = request.FILES['texto2'].read().decode('utf-8')
                #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                    #texto = f.read()
                #print(texto)
    except:
        pass
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    label = ""
    confidence = 0.0

    try:
        #Symptoms to diagnosis
        FN_model_id = "sonatafyai/Fake_news_Detection_SonatafyAI_RoBERTa"
        api_token = "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"     #token Sonatafy AI
        response = query(texto, FN_model_id, api_token)
        try:
            if response['error'] != '':
                time.sleep(5)
                response = query(texto, FN_model_id, api_token)
        except:
            label = '<b>' + response[0][0]['label'] + ' news</b>.'
            confidence = response[0][0]['score']

    except:
        print("Problems with fake news.")

    context = {
        'msg': texto,
        'label': label,
        'confidence': confidence,
    }
    
    return render(request, 'simple/fake_news_detection.html', context)


def about_ades(request):

    return render(request, 'simple/about_ades.html')

def historias_clinicas(request):

    return render(request, 'simple/historias_clinicas.html')

#@csrf_protect
def credits(request):

    return render(request, 'simple/credits.html')

def agradecimientos(request):

    return render(request, 'simple/agradecimientos.html')
