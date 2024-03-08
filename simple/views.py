from django.shortcuts import render
import spacy
import re
from spacy.language import Language
from spacy_language_detection import LanguageDetector
import sklearn
from joblib import load
#from transformers import pipeline
import requests
from huggingface_hub.inference_api import InferenceApi

#For Models from Hugging Face
def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

#Post-processing for Models based on LM
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

def index(request):

    #Para despertar los modelos del API de Hugging Face.
    #Esto no es viable porque se consume rapidamente las llamadas gratuitas al API?
    '''
    for i in range(10):
        #Negation and Uncertainty Scope Detection
        NegNER_model_id = "ajtamayoh/Negation_and_Uncertainty_Scope_Detection_RoBERTa_fine_tuned"
        api_token = "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" # get yours at hf.co/settings/tokens
        NegNER_scopes = query("despierta", NegNER_model_id, api_token)
    
        #Disease extraction
        SDN_model_id = "ajtamayoh/Disease_Identification_RoBERTa_fine_tuned"
        api_token = "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" # get yours at hf.co/settings/tokens
        #SDN_entities = query(" ".join(s), SDN_model_id, api_token)
        SDN_entities = query("despierta", SDN_model_id, api_token)

        #Species extraction
        #LNER_model_id = "ajtamayoh/NLP-CIC-WFU_Clinical_Cases_NER_Sents_tokenized_mBERT_cased_fine_tuned" #LivingNER
        LNER_model_id = "ajtamayoh/Species_Identification_RoBERTa_fine_tuned"
        api_token = "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" # get yours at hf.co/settings/tokens
        LNER_entities = query("despierta", LNER_model_id, api_token)

        #Procedures extraction
        PRO_model_id = "ajtamayoh/Procedures_Identification_RoBERTa_fine_tuned"
        api_token = "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" # get yours at hf.co/settings/tokens
        PRO_entities = query("despierta", PRO_model_id, api_token)
    '''


    texto = ""
    if request.method == "POST":
        if request.POST['texto1'] != "":
            texto = request.POST['texto1']
        else:
            texto = request.FILES['texto2'].read().decode('utf-8')
            #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                #texto = f.read()
            #print(texto)
    
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)

    s_original = texto # Applied the simple sentence tokenization

    SDN_ents = []
    LNER_ents = []
    PRO_ents = []
    NegNER_sco = []
    Unc_sco = []

    try:
        #OSR detection and linking with negation ( + Uncertainty Scope Detection )
        #NegNER_model_id = "ajtamayoh/NeRUBioS_RoBERTa_Training_Testing"
        #OSR detection and linking (sin UNC ni USCO)
        NegNER_model_id = "ajtamayoh/RE_NegREF_NSD_Nubes_Training_Test_dataset_roberta-base-biomedical-clinical-es_fine_tuned_v3"
        api_token = "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" # get yours at hf.co/settings/tokens
        NegNER_scopes = query(s_original, NegNER_model_id, api_token)
        #NegNER_scopes = token_classifier_neg(" ".join(s_original))
        NegNER_sco_ld = grouping_entities(NegNER_scopes)
        for ns in NegNER_sco_ld:
            ns_clean = ns["word"]
            ns_clean = ns_clean.replace("  ", " ")
            NegNER_sco.append(ns_clean)
            if ns_clean.replace(" ", "") != "de":
            	texto = texto.replace(ns_clean, '<button class="btn btn-secondary">'+ns_clean+'</button>')
            #print(NegNER_sco)
            #print(NegNER_sco_ld)
        #print(NegNER_sco_ld)
    except:
        print("Problemas con Negation")

    try:
        #Uncertainty
        Unc_model_id = "ajtamayoh/Negation_and_Uncertainty_Scope_Detection_mBERT_fine_tuned"
        api_token = "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"
        Unc_scopes = query(s_original, Unc_model_id, api_token)
        Unc_sco_ld = grouping_entities(Unc_scopes)
        for un in Unc_sco_ld:
            #un_clean = un["word"][1:]
            #un_clean = un_clean.replace("  ", " ")
            #Unc_sco.append(un_clean)
            if (un['entity_group'] == 'UNC' or un['entity_group'] == 'USCO') and un['word'].replace(" ", "") != "de":
                texto = texto.replace(un["word"], '<button class="btn btn-info">'+un["word"]+'</button>')
        print(Unc_sco_ld)
        #print(texto)
    except:
        print("Problemas con Uncertainty")

    try:
        #Disease extraction
        SDN_model_id = "ajtamayoh/Disease_Identification_RoBERTa_fine_tuned_Testing"
        api_token = "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" # get yours at hf.co/settings/tokens
        #SDN_entities = query(" ".join(s), SDN_model_id, api_token)
        SDN_entities = query(s_original, SDN_model_id, api_token)
        SDN_ents_ld = grouping_entities(SDN_entities)

        #SDN_ents_ld_post = post_processing_ents(" ".join(s), SDN_ents_ld) #Se debe ajustar el método de post-processing, ejecuta con error 500 desde la GUI.
        for et in SDN_ents_ld:
            if et["word"][1:] not in NegNER_sco:
                et_clean = et["word"][1:]
                et_clean = et_clean.replace("  ", " ")
                SDN_ents.append(et_clean)
                #s = " ".join(s).replace(et["word"], "<span style='background-color: yellow'>"+et["word"]+"</span>").split()
                texto = texto.replace(et_clean, '<button class="btn btn-warning">'+et_clean+'</button>')
                #print(SDN_ents)
        #print(SDN_ents)
        #print(texto)
    except:
        print("Problemas con Diseases")
    
    try:
        #Species extraction
        LNER_model_id = "ajtamayoh/Species_Identification_mBERT_fine_tuned_Train_Test"
        api_token = "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" # get yours at hf.co/settings/tokens
        LNER_entities = query(s_original, LNER_model_id, api_token)
        LNER_ents_ld = grouping_entities(LNER_entities)
        for et in LNER_ents_ld:
            LNER_ents.append(et["word"])
            texto = texto.replace(et["word"], '<button class="btn btn-primary">'+et["word"]+'</button>')
            #print(LNER_ents)
    except:
            print("Problemas con Species")

    try:
        #Procedures extraction
        PRO_model_id = "ajtamayoh/Procedures_Identification_RoBERTa_fine_tuned"
        api_token = "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" # get yours at hf.co/settings/tokens
        PRO_entities = query(s_original, PRO_model_id, api_token)
        PRO_ents_ld = grouping_entities(PRO_entities)
        for et in PRO_ents_ld:
            PRO_ents.append(et["word"][1:])
            texto = texto.replace(et["word"][1:], '<button class="btn btn-success">'+et["word"][1:]+'</button>')
            #print(PRO_ents)
    except:
        print("Problemas con Procedures")

    #print("EL TEXTO: ", texto)

    context = {
        'msg': texto,
        'Diseases': SDN_ents,
        'Procedures': PRO_ents,
        'Organisms': LNER_ents,
        'Neg_unc': NegNER_sco,
        'Unc': Unc_sco,
    }
    
    return render(request, 'simple/index.html', context)

def acerca_de_simple(request):

    return render(request, 'simple/acerca_de_simple.html')

def historias_clinicas(request):

    return render(request, 'simple/historias_clinicas.html')

def creditos(request):

    return render(request, 'simple/creditos.html')

def agradecimientos(request):

    return render(request, 'simple/agradecimientos.html')
