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

#Cargar el modelo en local no permite arrancar el servidor de pruebas
#from transformers import pipeline
#model_checkpoint_neg = "ajtamayoh/Negation_and_Uncertainty_Scope_Detection_RoBERTa_fine_tuned"
#token_classifier_neg = pipeline("token-classification", model=model_checkpoint_neg, aggregation_strategy="simple")

#Lo siguiente usa la API de google para text to speech
#from gtts import gTTS
#from playsound import playsound

#Lo siguiente permite text to speech sin conexión a internet
import pyttsx3  

#Modelo para negation scope detection con CRF
#modelo_neg = load('simple\modelos\Modelo_entrenado.joblib')

#from django.http import HttpResponse
#from django.template import loader

nlp = spacy.load("es_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42

Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

def features(token, is_cue, ancestros, rights, is_sent_start, subtree_str):
  dic = {}
  dic.update ({'text': token.text,
                  'tag': token.tag_,
                  'lemma': token.lemma_,
                  'is_cue': is_cue, 
                  'ancestors': ancestros,
                  'rights': rights,
                  'head': str(token.head),
                  'r_edge': str(token.right_edge),
                  'l_edge': str(token.left_edge),
                  'dep': token.dep_,
                  'idx': token.i,  
                  'is_punct': token.is_punct,
                  'is_sent_start': is_sent_start,
                  'subtree': subtree_str,
                 #'subtree': subtree,
                 #'vector': embedding,
                 #'tensor': tensor,
                 #'vector_norm': token.vector_norm,  #Empeora los resultados       
                 #'lex_id': token.lex_id,   #No aporta información discriminante.
                 #'rank': token.rank,       #No aporta información discriminante.
                 #'cluster': token.cluster, #No aporta información discriminante.
                 
                })
  
  return dic

def representation4CRF(data, lan, negations_list):
  import spacy
  import re
  
  if lan=='ES':
    import es_core_news_sm 
    #import es_core_news_md 
    nlp = es_core_news_sm.load()
    #nlp = es_core_news_md.load()
  elif lan=='EN':
    nlp = spacy.load('en')

  X = []    #Database for CRF model
  X_text = []
  X_POS = []
  for frase in data:
    frases = []
    POS = []

    X_text.append(frase)
    frase = re.sub('-','_',frase)
    frase = re.sub('&quot;','"',frase)
    frase = re.sub(':_s','EMOJI',frase)
    frase = re.sub('\*\*\*\*','*',frase)
    frase = re.sub("([A-Za-z]+)'([A-Za-z]+)",'\1_\2',frase)
    frase = re.sub('     ',' _ _ _ _',frase)
    frase = re.sub(r'([0-9]+)([A-Za-z]+)','\1_\2',frase)
    frase = re.sub(r'([A-Za-z]+),([0-9]+)','\1_\2',frase)
    frase = re.sub(';_\)','EMOJI',frase)
    frase = re.sub('p\.d\.','Posdata',frase)
    frase = re.sub('\*\*\*','asterisk',frase)
    frase = re.sub(r'([0-9]+)\.',r'\1_\.',frase)
    frase = re.sub(r'&amp;',r'ampersand',frase)
    
    doc = nlp(frase)
    
    for token in doc:
      
      POS.append(token.tag_)

      dic = {}
      subtree = []
      subtree_str = ""
      sub = token.subtree
      for t in sub:
        subtree.append(t)
        subtree_str += str(t) + ' '
      subtree_str[:-1]
      
      if token.is_sent_start == None:
        is_sent_start = False
      else:
        is_sent_start = True


      rights = ""
      rg = token.rights
      for r in rg:
        rights += str(r) + ' '
      rights=rights[:-1]


      ##################################################################
      #Transformando la representación de word embedding a un string
      emb = token.vector
      emb_list = emb.tolist()
      embedding = ""
      emb = 0
      for v in emb_list:
        embedding += str(v) + " "
        #emb+=v
      #emb = emb/len(emb_list)

      #Transformando la representación de tensor a un string
      ten = token.tensor
      ten_list = ten.tolist()
      tensor = ""
      for t in ten_list:
        tensor += str(t) + " "
      ##################################################################

      #Usando los embeddings de BERT preentrenados
      #r = bert_embedding([str(token)])
      #embedding_bert = r[0][1][0]

      ##################################################################

      ances = token.ancestors
      ancestros = ""
      for anc in ances:
        ancestros+=str(anc)+" "
      ancestros = ancestros[:-1]

      if token.text in negations_list:
        is_cue = True
      else:
        is_cue=False
        
      dic = features(token, is_cue, ancestros, rights, is_sent_start, subtree_str)
      

      frases.append(dic)

    X.append(frases)
    X_POS.append(POS)
  return X, X_POS


def formatOutput(s, POStags, neg):
    if s[-2]==",":
        s = s[:-2]   
    if s[-1] == 'así':
        s = s[:-1]
    if s[0] == ",":
        s = s[1:]
    if s[-1] == ',':
        s = s[:-1]
    if (POStags[-2] == 'ADV' and POStags[-1] == 'AUX') or (POStags[-2] == 'AUX' and POStags[-1] == 'ADV'):
        s = s[:-2]
        POStags = POStags[:-2]
    if POStags[-1] == "VERB" and neg==False:
        s = s[:-1]
        POStags = POStags[:-1]
    if s[-1].lower() == 'se':
        s = s[:-1]
    if POStags[-1] == "CCONJ":
        s = s[:-1]
        POStags = POStags[:-1]    
    if POStags[-1] == "ADP":
        s = s[:-1]
        POStags = POStags[:-1]
    return s

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
            if (e['entity_group'] == "B" or e['entity_group'] == "I") and (e['start'] == (output[-2]['end']+1)):
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

###################################################################
#Methods for socialdisner
def delete_accents(s):
  l = [('á', 'a'), ('é','e'), ('í','i'), ('ó','o'), ('ú','u')]
  for v in l:
    s = s.replace(v[0],v[1])
  return s

def post_processing_ents(hc, pred_grouped):
    """
    @params:
        hc: clinical case
        pred_grouped: the model's output with processed subwords.
    """
    import re

    list_output = pred_grouped.copy()

    lista_spans = []
    offs = []

    for p, p_out in zip(pred_grouped, list_output):

      off0 = int(p['start'])
      off1 = int(p['end'])
      span = p['word']

      span = re.sub("^, |^,|^\. |^\.|^: |^:|^; |^;|^\( |^\(|^\) |^\)","",span)

      if "\n" in span:
        span = re.sub("\n"," ",span)

      if " - " in span:
        span = re.sub(" - ","-",span)

      if "- " in span:
        span = re.sub("- ","-",span)

      if " -" in span:
        span = re.sub(" -","-",span)

      if "( " in span:
        span = re.sub("\( ","(",span)

      if " )" in span:
        span = re.sub(" \)",")",span)

      if span.endswith(" y") :
        span = span[:-2]

      if span.endswith(" de") or span.endswith(" en"):
        span = span[:-3]

      if span.endswith(" por") or span.endswith(" con") or span.endswith(" del"):
        span = span[:-4]

      if span.endswith(".") or span.endswith(",") or span.endswith(";") or span.endswith(":") or span.endswith("–") or span.endswith("-") or span.endswith("!"):
        span = span[:-1]

      if span.endswith(" .") or span.endswith(" ,") or span.endswith(" ;") or span.endswith(" :") or span.endswith(" –") or span.endswith(" -"):
        span = span[:-2]

      if span.startswith("#"):
        span = span[1:]

      if span.startswith("🔸 "):
        span = span[2:]

      pattern = r"^[a-z|á|é|í|ó|ú|/]{0,2}$|^[0-9]+$|^[A-Z]$|^#$| a$| el$| la$|^🔸$"
      match = re.findall(pattern, span)
      if len(match) > 0:
        continue

      if span not in lista_spans:
        #For multiword spans
        mwspan = "".join(span.split())
        if span != mwspan:
          spans = [span, mwspan]
        else:
          spans = [span]
        # Find all indices of 'span'
        for sp in spans:
          indices = [index for index in range(len(hc)) if delete_accents(hc.lower()).startswith(delete_accents(sp.lower()), index)]
          for ind in indices:
            off0 = ind
            off1 = ind+len(sp)
            extraction = hc[off0:off1]
            match = re.findall(pattern, extraction)
            no_subsumed = True
            for of in offs:
              if off0 == of[0] and off1 < of[1]:
                no_subsumed = False
            output = fname[:-4]+"\t"+str(off0)+"\t"+str(off1)+"\t"+"ENFERMEDAD"+"\t"+extraction+"\n"
            if len(match) == 0 and no_subsumed:
              offs.append((off0,off1))
              p_out.update({"word": extraction})

          lista_spans.append(sp)
    
    return list_output



##################################################################


def index(request):

    texto = ""
    if request.method == "POST":
        if request.POST['texto1'] != "":
            texto = request.POST['texto1']
        else:
            texto = request.FILES['texto2'].read().decode('utf-8')
            #with open(request.FILES["texto2"], 'r', encoding="UTF-8") as f:
                #texto = f.read()
            print(texto)

    #Omitir las comas
    #texto = re.sub(",", "", texto)
    
    #Eliminar saltos de linea y doble espacio
    texto = re.sub(r'\n+', ' ' ,texto)
    texto = re.sub(r'\s+', ' ', texto)


    doc = nlp(texto)
    language = doc._.language
    lg = ""
    if language['language'] == "es":
        doc = nlp(texto)
        lg = "es"
    elif language['language'] == "en":
        doc = nlp_en(texto)
        lg = "en"
    
    #Part-of-Speech tagging
    pos = []
    for token in doc:
        #pos.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop])
        pos.append([token.text, token.lemma_, token.pos_])
    

    #Buscar patrones que combinene (NP o NC) ADJ ADV PREP (a, por, de) en cualquier orden 
    info_general = []
    motivo_consulta = []
    hallazgos_presentes = []
    hallazgos_ausentes = []
    procedimientos_presentes = []
    procedimientos_ausentes = []
    tratamientos_presentes = []
    tratamientos_ausentes = []
    examenes = []
    diagnostico = []
    ADR_presentes = []
    ADR_ausentes = []
    antecedentes_presentes = []
    antecedentes_ausentes = []
    SDN_ents = []
    LNER_ents = []
    PRO_ents = []
    NegNER_sco = []
    gen_ents = []
    responses_zeroshot = []

    tags = ["NOUN", "PROPN", "ADJ", "ADV", "ADP", "NUM", "VERB", "CCONJ", "AUX"]
    #preps = ["a", "por", "de", "sin", 'apenas', 'nada', 'a_falta_de', 'ninguno', 'no_solo', 'no', 'ni', 'ninguna', 'ningún', 'negativo', 'negativos', 'negativa', 'negativas', "ausencia"]
    negations_list_es = ['apenas', 'nada', 'sin', 'a_falta_de', 'ninguno', 'no_solo', 'no', 'ni', 'ninguna', 'ningún', 'negativo', 'negativos', 'negativa', 'negativas', "ausencia"]
    motivos = ["consultar", "consulta", "acudir", "visitar", "ingresar", "ingresado", "sufrir", "remitir", "remitido", "admitir", "admitido", "referir", "refeer", "relatar"]
    POStags = []
    s = []
    #prp = False
    neg = False
    motivo = False
    procto = False
    trata = False
    exm = False
    diagn = False

    cuenta_pattern = 0
    for p in pos:
        print(f"{p[0]}\t{p[1]}\t{p[2]}")

        # Cuando encuentre una estos no rompe el sintagma
        if p[0]==',': 
            s.append(",")
            continue
        #El punto genera mucho ruido
        #if p[0]=='.': 
        #    s.append(".")
        #    continue
        if p[0]=='"':
            s.append('"')
            continue
        if p[0]=='(':
            s.append('(')
            continue
        if p[0]==')':
            s.append(')')
            continue
        if p[0]=='/':
            s.append('/')
            continue
        if p[0]==':':
            s.append(':')
            continue
        if p[2]=='PRON' and p[0].lower() == "se":
            s.append(p[0])
            continue
        #El PRON que genera mucho ruido
        #if p[2]=='PRON' and p[0].lower() == "que":
        #    s.append(p[0])
        #    continue
        if p[2]=='DET':
            continue

        #Valida casos especiales donde no se debe tomar como analítica. ... diferentes : 1. bla 2. bla
        if exm == True and POStags[-1]=='NUM' and p[0]=='.':
            exm = False

        if p[2] in tags:
            s.append(p[0])
            POStags.append(p[2])
        #    if p[0] in preps:
        #        prp = True
            
            #Para los indicadores de negación
            if p[0].lower() in negations_list_es or p[0].lower().startswith('descarta'):
                neg = True
            
            #Se omite sin embargo de los indicadores de negación
            if p[0].lower() == 'embargo' and s[-2].lower() == 'sin':
                neg = False
            
            #Para los motivos de consulta
            if p[1].lower() in motivos:
                motivo = True
            
            #Para procedimientos (stemming con "realiz" y con "proce")
            if p[0].lower().startswith("realiz") or p[0].lower().startswith("proce") or p[0].lower().startswith("coloc") or (p[0].lower().startswith("practic") and p[2]=="VERB"):
                #if ((p[0].lower().endswith('ico') or p[0].lower().endswith('ica')) and p[2] == 'ADJ') or ((p[0].lower().endswith('ía') or p[0].lower().endswith('ío')) and p[2] == 'NOUN') or ((p[0].lower().endswith('ia') or p[0].lower().endswith('io')) and p[2] == 'NOUN'):
                #El anterior condicional trae muchos falsos positivos solo con revisar esta parte morfológica, se debe pulir.
                procto = True
            
            #Para tratamientos (stemming con "trat")
            try:
                if p[0].lower().startswith("trat") and s[-2].lower() != 'de':
                    trata = True
            except:
                pass
            
            try:
                if trata == True and s[-2].lower().startswith("trat") and (p[0].lower() == 'de' or p[0].lower() == 'del'):
                    trata = False
            except:
                pass

            #Para diagnóstico
            try:
                if p[0].lower().startswith("diagn") and POStags[-2] != 'ADP':
                    diagn = True
            except:
                pass
            
            #Para analítica
            try:
                if p[2] == 'NUM' and s[-2]==":":
                    exm = True
            except:
                pass
        else:
            if (len(s) > 2):
                s = formatOutput(s, POStags, neg)
                s_original = s
                if len(s) > 2:
                    
                    try:
                        #Disease extraction
                        #SDN_model_id = "ajtamayoh/NLP-CIC-WFU_SocialDisNER_fine_tuned_NER_EHR_Spanish_model_Mulitlingual_BERT_v2" #SocialDisNER
                        SDN_model_id = "ajtamayoh/Disease_Identification_RoBERTa_fine_tuned"
                        api_token = "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" # get yours at hf.co/settings/tokens
                        SDN_entities = query(" ".join(s), SDN_model_id, api_token)
                        SDN_ents_ld = grouping_entities(SDN_entities)
                        #SDN_ents_ld_post = post_processing_ents(" ".join(s), SDN_ents_ld) #Se debe ajustar el método de post-processing, ejecuta con error 500 desde la GUI.
                        for et in SDN_ents_ld:
                            SDN_ents.append(et["word"])
                            s = " ".join(s).replace(et["word"], "<span style='background-color: yellow'>"+et["word"]+"</span>").split()
                        print(SDN_ents)
                    except:
                        print("Problemas con Diseases")

                    try:
                        #Species extraction
                        #LNER_model_id = "ajtamayoh/NLP-CIC-WFU_Clinical_Cases_NER_Sents_tokenized_mBERT_cased_fine_tuned" #LivingNER
                        LNER_model_id = "ajtamayoh/Species_Identification_RoBERTa_fine_tuned"
                        api_token = "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" # get yours at hf.co/settings/tokens
                        LNER_entities = query(" ".join(s_original), LNER_model_id, api_token)
                        LNER_ents_ld = grouping_entities(LNER_entities)
                        for et in LNER_ents_ld:
                            LNER_ents.append(et["word"])
                            s = " ".join(s).replace(et["word"], "<span style='background-color: aqua'>"+et["word"]+"</span>").split()
                        print(LNER_ents)
                    except:
                        print("Problemas con Species")

                    try:
                        #Procedures extraction
                        PRO_model_id = "ajtamayoh/Procedures_Identification_RoBERTa_fine_tuned"
                        api_token = "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" # get yours at hf.co/settings/tokens
                        PRO_entities = query(" ".join(s_original), PRO_model_id, api_token)
                        PRO_ents_ld = grouping_entities(PRO_entities)
                        for et in PRO_ents_ld:
                            PRO_ents.append(et["word"])
                            s = " ".join(s).replace(et["word"], "<span style='background-color: green'>"+et["word"]+"</span>").split()
                        print(PRO_ents)
                    except:
                        print("Problemas con Procedures")

                    try:
                        #Negation and Uncertainty Scope Detection
                        NegNER_model_id = "ajtamayoh/Negation_and_Uncertainty_Scope_Detection_RoBERTa_fine_tuned"
                        api_token = "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" # get yours at hf.co/settings/tokens
                        NegNER_scopes = query(" ".join(s_original), NegNER_model_id, api_token)
                        #NegNER_scopes = token_classifier_neg(" ".join(s_original))
                        NegNER_sco_ld = grouping_entities(NegNER_scopes)
                        for ns in NegNER_sco_ld:
                            NegNER_sco.append(ns["word"])
                            s = " ".join(s).replace(" "+ns["word"]+" ", " <span style='background-color: lightpink'>"+ns["word"]+"</span> ").split()
                        print(NegNER_sco)
                        print(NegNER_sco_ld)
                    except:
                        print("Problemas con Negation")

                    """
                    try:
                        #General Entities Extraction with the eHealth model
                       
                        Gen_Ent_model_id = "ajtamayoh/NER_ehealth_Spanish_mBERT_fine_tuned"
                        api_token = "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" # get yours at hf.co/settings/tokens
                        gen_entities = query(" ".join(s), Gen_Ent_model_id, api_token)
                        gen_ents_ld = grouping_entities(gen_entities)
                        for et in gen_ents_ld:
                            gen_ents.append(et["word"])
                        print(gen_ents)
                    except:
                        print("Problemas con eHealth")
                    """    

                    #For zero-shor model
                    """ 
                    inference = InferenceApi(repo_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", token=api_token)
                    inputs = " ".join(s)
                    params = {"candidate_labels": ["Información general", "Motivo de consulta", "hallazgos, diagnóstico", "tratamientos", "procedimientos", "analítica"]} 
                    response_zs = inference(inputs, params)
                    #responses_zeroshot.append(response_zs['sequence'])
                    print(response_zs)
                    """


                    if cuenta_pattern == 0:
                        info_general.append(" ".join(s))
                    elif cuenta_pattern >= 1 and motivo==True:
                        motivo_consulta.append(" ".join(s))
                    else:
                        #Resultados de exámenes
                        if exm == True and trata == False:
                            examenes.append(" ".join(s))
                        #diagnóstico
                        elif diagn == True and trata == False:
                            diagnostico.append(" ".join(s))
                        #Se sigue la lógica de que primero se enuncia el diagnóstico y luego el tratamiento. Esto puede
                        #tener excepciones
                        elif diagn == True and trata == True:
                            j = -1
                            for p in s:
                                j+=1
                                if p.lower().startswith('trat'):
                                    try:
                                        dg = s[:j-1] #se toma el tratamiento un token antes de encontrar el stem 'trata'
                                        tt = s[j-1:]
                                        dg = formatOutput(dg,POStags[:j-1],neg)
                                        tt = formatOutput(tt,POStags[j-1:],neg)
                                        diagnostico.append(" ".join(dg))
                                        if neg==False:
                                            tratamientos_presentes.append(" ".join(tt))
                                        else:
                                            tratamientos_ausentes.append(" ".join(tt))
                                    except:
                                        if neg==False:
                                            tratamientos_presentes.append(" ".join(s))
                                        else:
                                            tratamientos_ausentes.append(" ".join(s))
                                        break                             
                        else:    
                            if neg == False:   
                                if trata == True:
                                    tratamientos_presentes.append(" ".join(s))
                                elif procto == True:
                                    procedimientos_presentes.append(" ".join(s))
                                else:
                                    hallazgos_presentes.append(" ".join(s))
                            else:
                                if trata == True:
                                    tratamientos_ausentes.append(" ".join(s))
                                elif procto == True:
                                    procedimientos_ausentes.append(" ".join(s))
                                else:
                                    hallazgos_ausentes.append(" ".join(s))
                        
                cuenta_pattern+=1
            else:
                print(" ".join(s))
                pass

            #prp=False
            s = []
            POStags = []
            neg = False
            motivo = False
            procto = False
            trata = False
            exm = False
            diagn = False


    #NER
    entidades = []
    for ent in doc.ents:
        entidades.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    
    #Para eliminar valores repetidos en las listas
    info_general = list(set(info_general))
    motivo_consulta = list(set(motivo_consulta))
    hallazgos_presentes = list(set(hallazgos_presentes))
    hallazgos_ausentes = list(set(hallazgos_ausentes))
    procedimientos_presentes = list(set(procedimientos_presentes))
    procedimientos_ausentes = list(set(procedimientos_ausentes))
    tratamientos_presentes = list(set(tratamientos_presentes))
    tratamientos_ausentes = list(set(tratamientos_ausentes))
    examenes = sorted(list(set(examenes)))
    diagnostico = list(set(diagnostico))
    


    #Audios con API de google | Muy lenta
    #if request.method == "POST":
    #    s = gTTS("Información general. " + " ".join(info_general), lang="es-us")
    #    s.save("simple/static/simple/Audios/info_general.mp3")
    #    playsound('simple/static/simple/Audios/info_general.mp3')

    """
    #Audios con pyttsx3
    #Información general
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Información general , " + " , ".join(info_general) , "simple/static/simple/Audios/info_general.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait() 

    #Motivos de consulta
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Motivo de consulta , " + ", ".join(motivo_consulta) , "simple/static/simple/Audios/motivo_consulta.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Hallazgos descartados
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Hallazgos descartados , " + ", ".join(hallazgos_ausentes) , "simple/static/simple/Audios/hallazgos_ausentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Hallazgos observados
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Hallazgos observados , " + ", ".join(hallazgos_presentes) , "simple/static/simple/Audios/hallazgos_presentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Analítica
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Analítica , " + ", ".join(examenes) , "simple/static/simple/Audios/examenes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Procedimientos descartados
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Procedimientos descartados , " + ", ".join(procedimientos_ausentes) , "simple/static/simple/Audios/procedimientos_ausentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Procedimientos realizados
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Procedimientos realizados , " + ", ".join(procedimientos_presentes) , "simple/static/simple/Audios/procedimientos_presentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Tratamientos sin efecto positivo
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Tratamientos sin efecto positivo , " + ", ".join(tratamientos_ausentes) , "simple/static/simple/Audios/tratamientos_ausentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Tratamientos realizados
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Tratamientos con efecto positivo , " + ", ".join(tratamientos_presentes) , "simple/static/simple/Audios/tratamientos_presentes.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()

    #Diagnóstico
    s = pyttsx3.init()    
    rate = s.getProperty('rate')
    s.setProperty('rate', rate-10)
    s.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    s.save_to_file("Diagnóstico , " + ", ".join(diagnostico) , "simple/static/simple/Audios/diagnostico.mp3")
    #s.say("Información general." + " ".join(info_general))
    s.runAndWait()
    """


    context = {
        #'msg': pos,
        'entidades': entidades,
        'info_general': info_general,
        'motivo_consulta': motivo_consulta,
        'hallazgos_presentes': hallazgos_presentes,
        'hallazgos_ausentes': hallazgos_ausentes,
        'procedimientos_presentes': procedimientos_presentes,
        'procedimientos_ausentes': procedimientos_ausentes,
        'tratamientos_presentes': tratamientos_presentes,
        'tratamientos_ausentes': tratamientos_ausentes,
        'examenes': examenes,
        'diagnostico': diagnostico,
        'sdn_ents': SDN_ents,
        'lner_ents': LNER_ents,
        'pro_ents': PRO_ents,
        'negner_sco': NegNER_sco,
        'gen_ents': gen_ents,
        #'responses_zeroshot': responses_zeroshot
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
