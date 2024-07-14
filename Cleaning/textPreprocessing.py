# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:47:57 2024

@author: danie
"""
import re
import emoji
import unidecode

#%% Constants
# *****************************************************************************
# Cleaning modes
nSW:  list = [1,0,0,1,0,0,0] # No Stopwords
nSWP: list = [1,0,0,1,0,1,0] # No Stopwords
FC:   list = [1,1,1,1,1,1,1] # Full cleaning

STOPWORDS_PT: list = []
with open('./Stopwords/StopWordsPT.txt', 'r', encoding='utf-8') as f:
    STOPWORDS_PT = [line.rstrip() for line in f]
STOPWORDS_PTU: list = []
with open('./Stopwords/StopWordsPTU.txt', 'r', encoding='utf-8') as f:
    STOPWORDS_PTU = [line.rstrip() for line in f]

dictHTML: dict = {
        'À': "&Agrave;", 'Á': "&Aacute;", 'Â': "&Acirc;",  'Ã': "&Atilde;", 'Ä': "&Auml;", 'Å': "&Aring;",
        'à': "&agrave;", 'á': "&aacute;", 'â': "&acirc;",  'ã': "&atilde;", 'ä': "&auml;", 'å': "&aring;",
        'Æ': "&AElig;",  'æ': "&aelig;",  'ß': "&szlig;",  'Ç': "&Ccedil;", 'ç': "&ccedil;",
        'È': "&Egrave;", 'É': "&Eacute;", 'Ê': "&Ecirc;",  'Ë': "&Euml;",
        'è': "&egrave;", 'é': "&eacute;", 'ê': "&ecirc;",  'ë': "&euml;",   'ƒ': "&#131;",
        'Ì': "&Igrave;", 'Í': "&Iacute;", 'Î': "&Icirc;",  'Ï': "&Iuml;",   'ì': "&igrave;", 
        'í': "&iacute;", 'î': "&icirc;",  'ï': "&iuml;",   'Ñ': "&Ntilde;", 'ñ': "&ntilde;",
        'Ò': "&Ograve;", 'Ó': "&Oacute;", 'Ô': "&Ocirc;",  'Õ': "&Otilde;", 'Ö': "&Ouml;",
        'ò': "&ograve;", 'ó': "&oacute;", 'ô': "&ocirc;",  'õ': "&otilde;", 'ö': "&ouml;",
        'Ø': "&Oslash;", 'ø': "&oslash;", 'Œ': "&#140;",   'œ': "&#156;",   'Š': "&#138;", 'š': "&#154;",
        'Ù': "&Ugrave;", 'Ú': "&Uacute;", 'Û': "&Ucirc;",  'Ü': "&Uuml;",
        'ù': "&ugrave;", 'ú': "&uacute;", 'û': "&ucirc;",  'ü': "&uuml;",   'µ': "&#181;", '×': "&#215;",
        'Ý': "&Yacute;", 'Ÿ': "&#159;",   'ý': "&yacute;", 'ÿ': "&yuml;",
        '°': "&#176;",   '†': "&#134;",   '‡': "&#135;",   '<': "&lt;",     '>': "&gt;", '±': "&#177;",
        '«': "&#171;",   '»': "&#187;",   '¿': "&#191;",   '¡': "&#161;",   '·': "&#183;", '•': "&#149;",
        '™': "&#153;",   '©': "&copy;",   '®': "&reg;",    '§': "&#167;",   '¶': "&#182;",
        ' ': "&nbsp;",   '€': "&euro;",
}
dictHTML = {v:k for k,v in dictHTML.items()}
# *****************************************************************************
#%% Functions
# *****************************************************************************
def remove_simbols(text: str):
    # Convert html special characters to letters
    text = text.replace('&amp;','&') # fix '&' format
    for code in dictHTML:
        text = text.replace(code, dictHTML[code])  
    
    # remove particular simbols
    text = text.replace('$%', '') # expecific symbol of dataset
    text = text.replace('º', '') # symbol not removed by isalpha()
    text = text.replace('°', '') # symbol not removed by isalpha()
    text = text.replace('ª', '') # symbol not removed by isalpha()
    # text = re.sub(r'([a-zA-Z])-([a-zA-Z])', r'\1\2', text)
    # text = re.sub(r'([a-zA-Z])-([a-zA-Z])', r'\1\2', text)
    text = text.replace('_', ' ') # make sure separate words
    text = text.replace('-', ' ') # make sure separate words
    text = text.replace('#', '')
    text = re.sub(r'([-:/\\\\(){}!?"#*+,&\'.])\1+', r'\1', text)
    text = re.sub("{uid:[0-9]+}", " ", text) # Eliminate a tag appears in reviews
    return text

def remove_less2(text: str):
    res = ''
    for w in text.split():
        if len(w)>2:
            res += ' ' + w
        elif not w.isalpha():
            res += ' ' + w
    return res

def remove_stopwords(text: str, st: list):
    filtered_text = []
    for word in separate_punc(text):
        if word.lower() not in st:
            filtered_text.append(word)
    return remove_less2(' '.join(filtered_text))

def separate_punc(text: str):
    return "".join((char if char.isalnum() else f" {char} ") for char in text).split()

def join_bigram(text: str):
    
    prefixes = ['nao','não','sem','muito','muita','muitos','muitas']
    text = separate_punc(text)
    jump = False
    result = []
    for i in range(len(text)):
        if jump: 
            jump = False
            continue
        bigram = text[i:i+2] # select next possible bigram
        
        if len(bigram) < 2: # If not actually bigram
            result.append(bigram[0])
        else:
            if bigram[0] in prefixes and bigram[1].isalpha():
                result.append('_'.join(bigram))
                jump = True
            else:
                result.append(bigram[0])
                
    return ' '.join(result)

def remove_nonalpha(text: str):
    # check character per character: (keep whitespaces and underscores)
    filtered_text = map(lambda x: x if x.isalnum() or x==' ' or x=='_' else ' ', text)
    filtered_text = ''.join(filtered_text)
    return remove_less2(filtered_text)

def remove_num(text: str):
    # check character per character: (keep whitespaces)
    filtered_text = map(lambda x: ' ' if x.isdigit() else x, text)
    filtered_text = ''.join(filtered_text)
    return remove_less2(filtered_text)

def clean(text: str, c1: bool, c2: bool, c3: bool, 
           c4: bool, c5: bool, c6: bool, c7: bool):
    """
    c1: Remove accents (standardization)
    c2: Turn to lowercase
    c3: Remove emojis
    c4: Remove Stopwords
    c5: Create bigrams
    c6: Remove punctuation
    c7: Remove numbers
    """
    # Mandatory
    text = remove_simbols(text)
    # Optional
    if c1: # Remove accents (standardization)
        text = unidecode.unidecode(text, errors='preserve')
    if c2: # Turn to lowercase
        text = text.lower()
    if c3: # Remove emojis
        text = emoji.replace_emoji(text,' ')
    if c4: # Remove Stopwords
        if c1: stopwords = STOPWORDS_PTU
        else:  stopwords = STOPWORDS_PT
        text = remove_stopwords(text, stopwords)
    if c5: # Create bigrams
        text = join_bigram(text)
    if c6: # Remove punctuation
        text = remove_nonalpha(text)
    if c7: # Remove numbers
        text = remove_num(text)
    # Mandatory
    text = " ".join(text.split())
    
    return text

def clean_tokens(text: list, c1: bool, c2: bool, c3: bool, 
                   c4: bool, c5: bool, c6: bool, c7: bool):
    """
    c1: Remove accents (standardization)
    c2: Turn to lowercase
    c3: Remove emojis
    c4: Remove Stopwords
    c5: Create bigrams
    c6: Remove punctuation
    c7: Remove numbers
    """
    # Mandatory
    text = remove_simbols(' '.join(text))
    
    # Optional
    if c1: # Remove accents (standardization)
        text = unidecode.unidecode(text, errors='preserve')
    if c2: # Turn to lowercase
        text = text.lower()
    if c3: # Remove emojis
        text = emoji.replace_emoji(text,' ')
    if c4: # Remove Stopwords
        if c1: stopwords = STOPWORDS_PTU
        else:  stopwords = STOPWORDS_PT
        text = remove_stopwords(text, stopwords)
    if c5: # Create bigrams
        text = join_bigram(text)
    if c6: # Remove punctuation
        text = remove_nonalpha(text)
    if c7: # Remove numbers
        text = remove_num(text)
    # Mandatory
    text = text.split()
    return text

# *****************************************************************************
#%% END
if __name__ == '__main__':
    text0 = "Relação qualidade/ preço fraca tendo em conta a concorrência que tem ao redor.  Não recomendo existem sítios com cerveja artesanal de muito superior qualidade."
    text1 = "Nem sei muito bem o que dizer, porque adorei tudo!! Desde o ambiente á comida, adorei o espaço! Recomendo."
    text2 = "A relação qualidade preço é fantástica, o atendimento melhorou significativamente nos últimos tempos. Até agora é o meu sitio preferido para comer Hamburgueres em Lisboa. A nota negativa vai para o pão que deveria ser tostado para não ficar tão maçudo."
    text3 = "Nova hamburgueria na Praça de Londres. O espaço não é muito grande mas salvo erro estavam a fazer obras para ampliação. Tem uma esplanada agradável que se torna uma boa opção para os dias de Sol. A qualidade e escolha dos ingredientes e cuidada, sendo a apreciação global muito positiva. Não percebi se interrompem a cozinha entre o almoço e o jantar pois algumas das vezes que passei à porta a meio da tarde pareceu-me que estavam em limpezas e fechados. A confirmar-se, julgo que é algo que terão de alterar. "
    text4 = "Preço: 3 Comida: 4 Localização :3,5 Serviço: 3 (confuso) Ambiente: 3 (barulhento no interior, calmo no exterior). "
    text5 = "Pizzas carás para a qualidade. Mas senão são boas. Entrega ao domicílio muito lenta... Não recomendo nada. Demoraram me uma hora e meia para entregar a pizza."
    text6 = "No final da Junqueira - parte 4 Quase mesmo no final da rua, este snack-bar que pelos vistos também se intitula restaurante... disso tem pouco. Pratos e mini-pratos, para ser barato, mas é só isso mesmo. Bife à casa, alheira, etc... o habitual neste tipo de sítios, sem deixar de ser banal. Novamente, umas portas ao lado há uma opção bem melhor."
    text7 = "Bem, talvez não tenha sido um dia muito feliz no restaurante. O atendimento foi simpático e profissional, mas lento muito lento. Eu cheguei atrasado 20 min relativamente aos meus colegas, fui atendido primeiro, eles ainda esperaram pela comida mais uns 25min depois de mim. Totalmente desaconselhado no inverno, fomos colocados numa mesa na rua, pois a fila de espera para o interior (espaço muito pequeno) superava as 2h (estimativa) espaço desagradável no inverno, acredito q seja agradável no verão. Quanto a comida, banal não me surpreendeu, até pelo rácio preço qualidade, não é barato, e não é fabuloso. Por ser colocado na rua, passado 15 min a comida estava gelado. Ponto favorável a localização. Ponto mais negativo, o espaço outdoor não tem as condições mínimas para acolher jantares no inverno. Não voltarei."
    text8 = "Localização agradável, com uma óptima esplanada e um menu com uma variedade agradável para o frio e calor. Dos três restaurantes colados, acabei por escolher este pela pela simpatia sem o forçar a sentar. Tudo óptimo até após terminar a minha refeição, enquanto lia o meu livro começar a ouvir uma discussão vinda de dentro de restaurante entre os donos do estabelecimento. Tiveram a capacidade de seleccionar pessoal impecável, mas infelizmente o nível desta senhora ao incomodar os seus próprios clientes, fez com que ficasse sem vontade de voltar ( sendo que é a dona ). Odeio gritaria e peixarada, mais ainda quando é suposto as pessoas estarem a relaxar e a tomar a sua refeição de forma praseirosa. "
    text9 = "Pior atendimento que já tive em telheiras. Primeiro cobraram-me 2,9 euros por cerveja e tentaram-me passar por parvo com esquemas para no fim não terem multibanco. Antes de abrirem as cervejas disseram que já tinham registado e não podiam cancelar. Se quiserem ser roubados, já sabem onde ir..."
    
    for i in range(9):
        print(
        clean(eval(f"text{i}"), *nSW),
        clean(eval(f"text{i}"), *nSWP),
        clean(eval(f"text{i}"), *FC),
        sep='\n\n')
        
        print(30*"=",'\n')














