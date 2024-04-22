from googletrans import Translator




def trans(data):
    '''
    data is list of Document
    each Document has page_content and metadata
    pagecontent has the content in that page
    metadata has the source pdf and all
    '''
    translations = {}
    translator = Translator()
    for document in data:
        hindi_text = document.page_content
        source = document.metadata['source']
        source = source.split('\\')[1].strip()
        page = document.metadata['page']
        translated = translator.translate(hindi_text, src='hi', dest='en')
        english_text = translated.text
        if source not in translations:
            translations[source] = {}
        if page not in translations[source]:
            translations[source][page] = []
        translations[source][page].append({
            'hindi_text': hindi_text,
            'english_text': english_text})
        document.page_content = english_text
    return translations, data

    