import re
import os

# Import local NLTK data
from stopwords_words import stopwords_data, words_data
from WordPunctTokenizer import tokenizer
from BigramCollocationFinder import BigramCollocationFinder, BigramAssocMeasures
from pos_tag import pos_tag

cachedStopWords = set(stopwords_data)
cachedWords = set(words_data)

def remove_stopwords(text, language='english'):
    pattern = re.compile(r'\b(' + r'|'.join(cachedStopWords if language != 'english' else stopwords_data) + r')\b\s*')
    return pattern.sub('', text).strip()

def has_numbers(s):
    return any(char.isdigit() for char in s)

def has_special(s):
    return True if re.match(r'^\w+$', s) else False

def is_name(token):
    try:
        if has_special(token):
            return False
        if has_numbers(token):
            return False
        return pos_tag([token])[0][1] == 'NNP'
    except UnicodeDecodeError:
        return False

def is_valid_token(token):
    return token in cachedWords

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_bigrams(text):
    text = remove_stopwords(text)
    tokens = [token for token in set(tokenizer.tokenize(text)) if not is_number(token) and (is_valid_token(token) or is_name(token))]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.dice, 500)
    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)
    result = [x.lower() for x in tokens if x not in stopwords_data and len(x) > 3]
    return result

def strip_signature(text, sender=None):
    signature_starts = [u'Best Regards,', u'Best regards,', u'Warm Regards,' u'Regards,', u'Regards', u'Best Wishes,', u'Thanks,', u'Hope this helps,', u'Cheers,', u'Thank  You.', u'Sincerely,', u'Gratefully,', u'Thank you,', u'Thank you!', u'Thanking you,', u'Thanks and regards,', u'Thanks', u'Thanks & Regards,', u'Thanks & regards,']
    for sig in signature_starts:
        text = text.rsplit(sig)[0]
    text = text.rsplit('(From:).+(\r\n |\n |\\s)(Sent:).+(\r\n |\\s |\n)(To:).+(\r\n |\\s |\n)')[0]
    text = text.rsplit('(From:).+(\r\n |\n |\\s)(Sent:).+(\r\n |\\s |\n)(Subject:).+(\r\n |\\s |\n)')[0]
    text = text.rsplit('(From:).+(\r\n |\n |\\s)(Date:).+(\r\n |\\s |\n)(Subject:).+(\r\n |\\s |\n)')[0]
    split = text.rsplit(sender)
    return split[0] if sender is not None else text
