import spacy
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

def custom_tokenizer(nlp : Language):
    """
    Augments the current tokenizer used by the spacy.Language object 
    with a custom tokenizer that's used by the dataset creator
    """

    # defining the infixes here so that it's more visible
    custom_infixes = [
        r'-+',
        r'—+',
        r'\)'
    ]

    infixes = nlp.Defaults.infixes
    for infix in custom_infixes:
        infixes = infixes + infix # type: ignore
    
    infix_re = compile_infix_regex(infixes) # type : ignore
    return Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match
    )