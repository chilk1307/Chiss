"""
Query templates (plan 04, section 4.5). '$' is replaced by the pseudo token of the reference image and '{}' by the
relative caption. With several templates the normalized query features are averaged (--prompt-set ensemble).

'$' must stay a separate word: CLIP's tokenizer merges '$,' into another token, which would leave no '$' token to
replace. Hence "a photo of $ , {}" rather than "a photo of $, {}".
"""

PSEUDO_TOKEN_ID = 259  # id of '$' in the CLIP vocabulary (see encode_with_pseudo_tokens.py)

PROMPT_SETS = {
    'default': ["a photo of $ that {}"],
    'ensemble': [
        "a photo of $ that {}",
        "an image of $ that {}",
        "a picture of $ that {}",
        "$ that {}",
        "a photo of $ , {}",
        "a photo of $ which {}",
    ],
}
