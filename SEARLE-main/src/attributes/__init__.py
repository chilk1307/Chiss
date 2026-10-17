"""
Attribute verification (plan 02): parse the visual attributes requested by a relative caption, measure them on the
images with classical image processing, and re-rank the top-K candidates of a CLIP-based method.

    parser.py    caption -> constraints {family, value, negated}
    taxonomy.py  caption -> modification categories (colour, count, spatial...) for the per-category analyses
    meters.py    image -> attribute measurements (dominant colours, L*, colourfulness, pattern, object size)
    rerank.py    re-ranking of the top-K with the satisfaction of the constraints
"""
