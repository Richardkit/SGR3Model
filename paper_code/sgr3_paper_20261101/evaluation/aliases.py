def lexical(value):
    return ' '.join(value.strip().lower().replace('_', ' ').replace('-', ' ').split())
