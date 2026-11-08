from src.citations import find_invalid_citations, extract_available_pages

context = "[page 42] ... [page 43] ..."
draft = "See [page 42] and also [page 99]."
print(extract_available_pages(context))        # [42, 43]
print(find_invalid_citations(draft, [42, 43])) # [99]