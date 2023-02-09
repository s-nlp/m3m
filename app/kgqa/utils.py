from typing import Optional
from pywikidata import Entity

def label_to_entity_idx(label: str) -> Optional[str]:
    try:
        entities = [e.idx for e in Entity.from_label(label)]
    except:
        entities = None

    if entities is None:
        try:
            entities = [e.idx for e in Entity.from_label(label.capitalize())]
        except:
            entities = None
    
    if entities is None:
        try:
            entities = [e.idx for e in Entity.from_label(label.title())]
        except:
            entities = None
    
    if entities is None:
        return None

    idx = list(sorted(entities, key=lambda idx: int(idx[1:])))[0]
    return idx
