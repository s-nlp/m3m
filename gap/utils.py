entity_status_linker = {
    "INTERNAL": 0,
    "QUESTIONS_ENTITY": 1,
    "ANSWER_CANDIDATE_ENTITY": 2
}


def get_webnlg_like_mapper(triplets):
    webnlg_mapper = {}
    nodes = {}
    for triplet in triplets:
        nodes[triplet.source_entity.id] = triplet.source_entity
        nodes[triplet.target_entity.id] = triplet.target_entity

    for node in nodes.values():
        if node.label is None:
            node_label = "unknown entity"
        else:
            node_label = node.label
        webnlg_mapper[node.id] = {"source_label": node_label, "type": node.type}
    for triplet in triplets:
        link_dict = {}
        link_dict["target_label"] = webnlg_mapper[triplet.target_entity.id]["source_label"]
        link_dict["relation_label"] = triplet.relation
        links_array = webnlg_mapper[triplet.source_entity.id].get('links_array', -1)
        if links_array == -1:
            webnlg_mapper[triplet.source_entity.id]['links_array'] = [link_dict]
        else:
            webnlg_mapper[triplet.source_entity.id]['links_array'].append(link_dict)
    return webnlg_mapper


def get_json_format(webnlg_mapper):
    json_converet = {}
    ind = 0
    for j, indx in enumerate(webnlg_mapper):
        source_label = webnlg_mapper[indx].get('source_label', -1)
        entity_type = entity_status_linker[webnlg_mapper[indx]['type']]
        links_array = webnlg_mapper[indx].get('links_array', -1)
        if links_array == -1:
            pass
        else:
            for link_dict in links_array:
                target_label = link_dict['target_label']
                relation_label = link_dict['relation_label']
                json_converet[f'W{ind}'] = [source_label, source_label, [[relation_label, target_label]], entity_type]
                ind += 1
    return json_converet


def convert_to_webnlg_format(triplets):
    webnlg_format = {}
    webnlg_mapper = get_webnlg_like_mapper(triplets)
    json_converet = get_json_format(webnlg_mapper)
    webnlg_format["id"] = 1
    webnlg_format["kbs"] = json_converet
    webnlg_format["text"] = ["example of text"]

    return webnlg_format
