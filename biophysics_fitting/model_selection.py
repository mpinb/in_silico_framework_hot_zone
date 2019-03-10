import Interface as I
from hay_evaluation import objectives_BAC, objectives_step

def get_model_pdf_from_mdb(mdb):
    def augment_pdf(pdf, i, j):
        pdf['model_id'] = '_'.join([mdb.get_id(), str(i), str(j)])
        pdf['model_id'] = pdf['model_id'] + '_' + I.pd.Series(pdf.index).astype('str')
        return pdf.set_index('model_id')
    
    out = I.defaultdict(lambda: [])
    max_i = max([int(x) for x in mdb.keys() if I.utils.convertible_to_int(x)])
    for i in range(1, max_i + 1):
        if not str(i) in mdb.keys():
            continue
        max_j =  max([int(x) for x in mdb[str(i)].keys() if I.utils.convertible_to_int(x)])
        for j in range(1, max_j + 1):
            out[i].append(augment_pdf(mdb[str(i)][str(j)], i, j))
        out[i] = I.pd.concat(out[i])
    return out, I.pd.concat(out.values())

def get_pdf_selected(pdf, BAC_limit = 3.5, step_limit = 4.5,
                     objectives_BAC = objectives_BAC,
                     objectives_step = objectives_step):
    
    objectives = objectives_BAC + objectives_step
    pdf['sort_column'] = pdf[objectives].max(axis = 1)
    p = pdf[(pdf[objectives].max(axis = 1) < 5) & (pdf[objectives_BAC].max(axis = 1) < 3.5)].sort_values('sort_column').head()
    return p, str(p.index[0])
