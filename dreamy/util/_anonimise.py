_author_ = "lorenzoscottb" 

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

import pandas as pd
from tqdm import tqdm

ent2lexeme = {
    "PER" : "Person",
    "ORG" : "Organisation",
    "LOC" : "Location",
    "MISC" : "Miscellaneous",
  }

def get_cntextulised_entities(txt_ents):
  cntextulised_ent_dct = {}
  ent_cnt = []
  for ent_dct in txt_ents:
    if ent_dct["word"] not in cntextulised_ent_dct.keys():
      k = ent_cnt.count(ent_dct["entity_group"])
      cntextulised_ent_dct[ent_dct["word"]] = "{}{}".format(
          ent2lexeme[ent_dct["entity_group"]],
          k+1
      )
      ent_cnt.append(ent_dct["entity_group"])

  return cntextulised_ent_dct

def anonimise(dreams_list, mdl_id="Babelscape/wikineural-multilingual-ner", tsk="token-classification", ent2lexem=ent2lexeme, device=0, batch_size=16, return_original=True):

  df = pd.DataFrame(
      dreams_list,
      columns=['dreams']
  )

  HF_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=df))

  pipe = pipeline(
      tsk,
      model=mdl_id,
      aggregation_strategy="simple",
      device=device,
  )

  print('Annotating reports...')
  annotated_drms = [
      out for out in tqdm(pipe(KeyDataset(HF_dataset, "dreams"), batch_size=batch_size))
  ]

  print('Editing reports...')
  anon_dreams_list = []
  for drm_idx, txt_ents in enumerate(tqdm(annotated_drms)):
      text = dreams_list[drm_idx]
      anon_txt = text
      cntextulised_ent_dct = get_cntextulised_entities(txt_ents)
      for wrds, cntx_ent in cntextulised_ent_dct.items():
        anon_txt = anon_txt.replace(wrds, cntx_ent)

      anon_dreams_list.append(anon_txt)

  if return_original:
    return anon_dreams_list, dreams_list

  else:
    return anon_dreams_listlst
