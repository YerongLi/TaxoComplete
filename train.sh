# 500 not working, 3000 works
# awk '{ if (NR % 10 == 0) print }' wordnet_noun.taxo.bak > wordnet_noun.taxo

python ./src/train.py --config config_files/semeval_noun/config_clst20_s47.json
awk '{ if (NR % 10 == 0) print }' wordnet_noun.taxo.bak > wordnet_noun.taxo