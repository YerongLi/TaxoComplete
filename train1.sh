# 500 not working, 3000 works
# awk '{ if (NR % 20 == 0) print }' wordnet_noun.taxo.bak > wordnet_noun.taxo

python ./src/train1.py --config config_files/semeval_noun1/config_700.json
