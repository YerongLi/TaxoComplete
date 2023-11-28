# 500 not working, 3000 works
# awk '{ if (NR % 20 == 0) print }' wordnet_noun.taxo.bak > wordnet_noun.taxo

# python ./src/train1.py --config config_files/semeval_noun1/config_1500.json
# python ./src/train1.py 700 --config config_files/semeval_noun/config_clst20_s47.json
python ./src/train1.py --config config_files/cs/config_clst20_s47.json 500

