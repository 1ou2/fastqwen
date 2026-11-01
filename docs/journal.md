# h100
le but est de générer les données pour le finetuning. 
Pour cela on utilise une base de prompts du dataset ultrachat.
Ces prompts sont séparés en 4 catégories  (shards)
- S1 : 
- S2 : code

Mise en place d'un decision model pour classifier les prompts entre ces catégories mais aussi pour indiquer si on va avoir une réponse trop longue.
