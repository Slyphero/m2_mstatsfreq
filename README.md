# Rendu de projet 

## Installation
Il faut générer un environnement virtuel (commandes à adapter selon le système d'exploitation)
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
python main_<le_main_que_vous_souhaitez>.py
```
## Synthèse de textures

**Algorithme utilisé** : Efros Leung  

En ce qui concerne le paramètre epsilon, c'est un compromis entre variété et fidélité locale.
- Un epsilon proche de 0 aura une meilleure fidélité locale mais pourrait induire plus de répétition, comme si on "recopiait" la texture. 
- Un epsilon plus grand acceptera plus de candidats, aura une meilleure variété mais pourra induire plus de bruit.

En ce qui concerne la taille des patch, c'est aussi un compromis entre plusieurs points : 
- Vitesse de calcul 
- Nombre de candidats
- Préservation des motifs principaux / structures

Un patch plus grand préservera mieux les grandes structures en étant plus long à calculer, un patch plus petit sera plus rapide, autorisera beaucoup de candidats mais il préservera moins les motifs plus larges et pourrait sembler plus bruité.  

## Classification

**Algorithme** : K-Means  
**Descripteur** : Gabor  
**Déformations testées** : bruit et flou 

On constate que le classificateur est moins efficace sur les textures moins uniformes.  
On constate que les images synthétisées sont classées comme les vrais patches. Ainsi, on peut conclure que la synthèse préserve les statistques captées par le descripteur.

Le flou est plus destructeur car en lissatn, on perd l'information des hautes fréquences que les filtres utilisent.

