nos features et étoiles:
- Basic RAG (2)
- Advanced RAG (3)
- Persona (1)
- Logging & Monitoring Dashboard (2)
8 étoiles

prerequis éliminatoire

- lancer l'app + envoyer une prompt à quelque'un
- faut faire un fichier de config (il faut pouvoir activer / desactiver des features: rag, tool, memory, etc), adpater la config en fonciton de notre ia
- il faut pouvoir modifier la temperature et la taille du contexte (on doit observé les impactes de ces changements)
- pas d'erreur visible, il faut les interceptées
- il faut pouvoir afficher le prompt reconsituer complet: LOGS
- Compteur de token
- un meilleur pré prompt, potentiellement avoir une IA spécialisée dans un domaine
- mettre des exemples dans le README partout !
- debit par seconde de nos machines, il faut pouvoir le voir, le savoir et l'expliquer
- combien ca prend en RAM et en VRAM
- Prompts de test
- tableau structuré avec les modèles utilisés, faire des benchmarks

Les tests unitaires doivent afficher plus de choses que "OK" (mettre feature ??? 100% ?)
Il faut être capable de présenter 1 test, de montrer et prouver son fonctionnement en temps réel.

trouver les limites de notre modèle avec un prompt, hallicination, etc (essayer le codename, prompt long...)

Avoir un utilisateur cible, un cadre de travail, son besoin métier (persona, contexte précis), est-ce que le scénario est cohérent avec la latence locale

Jeu de données en accord avec le scénario, scénario définit depuis le bootstrap (aie aie aie)

chaque feature a 2 user story, critères d'acceptation
on aura un petit jeu de user story (prendre un user story et trouver la feature d'où il vient)

demo + garder le scénario + faire une scène de théatre (faut vraiment un script et le montrer au jury) les gars c'est sérieux !

il faut faire RGPD, que se passe t il en cas de fuite + quelles sont les contraintes techniques ?

Il faut une analyse coûts-bénéfices (pour les éboueurs et nous)

Try / catch c'est mal, il faut des sécurités dans notre code, savoir montrer où dans le code, on contre les injections de prompt, fuites, etc

ce qu'il reste à faire:

- il faut pouvoir modifier la temperature et la taille du contexte (on doit observé les impactes de ces changements, peut-être les documenter)
- mettre des exemples dans le README partout !
- debit par seconde de nos machines (surement les tokens), il faut pouvoir le voir, le savoir et l'expliquer
- combien ca prend en RAM et en VRAM
- tableau structuré avec les modèles utilisés, faire des benchmarks
- Il faut être capable de présenter 1 test, de montrer et prouver son fonctionnement en temps réel.
- trouver les limites de notre modèle avec un prompt, hallicination, etc (essayer le codename, prompt long...)
- est-ce que le scénario est cohérent avec la latence locale
- chaque feature a 2 user story, critères d'acceptation
on aura un petit jeu de user story (prendre un user story et trouver la feature d'où il vient)
- demo + garder le scénario + faire une scène une mise en scène (le script sera demandé)
- il faut faire RGPD, que se passe t il en cas de fuite + quelles sont les contraintes techniques ?
- Try / catch c'est mal, il faut des sécurités dans notre code, savoir montrer où dans le code, on contre les injections de prompt, fuites, etc
- Il faut une analyse coûts-bénéfices (pour les enseignants et nous)

-------------------------------------------------

7. Model Operation - AI Culture (0 à 4 pts)
Ce que tu demandes :
« C'est quoi un paramètre de modèle dans l'architecture Transformer ? En pratique, quelle différence entre un modèle 7B et un modèle 70B sur nos machines ? » (poids synaptique / possinaptyque)
« Qu'est-ce que la quantification (ex. Q4_K_M vs FP16) ? Quel est son impact sur la mémoire, le débit et la dégradation de qualité ? »
« Pourquoi avoir choisi votre modèle plutôt qu'un modèle d'une autre famille (Llama, Mistral, Qwen, Gemma, Phi) ? »
« Quelle est la différence fondamentale d'architecture et d'usage entre un modèle génératif et un modèle d'embedding ? »
Comment tu valides (1 pt par assertion) :
[ ] Paramètres & échelle : compréhension claire des poids synaptiques, de l'empreinte mémoire théorique (2 octets/param en FP16, etc.) et du compromis vitesse/capacité.
[ ] Quantification : réduction de précision numérique (ex. float32/16 vers int8/int4), gain en bande passante mémoire et perte marginale de perplexité.
[ ] Familles de modèles : distinction lucide des forces respectives (capacités multilingues, raisonnement, support d'outils, context window native).
[ ] Génératif vs Embedding : décodeur autoregressif (génération séquentielle de tokens) vs encodeur/pooling (projection vectorielle dense pour la similarité sémantique).

9. Evaluation - Measured Improvement (0 à 3 pts)
Ce que tu demandes :
« Montrez-moi votre mesure comparative avec et sans une feature donnée (ex. pipeline RAG activé vs désactivé). »
« Quelle métrique quantitative avez-vous utilisée pour mesurer ce gain ? »
« Relancez cette mesure comparative en direct devant moi. »
Comment tu valides (1 pt par assertion) :
[ ] Comparaison avec baseline : test A/B net (avec/sans cache, avec/sans compression, avec/sans RAG).
[ ] Métrique explicite : taux d'exactitude factuelle, score ROUGE/BLEU, score de juge LLM, latence mesurée ou nombre de tokens consommés.
[ ] Rejouabilité : le script de benchmark tourne et reproduit les mêmes ordres de grandeur en direct.

16. Decision-Making - Choice Justification & Alternatives (0 à 3 pts + 0 à 3 pts)
Ce que tu demandes :
« Quelle feature avez-vous délibérément choisi de NE PAS implémenter, et pourquoi sur le plan architectural ? »
« Sur votre décision technique majeure (ex. choix de la base vectorielle, taille de chunk, format de mémoire), quelles alternatives avez-vous éliminées et quels compromis avez-vous arbitrés ? »
« Justifiez vos choix en vous basant uniquement sur vos contraintes réelles (Ollama, Python pur, matériel local). »
« Si vous deviez repartir de zéro demain avec la même équipe et les mêmes contraintes, que changeriez-vous techniquement ? »
Comment tu valides :
Justification (1 pt par assertion, max 3) :
[ ] Scope délimité par le scénario, avec exclusion argumentée d'une feature inutile au besoin.
[ ] Arbitrage documenté entre au moins deux contraintes contradictoires (ex. latence vs précision, granularité de chunk vs coût contextuel).
[ ] Justification ancrée dans les faits du projet (interdiction des frameworks, hardware spécifique) et non dans la facilité (« on connaissait déjà »).
Alternative proposée (1 pt par assertion, max 3) :
[ ] Proposition d'une vraie alternative concrète d'architecture, d'outil ou d'organisation.
[ ] L'alternative résout explicitement un goulot d'étranglement ou un compromis identifié durant le projet.
[ ] L'alternative reste réaliste et exécutable dans les règles du sujet (Ollama local, sans LangChain, hardware du groupe).
