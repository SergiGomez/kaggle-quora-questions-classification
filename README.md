## Quora Insincere Questions Classification
Build a model that identifies and flags insincere questions

Link to the Kaggle challenge
www.kaggle.com/c/quora-insincere-questions-classification

### Background
An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.

Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.

In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.

### Code structure
Important note:
All the project has been developed in the same script (exec.py) given that it has to be run as a Kernels Only Competition, requiring that all submissions be made via a Kernel output.

- scan_hyperparameters.sh: Bash programs that schedule training executions with different combinations of hyperparameters given, on two GPU's
- init_params.py: Parameters that user should input before executing the main script
- exec.py: script that runs the training and scoring functions and prepares the submission
