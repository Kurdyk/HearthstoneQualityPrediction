# HearthstoneQualityPrediction
A machine learning project to try to classify Hearthstone cards

## Project organization
The project folders is organized as follows:
- the graphs folder contains graphical vizualisations needed to make the project's report.
- the OldData folder contains legacy data not used anymore and the scripts to parse them.
- the src folder contains the source code of the project as well as the main executable.

## Executing the project
First be sure that you meet the requirements of the requirements.txt file with for example: `conda create --name <env_name> --file requirements.txt`
Then, if you want:
- to run the cross validation on the linear regression : `python3 -m src.Models.regression` (takes some time)
- to run the cross validation on Kmeans : `python3 -m src.Models.k_means`
- to run the grid search that tunes the hyperparameters for a regression with a PCA : `python3 -m src.hyperparameters_tuning`
- to run our recommandatation and grading script : `python3 -m src.recommandation`
- to create you up-to-date dataset : `python3 -m src.webScraperHSTopDecks`

### Remark 
The grading system also works on custom cards, obviously new/random keywords won't give a good predictions.
Some card examples are located in src/card_examples but here are a few format rules:
- name : all lower case, no special caracters, remplace spaces by '-'. This is to match the format of the HSTopdeck.csv file
- card_type, rarity : put a majuscle.
- classes, minion_type : put a majuscle and separates multiples with ','. eg: Paladin,Priest

