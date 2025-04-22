# IEEE-CIS Fraud Detection
Competition: https://www.kaggle.com/competitions/ieee-fraud-detection

# Data Inspection
- Inspecting NaNs - NaN-heavy dataset გვაქვს.
- Fraud ტრანზაქციების წილი - 3.5%, რაც ნიშნავს რო imbalanced data გვაქვს.
- `train_transaction` და `train_identity` row-ების რაოდენობით განსხვავდება, შევხედე sample-ებს `TransactionID`-ს გარეშე, რამდენი იყო მათგან Fraud და Non-Fraud.  
~8% ტრანზაქციებისა (out of 144233) რომელსაც `TransactionID` აქვს, არის Fraud.  
~2% ტრანზაქციების (out of 446307) რომელსაც `TransactionID` არ აქვს, არის Fraud.

# Preprocessing

- Handling N/As  
კატეგორიულ NaN-ებს ვანაცვლებ მოდით, რიცხვითს მედიანით.

- Categoric to Numeric Conversions  
კატეგორიულ column-ებს, რომელთა unique value-ების რაოდენობა <= რაღაც `onehot_threshold`-ზე, ვუტარებ One-Hot Encoding-ს, დანარჩენებს WoE Encoding-ს.

დანარჩენი step-ები model-specific არის, ამიტომ კონკრეტულ მოდელებში მაქ დეტალურად აღწერილი რას როგორ ვაკეთებ.

# Training

## Logistic Regression
საწყის ეტაპზე გამოვიყენე ყველაზე მარტივი მოდელი, Logistic Regression, რომელმაც ძალიან ცუდი შედეგი მომცა (classification report დალოგილი მაქვს artifact-ად MLFlow-ზე). მოდელს უჭირდა fraud-ების ამოცნობა (0.16 recall), რასაც ავხსნი იმით რომ fraud vs non-fraud ტრანზაქციები არაა linearly separable, რის გამოც მოდელმა მაღალი bias მომცა და underfitted გაიჩითა.

Logistic Regression-ისთვის არ ჩავთვალე საჭიროდ დამატებითი სამუშაოების ჩატარება (feature engineering, feature selection, hyperparameter tuning, feature importances, etc.), კორელაციის ფილტრიც მხოლოდ WoE column-ებზე მაქვს ჩატარებული რო დრო არ დამეხარჯა, უბრალოდ საწყისად ყველაზე მარტივი მოდელი ავიღე (საჩვენებლად-ish).

## Random Forest
რახან tree-based მოდელის საჭიროება დადგა, მხოლოდ ერთი plain DecisionTreeClassifier-ის გამოყენებას ვამჯობინე RandomForestClassifier გამომეყენებინა პირდაპირ.  

NaN-ების შესავსებად ისევ იგივენაირი მარტივი მიდგომა გამოვიყენე, რადგან წარმოდგენა არ მაქვს როგორ უნდა შევავსო დამასკული feature-ები ისე რომ რაიმე meaning ჩავაქსოვო. ისე გამოდგა, რომ ამ მარტივ filler-ს ვიყენებ ყველა მოდელისთვის.

კორელაციის ფილტრში ამ ეტაპზე როგორც WoE, ასევე ჩვეულებრივ არა-OneHot column-ებიც გავატარე და 219 feature-ს დავემშვიდობე.

მცირედი Feature Engineering ჩავუტარე `TransactionDT` feature-ს. მინიმალური value რომელიც ამ feature-ს აქვს, იყო `86400 = 24*60*60`, რაც ერთ დღეს უდრის, ანუ წამებშია გაზომილი და შინაარსობრივად რაღაც თარიღიდან დაშორებას ასახავს დროში. ასე პროსტა აღებული დროის დაშორება ჩავთვალე რომ არ უნდა მქონოდა dataset-ში და უნდა განმეზოგადებინა, ამიტომ გამოვიყვანე feature-ები, როგორიცაა საათი (დღის რომელ საათში გაკეთდა ტრანზაქცია), კვირის დღე, და weekend-ია თუ არა.

დავამატე Feature Selection, გამოვიყენე `RandomForestClassifier.feature_importances_`. ამ მოდელისთვის არ გამომიყენებია grid search იმიტორო memory limit-ს სცდებოდა და kernel-ს მირესტარტებდა. მხოლოდ pipeline დავა-define-ე და feature selection-სთვის რა პარამეტრებითაც გავუშვი მოდელი, იგივე პარამეტრები შევიტანე საბოლოო მოდელშიც. როგორც სემინარზე გავარჩიეთ, აქაც კუმულაციურად importance-ების 95%-ს რომელი top feature-ებიც ფარავდნენ ისინი დავიტოვე და ისე დავ-fit-ე საბოლოო მოდელი (მხოლოდ 68 feature დაიტოვა).

ამ მოდელმა Recall და area under ROC curve გამიზარდა საკმაოდ.

**Note: რახან პრეპროცესინგის pipeline მოდელებისთვის საერთო მაქვს, როგორც წინა დავალებაში ავტვირთე ცალკე preprocessor pipeline მოდელად, იგივენაირად ვქენი აქაც.**

## XGBoost
ამ მოდელისთვის Cleaning ცალკე run-ად არ მაქვს დალოგილი, იმიტორო არ განსხვავდება წინა მოდელის Cleaning-სგან არაფრით.  
Feature Selection-სთვის სემინარზე განხილული split, gain და cover importance-ები დავითვალე და საბოლოო გადაწყვეტილება gain importance-ებზე შევაჩერე, იმიტორო gain-ის gain-ი არი გადამწყვეტი რის გამოც sample-ების და-split-ვა ხდება ხის კონკრეტულ ლეველზე და მეც ეგ მინდოდა.  
საბოლოოდ მივიღე 177 feature-ის importance. იგივე მოდელი შემდგომ გავტესტე 75, 100, 120 და 177 top feature-ზე, და ამათგან 120-იანზე იმუშავა მოდელმა ყველაზე კარგად (AUC), ამიტომ ის შევარჩიე საბოლოო მოდელად.  
ამ ეტაპზე ჰიპერპარამეტრების ოპტიმიზაციისთვის Grid Search არ გამომიყენებია, არც KFold CV, იმიტორო Kaggle memory error-ს მაძლევდა და კერნელს მითიშავდა.  
Feature Selection-სთვის ვცადე SHAP-ის გამოყენებაც, მარა ძალიან დიდი ხანი ველოდე უშედეგოდ, არ გაჩერებულა და მაგიტო გადავწყვიტე ზემოთ ნახსენები model-specific improtance-ების დათვლა და მაგის მიხედვით გადარჩევა.

ამ მოდელმა მომცა საუკეთესო შედეგი AUC-ზე, ამიტომ registry-შიც ეს მოდელი დავარეგისტრირე და ამაზე გავუშვი inference.

# MLFlow
ექსპერიმენტებად:

- [General Preprocessor](https://dagshub.com/b3tameche/kaggle-fraud-detection.mlflow/#/experiments/3)
- [Logistic Regression](https://dagshub.com/b3tameche/kaggle-fraud-detection.mlflow/#/experiments/1)
- [Random Forest](https://dagshub.com/b3tameche/kaggle-fraud-detection.mlflow/#/experiments/2)
- [XGBoost](https://dagshub.com/b3tameche/kaggle-fraud-detection.mlflow/#/experiments/4)

თითოეული მოდელისთვის მეტრიკები artifact-ებად მაქვს დალოგილი.
