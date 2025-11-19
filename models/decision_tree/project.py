import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
df=pd.read_csv("../../data/titanic.csv")
input=df.drop(['PassengerId','Name','Parch','SibSp','Cabin','Embarked','Survived','Ticket'],axis='columns')
# print(input)             
target=df['Survived']
le_Sex=LabelEncoder()
input['Sex_n']=le_Sex.fit_transform(input['Sex'])
input=input.drop('Sex',axis='columns')
# input.to_csv("cleaned_titanic.csv", index=False)
model=tree.DecisionTreeClassifier()
model.fit(input,target)
print(model.score(input,target))
print(model.predict([[3,22,7.25,1]]))