from sklearn import svm

# Support Vector Machine it's very benefical classification with high dimensional space

# Bear or panda dataset

# with given value height, width, and foot to recognise a bear or panda
x = [[188,100,25],[178,100,20],[189,110,20],[180,100,20],[179,98,25]]
y = ['bear','panda','bear','panda','panda']

clf = svm.SVC()
clf.fit(x,y)

print(clf.predict([[180,100,25]]))