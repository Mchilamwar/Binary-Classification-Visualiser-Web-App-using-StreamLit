import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("Binary Classification visualiser")
    st.sidebar.title("Model Selection ")

    #Loading and preprocessing the dataset using function
    @st.cache(persist=True)
    def load_data(file):
        les=LabelEncoder()
        df=pd.read_csv(file)
        for col in df.columns:
            df[col]=les.fit_transform(df[col])
        
        return df
    
    #Splitting the data in Train and Test dataset
    @st.cache(persist=True)
    def tsplit(df):
        y=df['type']
        x=df.drop(columns=['type'])
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0) 
        return X_train,X_test,y_train,y_test

    #Defining plotting function 

    def plot(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,X_test,y_test,display_labels=['Edible','Poisonous'])
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC curve")
            plot_roc_curve(model,X_test,y_test)
            st.pyplot()
        
        if 'Precision recall curve' in metrics_list:
            st.subheader("Precision recall curve")
            plot_precision_recall_curve(model,X_test,y_test)
            st.pyplot()
        
 # Upload Load and Preprocess Data
    st.sidebar.subheader("Upload Data")
    fil=st.sidebar.file_uploader("Select/Upload File")
    if fil!=None:
        df=load_data(fil)
        X_train,X_test,y_train,y_test=tsplit(df)
        st.markdown("Here we are coosing Mushrooms dataset to Predict if the mushrooms are poisonous or edible ��🍄 ")


    
    if st.sidebar.checkbox("Show raw data ",value=False):    
        st.subheader("Here is the preprocessed raw dataset :-")
        st.write(df)

    
    # Model Selection, Creation and Implimentation

    st.sidebar.subheader("Select classifier algorithm")
    choose=st.sidebar.selectbox("Classifier",("select classifier","Support Vector Machine (SVM)","Logistic Regression","Random Forest Classifier","Decision Tree Classifier"),index=0)
    
    

    if choose=='Support Vector Machine (SVM)':
        st.sidebar.subheader("Choose Model Hyperparameters")
        C=st.sidebar.number_input("C (regularization parameters)",min_value=0.01,max_value=10.0,step=0.01,key='csvm')
        kernel=st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key='gamma')
        metrics=st.sidebar.multiselect("Which Evaluation Metrics you want ?",("Confusion matrix","ROC Curve","Precision recall curve"))
        
        if st.sidebar.button("Apply Classifier",key='apply'):
            st.subheader("Support Vector Machine (SVM)")
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            score=model.score(X_test,y_test)
            ypred=model.predict(X_test)
            prec=precision_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            rec=recall_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            fsc=f1_score(y_test,ypred)
            st.write("Score :",score.round(2))
            st.write("Precision :",prec)
            st.write("Recall :",rec)
            st.write("F1-Score :",fsc.round(2))
            plot(metrics)

    
    elif choose=='Logistic Regression':
        st.sidebar.subheader("Choose Model Hyperparameters")
        C=st.sidebar.number_input("C (regularization parameters)",min_value=0.01,max_value=10.0,step=0.01,key='clr')
        solv=st.sidebar.radio("Select Solver :",("liblinear","lbfgs","sag","saga"))
        m_iter=st.sidebar.slider("Max Iterations",min_value=100,max_value=1000)
        metrics=st.sidebar.multiselect("Which Evaluation Metrics you want ?",("Confusion matrix","ROC Curve","Precision recall curve"))
        
        if st.sidebar.button("Apply Classifier",key='apply'):
            st.subheader("Logistic Regression Classifier")
            model=LogisticRegression(C=C,solver=solv,max_iter=m_iter)
            model.fit(X_train,y_train)
            score=model.score(X_test,y_test)
            ypred=model.predict(X_test)
            prec=precision_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            rec=recall_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            fsc=f1_score(y_test,ypred)
            st.write("Score :",score.round(2))
            st.write("Precision :",prec)
            st.write("Recall :",rec)
            st.write("F1-Score :",fsc.round(2))
            plot(metrics)

    
    elif choose=='Random Forest Classifier':
        st.sidebar.subheader("Choose Model Hyperparameters")
        n_estim=st.sidebar.slider("Number of estimator trees in Forest :",min_value=100,max_value=500,step=10,key='nestim')
        max_depth=st.sidebar.number_input("Maximum depth",min_value=1,max_value=20,step=1,key='mdepth')
        boot=st.sidebar.radio("Bootstrap samples when building tree ?",("True","False"),key='boots')
        metrics=st.sidebar.multiselect("Which Evaluation Metrics you want ?",("Confusion matrix","ROC Curve","Precision recall curve"))
        
        if st.sidebar.button("Apply Classifier",key='apply'):
            st.subheader("Random Forest Classifier ")
            model=RandomForestClassifier(n_estimators=n_estim,max_depth=max_depth,bootstrap=boot)
            model.fit(X_train,y_train)
            score=model.score(X_test,y_test)
            ypred=model.predict(X_test)
            prec=precision_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            rec=recall_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            fsc=f1_score(y_test,ypred)
            st.write("Score :",score.round(2))
            st.write("Precision :",prec)
            st.write("Recall :",rec)
            st.write("F1-Score :",fsc.round(2))
            plot(metrics)        
    
    elif choose=='Decision Tree Classifier':
        st.sidebar.subheader("Choose Model Hyperparameters")
        max_depth=st.sidebar.number_input("Maximum depth",min_value=1,max_value=20,step=1,key='mdepth')
        cri=st.sidebar.radio("Select the criterion",("gini","entropy"),key='criterion')
        metrics=st.sidebar.multiselect("Which Evaluation Metrics you want ?",("Confusion matrix","ROC Curve","Precision recall curve"))
        
        if st.sidebar.button("Apply Classifier",key='apply'):
            st.subheader("Decision Tree Classifier ")
            model=DecisionTreeClassifier(criterion=cri,max_depth=max_depth)
            model.fit(X_train,y_train)
            score=model.score(X_test,y_test)
            ypred=model.predict(X_test)
            prec=precision_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            rec=recall_score(y_test,ypred,labels=['Edible','Poisonous']).round(2)
            fsc=f1_score(y_test,ypred)
            st.write("Score :",score.round(2))
            st.write("Precision :",prec)
            st.write("Recall :",rec)
            st.write("F1-Score :",fsc.round(2))
            plot(metrics)       









if __name__=='__main__':
    main()