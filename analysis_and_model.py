import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    st.header("Загрузка данных")
    data_source = st.radio("Выберите источник данных", ["CSV файл", "UCI Repository"])
    
    data = None
    if data_source == "CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    else:
        if st.button("Загрузить данные из UCI Repository"):
            try:
                dataset = fetch_ucirepo(id=601)
                data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
                st.success("Данные успешно загружены!")
            except Exception as e:
                st.error(f"Ошибка при загрузке данных: {e}")
    
    if data is not None:
        # Предобработка данных
        st.header("Предобработка данных")
        
        # Удаление ненужных столбцов
        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
        
        # Преобразование категориальных переменных
        if 'Type' in data.columns:
            data['Type'] = LabelEncoder().fit_transform(data['Type'])
        
        # Проверка на пропущенные значения
        st.subheader("Проверка на пропущенные значения")
        st.write(data.isnull().sum())
        
        # Масштабирование данных
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        
        # Разделение данных
        st.header("Разделение данных")
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"Обучающая выборка: {X_train.shape[0]} записей")
        st.write(f"Тестовая выборка: {X_test.shape[0]} записей")
        
        # Обучение моделей
        st.header("Обучение моделей")
        model_choice = st.selectbox("Выберите модель", 
                                  ["Logistic Regression", "Random Forest", "XGBoost", "SVM"])
        
        model = None
        if model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif model_choice == "SVM":
            model = SVC(kernel='linear', probability=True, random_state=42)
        
        if model is not None and st.button("Обучить модель"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Оценка модели
            st.header("Оценка модели")
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader(f"Accuracy: {accuracy:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            class_report = classification_report(y_test, y_pred)
            st.text(class_report)
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'{model_choice} (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
            
            # Интерфейс для предсказания
            st.header("Предсказание на новых данных")
            with st.form("prediction_form"):
                st.write("Введите параметры оборудования:")
                
                col1, col2 = st.columns(2)
                with col1:
                    type_ = st.selectbox("Тип оборудования", ["L", "M", "H"])
                    air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                    process_temp = st.number_input("Температура процесса [K]", value=310.0)
                with col2:
                    rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                    torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                    tool_wear = st.number_input("Износ инструмента [min]", value=0)
                
                submit_button = st.form_submit_button("Сделать предсказание")
                
                if submit_button:
                    # Преобразование введенных данных
                    input_data = pd.DataFrame({
                        'Type': [0 if type_ == 'L' else 1 if type_ == 'M' else 2],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotational_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })
                    
                    # Масштабирование
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                    
                    # Предсказание
                    prediction = model.predict(input_data)
                    prediction_proba = model.predict_proba(input_data)[:, 1]
                    
                    st.subheader("Результат предсказания")
                    if prediction[0] == 1:
                        st.error(f"Вероятность отказа: {prediction_proba[0]:.2%} - Ожидается отказ оборудования!")
                    else:
                        st.success(f"Вероятность отказа: {prediction_proba[0]:.2%} - Оборудование в норме")