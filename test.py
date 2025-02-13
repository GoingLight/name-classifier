import joblib
import pickle

with open("name_exceptions.pkl", "rb") as f:
    exception_corrections = pickle.load(f)

def predict_gender(name):
    name = name.lower()

    if name in exception_corrections:
        return exception_corrections[name]

    if name.endswith("хон"):
        root_name = name[:-3]
        return predict_gender(root_name)

    if name.endswith("хужа"):
        root_name = name[:-4]
        return predict_gender(root_name)

    if name.endswith("хўжа"):
        root_name = name[:-4]
        return predict_gender(root_name)

    model = joblib.load("gender_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    name_tfidf = vectorizer.transform([name])
    prediction = model.predict(name_tfidf)[0]

    return "Мужчина" if prediction == 1 else "Женщина"

while True:
    name = input("Введите имя (или 'exit' для выхода): ")
    if name.lower() == "exit":
        break
    print(predict_gender(name))
