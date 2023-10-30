import pandas as pd  # Importamos la biblioteca Pandas para el manejo de datos
import numpy as np  # Importamos la biblioteca NumPy para operaciones matemáticas

class NaiveBayesClassifier:
    def __init__(self):
        # Inicializa diccionarios para almacenar probabilidades
        self.class_probabilities = {}  # Almacenará las probabilidades de clase
        self.feature_probabilities = {}  # Almacenará las probabilidades de atributos categóricos
        self.feature_stats = {}  # Almacenará las estadísticas de atributos numéricos
        self.classes = []  # Almacenará las clases únicas del conjunto de datos

    def load_data(self, filename):# Cargar datos desde un archivo CSV o TXT
        self.data = pd.read_csv(filename)  # Lee el archivo de datos y lo almacena en un DataFrame (Con uso de biblioteca Pandas)

    def split_data(self, test_size=0.3, random_state=None): #dividimos los datos en enrenamiento y prueba (Para prueba es un 30% = 0.3)
        num_instances = len(self.data) # Calcula el número total de instancias en el conjunto de datos.
        num_train = int(num_instances * (1 - test_size))# Calcula el número de instancias de entrenamiento restando el tamaño de prueba.
        train_data = self.data.sample(n=num_train, random_state=random_state) # Obtiene una muestra aleatoria de instancias para el conjunto de entrenamiento.
        test_data = self.data.drop(train_data.index) # Elimina las instancias de entrenamiento del conjunto de datos para obtener el conjunto de prueba.
        return train_data, test_data # Retorna los conjuntos de entrenamiento y prueba resultantes.

    def calculate_prior_probabilities(self, test_data, target_column):# Calcular probabilidades a priori solo en prueba
        class_counts = test_data[target_column].value_counts() # Calcula el recuento de las clases en el conjunto de prueba.
        total_instances = len(test_data) # Calcula el número total de instancias en el conjunto de prueba.
        for cls in class_counts.index: # Itera a través de las clases en el conjunto de prueba.
            self.class_probabilities[cls] = class_counts[cls] / total_instances # Calcula y almacena la probabilidad a priori de cada clase.

    def calculate_feature_statistics(self, train_data):#Calcular estadisticas de los atributos
        # Calcular tablas de frecuencia para atributos categóricos
        # y media y desviación estándar para atributos numéricos
        for column in train_data.columns[:-1]:  # Excluyendo la columna de clase
            if train_data[column].dtype == 'object': # Para atributos categóricos, calcula tablas de frecuencia condicionales
                self.feature_probabilities[column] = train_data.groupby([train_data.columns[-1], column]).size() / train_data.groupby([train_data.columns[-1]]).size()
            else:  # Para atributos numéricos, calcula la media y la desviación estándar condicionales
                self.feature_stats[column] = {
                    'mean': train_data.groupby([train_data.columns[-1]])[column].mean(),#Media
                    'std': train_data.groupby([train_data.columns[-1]])[column].std()#Desviacion Estandar
                }

    #Aqui comenzamos con la implementacion del modelo de Bayes
    def fit(self, train_data, target_column):# Ajusta el modelo con los datos a entrenar
        self.calculate_prior_probabilities(train_data, target_column)  # Calcula las probabilidades a priori basadas en los datos de entrenamiento
        self.calculate_feature_statistics(train_data)# Calcula las estadísticas de los atributos (tablas de frecuencia para atributos categóricos y estadísticas para atributos numéricos)
        self.classes = train_data[target_column].unique()  # Obtiene las clases únicas y las almacena

    def calculate_class_probability(self, instance, cls):
        # Calcula la probabilidad de pertenencia a una clase para una instancia dada
        class_probability = self.class_probabilities[cls]
        # Itera a través de los atributos y sus valores en la instancia
        for feature, value in instance.items():
            # Verifica si el atributo está en el diccionario de probabilidades de atributos
            if feature in self.feature_probabilities:
                # Si el valor está presente en las probabilidades de atributos, multiplícalo
                if value in self.feature_probabilities[feature][cls].index:
                    class_probability *= self.feature_probabilities[feature][cls][value]
                else: # Si el valor no está presente, aplica suavizado Laplaciano
                    class_probability *= 1e-6  # Aplica suavizado Laplaciano para valores no vistos
            else: # Si el atributo es numérico:
                mean = self.feature_stats[feature]['mean'][cls]
                std = self.feature_stats[feature]['std'][cls]

                if std == 0: # Esta línea sigue la fórmula de la distribución normal (Gaussiana)
                    class_probability *= 1e-6  # Aplica suavizado Laplaciano si la desviación estándar es cero
                else:
                    # Calcula la probabilidad usando la distribución normal
                    class_probability *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean) ** 2) / (2 * (std ** 2))) # Esta línea sigue la fórmula de la distribución normal (Gaussiana)

        return class_probability  # Devuelve la probabilidad acumulada

    def classify(self, instance): # Calcula la probabilidad de pertenencia a una clase para una instancia dada
        # Clasifica una instancia en una de las clases
        max_probability = -float("inf") # Inicializa la variable `max_probability` con un valor negativo infinito
        for cls in self.classes:  # Calcula la probabilidad de que la instancia pertenezca a la clase actual
            probability = self.calculate_class_probability(instance, cls) # Calcula la probabilidad de que la instancia pertenezca a la clase actual
            if probability > max_probability: # Si es mayor, actualiza la máxima probabilidad y la clase predicha
                max_probability = probability
                predicted_class = cls
        return predicted_class # Devuelve la clase predicha con la probabilidad máxima

    def evaluate(self, test_data, target_column):
        # Evalúa el modelo en un conjunto de prueba y devuelve la precisión
        correct = 0 # Inicializa una variable `correct` para contar el número de predicciones correctas
        total = len(test_data)  # Obtiene el número total de instancias en el conjunto de prueba
        for _, instance in test_data.iterrows(): # Convierte la instancia a un diccionario sin la columna de etiqueta de clase
            instance_data = instance.drop(target_column).to_dict() # Utiliza la función `classify` para predecir la clase de la instancia
            predicted_class = self.classify(instance_data)  # Utiliza la función `classify` para predecir la clase de la instancia
            if predicted_class == instance[target_column]: # Compara la clase predicha con la etiqueta real de la instancia
                correct += 1  # Si la predicción es correcta, incrementa el contador `correct`
        accuracy = correct / total # Calcula la precisión dividiendo el número de predicciones correctas entre el total
        return accuracy  # Devuelve la precisión calculada

if __name__ == "__main__":
    #Uso
    classifier = NaiveBayesClassifier()

    # Pedir al usuario el nombre del archivo y su extensión
    file_name = input("Por favor, ingrese el nombre del archivo con la extensión (por ejemplo, 'dataset.csv'): ")

    # Cargar datos desde el archivo proporcionado por el usuario
    classifier.load_data(file_name)

    # Pedir al usuario el número de iteraciones
    num_iterations = int(input("Ingrese el número de iteraciones: "))

    # Lista para almacenar las probabilidades finales de cada iteración
    total_probs = {cls: 0 for cls in classifier.classes}

    for i in range(num_iterations):
        print(f"Iteración {i + 1}")

        # Usar split_data para dividir aleatoriamente los datos en conjuntos de entrenamiento y prueba
        train_data, test_data = classifier.split_data(test_size=0.3, random_state=i)

        target_column = train_data.columns[-1]  # Obtener el nombre de la columna de etiquetas de clase dinámicamente

        # calcular probabilidades a priori
        classifier.calculate_prior_probabilities(test_data, target_column)

        # Calcular estadísticas de características
        classifier.calculate_feature_statistics(test_data)

        classifier.fit(train_data, target_column)
        accuracy = classifier.evaluate(test_data, target_column)

        # Clasificar instancia de prueba
        test_instance = test_data.iloc[0].drop(target_column)
        predicted_class = classifier.classify(test_instance)

        # Calcular probabilidades finales
        probabilities = {}
        for cls in classifier.classes:
            prob = classifier.calculate_class_probability(test_instance, cls)
            probabilities[cls] = prob

        # Mostrar los conjuntos de entrenamiento y prueba en forma visual con espacio
        print("\nConjunto de entrenamiento:")
        print("-" * 60)
        for _, row in train_data.iterrows():
            print("|", end=" ")
            for col in train_data.columns:
                print(f"{col}: {row[col]}", end=" | ")
            print("\n" + "-" * 40)

        print("\nConjunto de prueba:")
        print("-" * 60)
        for _, row in test_data.iterrows():
            print("|", end=" ")
            for col in test_data.columns:
                print(f"{col}: {row[col]}", end=" | ")
            print("\n" + "-" * 60)

        # Mostrar las tablas de probabilidad a priori
        print("\nProbabilidad A priori de Prueba:")
        print("+" * 40)
        for cls, prob in classifier.class_probabilities.items():
            print(f"| Clase: {cls:<11} | Probabilidad: {prob:.2f} |")
        print("+" * 40)

        # Tablas de frecuencia
        print("Tablas de frecuencia (Prueba):")
        for feature, table in classifier.feature_probabilities.items():
            print(f"\n{feature}:")
            print("-" * 30)

            # Mostrar la tabla de test_data
            print(table)

            print("-" * 30)

        # Estadísticas
        print("\nEstadísticas (Prueba):")
        for feature, stats in classifier.feature_stats.items():
            print(f"\n{feature}:")
            print("-" * 30)

            # Mostrar estadísticas de test_data
            mean = stats['mean'].map(lambda x: '{:.4f}'.format(x))
            std = stats['std'].map(lambda x: '{:.4f}'.format(x))

            print(f"| Media: {mean} | Desviacion Estandar. Std: {std} |")

            print("-" * 30)

        # Mostrar Probabilidades Finales
        print("\nProbabilidades Finales (Prueba):")
        print("-" * 30)
        print("| Clase        | Probabilidad |")
        print("-" * 30)
        for cls, prob in probabilities.items():
            print(f"| {cls:<12} | {prob:.4f}       |")
        print("-" * 30)

        #Mostramos que tan exacta es la iteracion
        print(f"Exactitud en esta iteración: {accuracy:.2f}\n")

        # Línea divisoria
        print("///" * 50)

        # Imprimir la suma total de todas las probabilidades finales
        for cls, prob in probabilities.items():
            if cls in total_probs:
                total_probs[cls] += prob
            else:
                total_probs[cls] = prob

print("Suma total de probabilidades:")
for cls, total_prob in total_probs.items():
    print(f"{cls}: {total_prob}")
