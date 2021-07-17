# Cargamos las clases de los módulos que creamos:
from utils import Utils
from models import Models

# Es el Script PRINCIPAL
if __name__ == "__main__":
    
    # Inicializamos la clase
    utils = Utils()
    models = Models()

    # Cargamos la info
    data = utils.load_from_csv('./in/income.csv')

    # Mostramos un pedacito de nuestro dataset:
    print( 'Aquí tienes una muestra de tus datos:')
    print(' ')
    print(data.head(5))

    ## Creación de las graficas:
    utils.grafica_barras(data, 'age')
    print('Ya imprimí la imagen :)')

    ## Creación del modelo y evaluación
    dropear = ['fnlwgt','capital-gain','capital-loss','income','native-country','income_bi']
    dummies = ['workclass','education','marital-status','occupation','relationship','race','sex']
    target = 'income_bi'

    X, y = utils.features_target(data, dropear, dummies, target)

    print('Vamos a entrenar el modelo...')

    resultados = models.tree_training(X,y)
    
    print ( " ")
    print(f'El score del modelo en train es:  {resultados[0]}')
    print ( " ")
    print(f'El score del modelo en test es:  {resultados[1]}')
    print ( " ")
    print ( "Ya terminé :) ")